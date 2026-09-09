"""Session management with context caching for OpenAI Realtime API."""
import logging
import time
from typing import Optional, Dict
from pipecat.processors.aggregators.llm_context import LLMContext

from app.context_restore import _strip_tool_plumbing, restore_context_silently
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, StartFrame

logger = logging.getLogger(__name__)


class ContextCacheEntry:
    """Entry in the context cache for a specific client."""
    
    def __init__(self, context: LLMContext, timestamp: float):
        self.context = context
        self.timestamp = timestamp


class SessionManager:
    """Manages OpenAI Realtime sessions with context caching per client device.
    
    For each new WebSocket connection, a new session is created, but the context
    from previous sessions for the same client is preserved if the last connection
    closed within the reuse timeout period.
    """
    
    def __init__(self, reuse_timeout: float = 300.0, max_restored_messages: int = 0):
        """Initialize session manager.

        Args:
            reuse_timeout: Time in seconds after which cached context expires
            max_restored_messages: Cap on how many of the most-recent cached
                messages are restored into a new session (0 = unlimited). The
                OpenAI Realtime conversation grows server-side and pipecat 0.0.97
                has no truncation, so every response.create re-bills the whole
                history (audio transcripts + tool results). The device reconnects
                often (follow-up windows, keepalive drops), and each reconnect
                restores the cached context — so capping it here bounds the
                per-turn token cost (and the rate-limit risk) without losing
                recent conversational continuity. A leading system message, if
                present, is always kept.
        """
        self.reuse_timeout = reuse_timeout
        self.max_restored_messages = max(0, int(max_restored_messages))
        # Dictionary mapping client_id to ContextCacheEntry
        self.context_caches: Dict[str, ContextCacheEntry] = {}
        # Dictionary mapping client_id to current service
        self.current_services: Dict[str, OpenAIRealtimeLLMService] = {}
        # Dictionary mapping client_id to context aggregator pair
        self.context_aggregators: Dict[str, LLMContextAggregatorPair] = {}
    
    def get_cached_context(self, client_id: str) -> Optional[LLMContext]:
        """Get cached context for a specific client if it's still valid.
        
        Args:
            client_id: Unique identifier for the client device
            
        Returns:
            Cached LLMContext if valid, None otherwise
        """
        if client_id not in self.context_caches:
            return None
        
        cache_entry = self.context_caches[client_id]
        time_since_cache = time.time() - cache_entry.timestamp
        
        if time_since_cache < self.reuse_timeout:
            logger.info(f"♻️ Using cached context for client {client_id} from {time_since_cache:.1f}s ago")
            return cache_entry.context
        else:
            # Cache expired
            logger.info(f"⏰ Context cache expired for client {client_id} ({time_since_cache:.1f}s ago, timeout: {self.reuse_timeout}s)")
            del self.context_caches[client_id]
            return None
    
    def cache_context_from_service(self, client_id: str, service: OpenAIRealtimeLLMService):
        """Extract and cache context from a service before it's closed.
        
        Args:
            client_id: Unique identifier for the client device
            service: The OpenAI Realtime service to extract context from
        """
        # First try to get context from the context aggregator (more reliable)
        context = None
        if client_id in self.context_aggregators:
            aggregator_pair = self.context_aggregators[client_id]
            # The context is shared between user and assistant aggregators
            user_aggregator = aggregator_pair.user()
            if hasattr(user_aggregator, '_context') and user_aggregator._context:
                context = user_aggregator._context
                logger.debug(f"🔍 Found context in aggregator for client {client_id}")
        
        # Fallback: try to get context from service
        if not context and service and hasattr(service, '_context') and service._context:
            context = service._context
            logger.debug(f"🔍 Found context in service for client {client_id}")
        
        # Cache the context if we found one
        if context:
            messages = context.get_messages() if hasattr(context, 'get_messages') else []
            message_count = len(messages) if messages else 0
            self.context_caches[client_id] = ContextCacheEntry(
                context=context,
                timestamp=time.time()
            )
            logger.info(f"💾 Cached context from previous session for client {client_id} ({message_count} messages)")
        else:
            if not service:
                logger.warning(f"⚠️ No service provided to cache context for client {client_id}")
            elif client_id not in self.context_aggregators:
                logger.warning(f"⚠️ No context aggregator found for client {client_id}")
            elif not hasattr(service, '_context'):
                logger.warning(f"⚠️ Service has no '_context' attribute for client {client_id}")
            elif not service._context:
                logger.warning(f"⚠️ Service context is None for client {client_id}")
            else:
                logger.debug(f"No context to cache from service for client {client_id}")
    
    def create_context_for_new_session(self, client_id: str) -> LLMContext:
        """Create a new context for a new session, reusing cached context if available.
        
        Args:
            client_id: Unique identifier for the client device
            
        Returns:
            LLMContext for the new session (cached or new)
        """
        # Log available cache keys for debugging
        if self.context_caches:
            logger.debug(f"🔍 Available cached contexts: {list(self.context_caches.keys())}")
        else:
            logger.debug("🔍 No cached contexts available")
        
        cached_context = self.get_cached_context(client_id)
        if cached_context:
            # Create a new context instance with the same messages
            # Use the constructor to properly copy messages and tools
            cached_messages = cached_context.get_messages()
            restore_messages = cached_messages.copy() if cached_messages else None
            # Strip tool calls/results before trimming: their ids belong to the
            # previous session and the server rejects the whole context if they
            # come back. See _strip_tool_plumbing above.
            before = len(restore_messages) if restore_messages else 0
            restore_messages = _strip_tool_plumbing(restore_messages)
            after = len(restore_messages) if restore_messages else 0
            if before != after:
                logger.info(
                    f"🧹 Dropped {before - after} tool message(s) from restored "
                    f"context for client {client_id} (stale tool call ids)"
                )
            # Cap the restored history to the most-recent N messages so the
            # per-turn token cost stays bounded (see __init__ docstring). Keep a
            # leading system message if there is one, then the last N of the rest.
            if restore_messages and self.max_restored_messages > 0 and \
                    len(restore_messages) > self.max_restored_messages:
                head = []
                body = restore_messages
                if isinstance(restore_messages[0], dict) and restore_messages[0].get("role") == "system":
                    head = [restore_messages[0]]
                    body = restore_messages[1:]
                trimmed = head + body[-self.max_restored_messages:]
                logger.info(
                    f"✂️ Trimmed restored context for client {client_id}: "
                    f"{len(restore_messages)} → {len(trimmed)} messages (cap {self.max_restored_messages})"
                )
                restore_messages = trimmed
            new_context = LLMContext(
                messages=restore_messages,
                tools=cached_context.tools if hasattr(cached_context, 'tools') else None,
                tool_choice=cached_context.tool_choice if hasattr(cached_context, 'tool_choice') else None
            )
            logger.info(f"✅ Created new context for client {client_id} with {len(new_context.get_messages())} messages from cache")
            return new_context
        else:
            logger.info(f"🆕 Creating new empty context for client {client_id}")
            return LLMContext()
    
    def get_current_service(self, client_id: str) -> Optional[OpenAIRealtimeLLMService]:
        """Get current OpenAI service for a specific client.
        
        Args:
            client_id: Unique identifier for the client device
            
        Returns:
            Current OpenAIRealtimeLLMService if exists, None otherwise
        """
        return self.current_services.get(client_id)
    
    def set_current_service(self, client_id: str, service: OpenAIRealtimeLLMService):
        """Set the current active service for a client.
        
        Args:
            client_id: Unique identifier for the client device
            service: The currently active OpenAI Realtime service
        """
        self.current_services[client_id] = service
    
    def set_context_aggregator(self, client_id: str, aggregator_pair: LLMContextAggregatorPair):
        """Set the context aggregator pair for a client.
        
        Args:
            client_id: Unique identifier for the client device
            aggregator_pair: The LLMContextAggregatorPair instance for this client
        """
        self.context_aggregators[client_id] = aggregator_pair
    
    def remove_context_aggregator(self, client_id: str):
        """Remove the context aggregator pair for a client.
        
        Args:
            client_id: Unique identifier for the client device
        """
        if client_id in self.context_aggregators:
            del self.context_aggregators[client_id]
    
    def cleanup_before_new_session(self, client_id: str):
        """Cleanup before creating a new session for a client.
        
        This should be called before creating a new session to cache
        the context from the current service.
        
        Args:
            client_id: Unique identifier for the client device
        """
        # Cache context from service/aggregator
        if client_id in self.current_services:
            self.cache_context_from_service(client_id, self.current_services[client_id])
            del self.current_services[client_id]
        
        # Remove context aggregator (will be recreated for new session)
        self.remove_context_aggregator(client_id)
    
    def create_context_aggregator(self, client_id: str) -> LLMContextAggregatorPair:
        """Create a context aggregator pair for a new session.
        
        Args:
            client_id: Unique identifier for the client device
            
        Returns:
            LLMContextAggregatorPair with cached or new context
        """
        context = self.create_context_for_new_session(client_id)
        aggregator_pair = LLMContextAggregatorPair(context)
        self.set_context_aggregator(client_id, aggregator_pair)
        return aggregator_pair
    
    def create_context_initializer(
        self,
        client_id: str,
        context_aggregator: LLMContextAggregatorPair,
        service,
        provider: str,
    ) -> Optional['ContextInitializer']:
        """Create a context initializer if cached messages exist.

        Args:
            client_id: Unique identifier for the client device
            context_aggregator: The context aggregator pair (used only to read
                the already-restored context; see ContextInitializer's
                docstring for why delivery no longer goes through it)
            service: The connection's live LLM service (OpenAI or Gemini)
            provider: "openai" or "gemini" -- selects the delivery channel

        Returns:
            ContextInitializer if cached messages exist, None otherwise
        """
        context = context_aggregator.user().context
        if len(context.get_messages()) > 0:
            return ContextInitializer(
                cached_context=context,
                client_id=client_id,
                service=service,
                provider=provider,
            )
        return None
    
    def handle_client_disconnect(self, client_id: str, service: OpenAIRealtimeLLMService) -> None:
        """Handle client disconnection by caching context.
        
        Args:
            client_id: Unique identifier for the client device
            service: The departing connection's service.
        """
        if self.current_services.get(client_id) is not service:
            logger.debug(f"Ignoring stale disconnect for client {client_id}")
            return

        logger.info(f"🔌 Client {client_id} disconnected - caching context")
        try:
            self.cache_context_from_service(client_id, service)
            logger.info(f"💾 Cached context for disconnected client {client_id}")
        except Exception as e:
            logger.exception(f"Error caching context for disconnected client {client_id}: {e}")
        finally:
            if self.current_services.get(client_id) is service:
                del self.current_services[client_id]
                self.remove_context_aggregator(client_id)


class ContextInitializer(FrameProcessor):
    """Processor that restores cached context after StartFrame passes through
    the pipeline.

    FIX (Task 9b): the previous version pushed an `LLMMessagesUpdateFrame` at
    `context_aggregator.user()` via `push_frame`. That never worked, on
    either engine, possibly ever: `FrameProcessor.push_frame` forwards a
    frame to whatever is linked next in the pipeline -- it does NOT invoke
    the target processor's own `process_frame`. So the frame sailed straight
    past the aggregator's `_handle_llm_messages_update` (which is where
    `set_messages`/REPLACE semantics and the run_llm=False no-response
    guarantee actually live) and landed, unhandled, at the LLM service.
    Neither `OpenAIRealtimeLLMService` nor `GeminiLiveLLMService` has a
    branch for `LLMMessagesUpdateFrame` in `process_frame`, so it fell
    through to the default "forward it on" case and vanished. The log line
    that used to fire here (`📤 Sent cached context...`) printed
    unconditionally, which is exactly why nobody noticed.

    This is the same defect class Task 6b hit and fixed for a different
    feature (telling the model who is speaking) -- see `make_speaker_note`
    and `_send_gemini_note_silently` in websocket_handler.py. The fix here
    follows the same shape: talk to each engine's own session directly
    (`app.context_restore.restore_context_silently`) instead of routing
    through pipecat's frame system, which this specific delivery -- from
    outside the pipeline's own frame flow, before any real turn has
    happened -- was never able to reach.

    Note that `context_aggregator.user().context` is frequently the SAME
    `LLMContext` object as `cached_context` (SessionManager pre-seeds a new
    session's aggregator with the restored messages at construction time),
    so this class no longer needs a reference to the aggregator at all --
    only the messages themselves and the live service to deliver them to.
    """

    def __init__(self, cached_context, client_id, service, provider, **kwargs):
        super().__init__(**kwargs)
        self.cached_context = cached_context
        self.client_id = client_id
        self.service = service
        self.provider = provider
        self.context_sent = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and restore cached context after StartFrame."""
        if isinstance(frame, StartFrame):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)

            # Restore cached context after StartFrame has passed through the
            # pipeline, directly on the engine's own session -- never
            # triggering a response. The bot waits for the user to speak.
            if self.cached_context and not self.context_sent:
                messages = self.cached_context.get_messages()
                if len(messages) > 0:
                    self.context_sent = True
                    try:
                        restored = await restore_context_silently(
                            self.provider, self.service, messages
                        )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to restore cached context for client "
                            f"{self.client_id}: {e!r}"
                        )
                        restored = False
                    if restored:
                        logger.info(
                            f"📤 Restored cached context ({len(messages)} "
                            f"messages) for client {self.client_id} (waiting for user)"
                        )
                    else:
                        logger.info(
                            f"Cached context for client {self.client_id} was "
                            f"not restored (engine not connected yet, or "
                            f"nothing to send)"
                        )
            return

        await self.push_frame(frame, direction)
