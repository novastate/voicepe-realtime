"""Helpers for reusing a conversation across Realtime sessions."""

import logging

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.realtime import events as openai_rt_events

from app.providers import supports_client_events

logger = logging.getLogger(__name__)


def _strip_tool_plumbing(messages):
    """Drop tool calls and tool results from a restored conversation.

    Tool call IDs belong to the OpenAI Realtime *conversation* that produced
    them. Replaying them into a fresh session makes the server reject the
    whole context with:

        invalid_request_error / invalid_tool_call_id
        "Tool call ID 'call_...' not found in conversation."

    The receive loop then closes and the turn dies silently — the user asks a
    question and simply gets no answer. Trimming to the last N messages makes
    it worse, because it can also orphan a tool result from its call.

    What the user actually wants back is what was *said*, not the plumbing, so
    keep the plain turns and drop the rest.
    """
    if not messages:
        return messages
    cleaned = []
    for message in messages:
        if not isinstance(message, dict):
            cleaned.append(message)
            continue
        # A tool result, or anything still pointing at a previous call id.
        if message.get("role") == "tool" or message.get("tool_call_id"):
            continue
        if message.get("tool_calls") or message.get("function_call"):
            # Keep whatever the assistant said out loud, drop the call itself.
            message = {
                key: value
                for key, value in message.items()
                if key not in ("tool_calls", "function_call")
            }
            if not message.get("content"):
                continue
        cleaned.append(message)
    return cleaned


def _cap_restored_messages(messages, max_messages):
    """Cap a restored/reseeded conversation to the most-recent N messages,
    keeping a leading system message if present.

    Deliberately shared by `SessionManager.create_context_for_new_session`
    (client-reconnect restore) and `SafeRealtimeLLMService`'s
    `_reseed_context_after_reset` (60-minute-cap / keepalive-death reseed) --
    both exist for the identical reason: OpenAI's Realtime adapter packs a
    multi-message conversation into ONE synthetic item that gets re-billed
    in full on every turn, and pipecat 0.0.97 has no server-side truncation,
    so an unbounded restored/reseeded history means unbounded per-turn cost
    and rate-limit risk. Both call sites resolve `max_messages` from the
    SAME `MAX_CONTEXT_MESSAGES` setting (default 12) so the bound holds no
    matter which path last touched the conversation.

    `max_messages <= 0` means unlimited (matches SessionManager's own
    convention). Returns `messages` unchanged if there's nothing to trim.
    """
    if not messages or max_messages <= 0 or len(messages) <= max_messages:
        return messages
    head = []
    body = messages
    if isinstance(messages[0], dict) and messages[0].get("role") == "system":
        head = [messages[0]]
        body = messages[1:]
    return head + body[-max_messages:]


def _missing_attrs(service, *names):
    """Return the subset of `names` this service instance does NOT have.

    Used to name exactly which structural-guard attribute vanished, rather
    than unconditionally naming all of them regardless of which is actually
    missing -- a message that always lists every candidate can't be told
    apart from one that's actually diagnosing the real gap.
    """
    return [name for name in names if not hasattr(service, name)]


async def _seed_openai_context_silently(service, messages) -> bool:
    """Seed a real OpenAI Realtime session with a restored conversation.

    FIX (Task 9b, same defect class as Task 6b's speaker note): pushing an
    `LLMMessagesUpdateFrame` at `context_aggregator.user()` never reaches
    anything. `FrameProcessor.push_frame` forwards the frame to whatever is
    linked next (the LLM service) instead of into the aggregator's own
    `process_frame`, and `OpenAIRealtimeLLMService.process_frame` has no
    branch for that frame type anyway -- it just forwards it on unchanged.
    See `ContextInitializer` in session_manager.py for the full trace.

    This mirrors `OpenAIRealtimeLLMService._create_response`'s own "Send
    initial messages" loop -- the exact, already-shipped mechanism that
    primes a brand-new session's conversation before its first
    `response.create` -- just triggered here, proactively, right after
    connect instead of lazily on the aggregator's first completed turn.

    That timing is the whole point: with `semantic_vad_create_response=True`
    the SERVER, not pipecat, decides when to auto-generate a reply, from the
    raw audio it is already receiving the moment the pipeline starts. If the
    restored history is only sent once pipecat's own aggregator finishes a
    completed utterance (the previous, broken behaviour's eventual fallback
    path), the server can auto-reply to the very first thing the user says
    with an EMPTY conversation, because the restore lost the race against
    the user's own voice. Sending it here, before any audio flows, removes
    that race entirely.

    `service._llm_needs_conversation_setup` is cleared afterwards so
    `_create_response()`'s copy of this exact loop does not run again (and
    resend the very same items) the first time a real turn completes.
    `service._messages_added_manually` is populated exactly as
    `_create_response()` populates it, so the server's echoed
    `conversation.item.added` for this item is not mistaken for a new
    assistant turn starting (see `_handle_evt_conversation_item_added`).

    KNOWN, HARMLESS GAP: unlike `_create_response()`'s own conversation-setup
    block, this does not also call `_update_settings()`. That call is the
    only place a leading "system" message *in the restored context* would
    become the live session's `instructions` (OpenAI's adapter pulls a
    leading system message out of the messages list into a separate
    `system_instruction` value -- see `open_ai_realtime_adapter.py` -- which
    `_create_response()` never reads either, so this isn't a regression).
    Session instructions come from `ProviderOptions.instructions` via
    `_handle_evt_session_created` on every connection regardless, so this is
    harmless today. It would stop being harmless if this code ever needed to
    honor a *different* instruction string coming back from a restored
    conversation specifically.

    FIX (final review): this used to send as soon as `_websocket` existed.
    An open socket is NOT a ready session. `ContextInitializer` is the LAST
    stage in the pipeline, so its StartFrame arrives immediately after
    `OpenAIRealtimeLLMService.start()` -> `_connect()`: the socket is open,
    `session.created` has not been read yet, and `_api_session_ready` is
    still False -- so on every reconnect that has history, these
    `conversation.item.create` events raced ahead of the add-on's own
    `session.update`. If OpenAI answers one of them with an error event,
    pipecat's `_receive_task_handler` treats it as fatal
    (`_handle_evt_error(evt); return`), the reader task dies, and the device
    is deaf until something else reconnects it.

    Deferring instead of sending reuses -- deliberately, rather than
    duplicating -- the machinery `SafeRealtimeLLMService._reseed_context_after_reset`
    already uses for the identical question: stash the prepared messages on
    `service._pending_reseed_messages`, and let that class's
    `_handle_evt_session_updated` override deliver them the moment
    `_api_session_ready` is confirmed True. One stash, one drain, one
    ordering guarantee for both paths; a second, subtly different deferral
    would be two mechanisms doing one job.

    Returns True if something was actually sent, False if it was deferred
    (session not ready yet) or skipped (nothing to send, no socket). If the
    pipecat internals this depends on have changed shape, that is logged
    loudly and the restore is skipped rather than guessed at -- the same
    structural-guard pattern `_send_gemini_note_silently` uses for its own
    bridge.
    """
    if not messages:
        return False

    missing = _missing_attrs(
        service,
        "_llm_needs_conversation_setup",
        "_messages_added_manually",
        # The deferral above depends on both of these: the readiness signal
        # itself, and the stash `_handle_evt_session_updated` drains. Without
        # the stash (a service that is not a SafeRealtimeLLMService) a
        # deferred restore would be silently dropped forever, which is worse
        # than not restoring at all -- so name it and skip.
        "_api_session_ready",
        "_pending_reseed_messages",
    )
    if missing:
        logger.error(
            f"⚠️ {type(service).__name__} is missing "
            f"{', '.join(f'`{m}`' for m in missing)} -- the conversation-setup "
            f"bookkeeping this context restore depends on is gone (renamed or "
            f"removed upstream). Skipping context restore rather than "
            f"guessing at a replacement."
        )
        return False

    if not service._api_session_ready:
        # Defer, do not drop: hand the messages to the SAME stash the
        # post-reset re-seed uses (see the docstring above). They go out from
        # `_handle_evt_session_updated`, after our `session.update` has
        # round-tripped, never before it. A normal, expected state right
        # after connect -- so no warning, just a False return the caller
        # logs honestly.
        service._pending_reseed_messages = list(messages)
        return False

    if getattr(service, "_websocket", None) is None:
        # Session claims ready but there is no socket to send on. Not the
        # startup race above (that is caught by the readiness check) --
        # nothing to seed. Not an error.
        return False

    context = LLMContext(messages=list(messages))
    adapter = service.get_llm_adapter()
    items = adapter.get_llm_invocation_params(context).get("messages", [])
    if not items:
        return False

    # NOTE: when this is invoked from ContextInitializer on a brand-new
    # client connection, WebSocketHandler._preseed_context has ALREADY set
    # `_llm_needs_conversation_setup = False` (its own, separate guard
    # against a spontaneous greeting on connect -- see main.py) before this
    # function ever runs. So the assignment below is a no-op in that path
    # today, not a bug: it exists to also cover reset_conversation()'s
    # re-seed (see openai_realtime.py), which runs *without* going through
    # _preseed_context and would otherwise leave the flag True, letting
    # _create_response() resend this same conversation a second time.
    for item in items:
        evt = openai_rt_events.ConversationItemCreateEvent(item=item)
        service._messages_added_manually[evt.item.id] = True
        await service.send_client_event(evt)

    service._llm_needs_conversation_setup = False
    return True


async def _seed_gemini_context_silently(service, messages) -> bool:
    """Seed a real Gemini Live session with a restored conversation.

    Same shape as `_send_gemini_note_silently` in websocket_handler.py, which
    solved the identical delivery problem for a single speaker-note message:
    `send_client_content(turn_complete=False)` loads the turns without an
    immediate reply, and setting `_needs_turn_complete_message = True` hands
    off to the service's own already-wired `_handle_user_stopped_speaking`,
    which silently closes the turn the next time the user's real speech ends
    -- see that function's docstring for why this bridge is safe to set from
    outside (a plain instance-attribute write, not a hook or monkeypatch).

    Structural guard is identical: if a future pipecat rename or removal
    drops the flag, log loudly and skip rather than guess at a replacement.

    FIX ROUND 1 (review): this used to stop at the silent send, leaving
    `service._context` untouched (still None on a fresh connection). The
    very first real `LLMContextFrame` -- from the user's actual first
    utterance -- then hit `GeminiLiveLLMService._handle_context`'s
    `if not self._context:` branch, which calls `_create_initial_response()`
    and RE-SENDS the whole restored history a second time. Worse:
    `inference_on_context_initialization` defaults to True and
    `app/providers/gemini_live.py` never overrides it, so that second send
    goes out with `turn_complete=True` -- an audible, unprompted monologue
    reciting the restored conversation the moment the device reconnects.
    This is exactly the failure Task 6b's three fix rounds were about.

    Setting `service._context = LLMContext()` here (mirroring
    `WebSocketHandler._preseed_context`'s identical trick on the OpenAI side)
    makes that first real context frame take `_handle_context`'s `else`
    branch instead -- which only replaces `self._context` and forwards any
    newly-completed tool results, never calls `_create_initial_response()`,
    and never sends anything on its own. Empty (not the restored messages)
    is deliberate and sufficient: the `else` branch unconditionally
    overwrites `self._context` with whatever context frame it receives, so
    there is nothing to preserve here -- only the branch taken matters.

    Returns True if something was actually sent, False if skipped (no
    session yet, or nothing to send).
    """
    if not messages:
        return False

    missing = _missing_attrs(service, "_needs_turn_complete_message", "_context")
    if missing:
        logger.error(
            f"⚠️ {type(service).__name__} is missing "
            f"{', '.join(f'`{m}`' for m in missing)} -- the silent turn-close "
            f"bridge and/or the first-context guard this context restore "
            f"depends on are gone (renamed or removed upstream). Skipping "
            f"context restore rather than guessing at a replacement."
        )
        return False

    session = getattr(service, "_session", None)
    if session is None:
        return False  # Not connected yet -- nothing to seed. Not an error.

    context = LLMContext(messages=list(messages))
    adapter = service.get_llm_adapter()
    turns = adapter.get_llm_invocation_params(context).get("messages", [])
    if not turns:
        return False

    await session.send_client_content(turns=turns, turn_complete=False)
    service._needs_turn_complete_message = True
    # Stop the first REAL turn's context frame from re-triggering
    # _create_initial_response() and resending (audibly) what we just sent
    # silently -- see the guard explanation above.
    if service._context is None:
        service._context = LLMContext()
    return True


async def restore_context_silently(provider, service, messages) -> bool:
    """Restore a cached conversation into a live session without starting a
    turn -- REPLACE semantics (this IS the whole restored conversation, not
    an addition to whatever the fresh session already has, which is nothing).

    Picks the engine-specific channel proven in Task 6b for the same class of
    problem: a pipecat frame pushed at an aggregator never reaches the
    engine's own `process_frame` (see `ContextInitializer`'s docstring in
    session_manager.py). Returns True if the restore actually reached the
    engine, False if it was skipped (not yet connected, or nothing to send).
    """
    if supports_client_events(provider):
        return await _seed_openai_context_silently(service, messages)
    return await _seed_gemini_context_silently(service, messages)
