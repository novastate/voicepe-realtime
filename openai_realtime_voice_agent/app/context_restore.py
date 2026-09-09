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

    Returns True if something was actually sent, False if skipped (no
    connection yet, or nothing to send). If the pipecat internals this
    depends on have changed shape, that is logged loudly and the restore is
    skipped rather than guessed at -- the same structural-guard pattern
    `_send_gemini_note_silently` uses for its own bridge.
    """
    if not messages:
        return False

    if not hasattr(service, "_llm_needs_conversation_setup") or not hasattr(
        service, "_messages_added_manually"
    ):
        logger.error(
            f"⚠️ {type(service).__name__} is missing the conversation-setup "
            f"bookkeeping (`_llm_needs_conversation_setup` / "
            f"`_messages_added_manually`) this context restore depends on -- "
            f"renamed or removed upstream. Skipping context restore rather "
            f"than guessing at a replacement."
        )
        return False

    if getattr(service, "_websocket", None) is None:
        return False  # Not connected yet -- nothing to seed. Not an error.

    context = LLMContext(messages=list(messages))
    adapter = service.get_llm_adapter()
    items = adapter.get_llm_invocation_params(context).get("messages", [])
    if not items:
        return False

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

    Returns True if something was actually sent, False if skipped (no
    session yet, or nothing to send).
    """
    if not messages:
        return False

    if not hasattr(service, "_needs_turn_complete_message"):
        logger.error(
            f"⚠️ {type(service).__name__} has no `_needs_turn_complete_message` "
            f"attribute -- the silent turn-close bridge this context restore "
            f"depends on is gone (renamed or removed upstream). Skipping "
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
