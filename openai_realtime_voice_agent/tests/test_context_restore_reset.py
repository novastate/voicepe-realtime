"""The hourly hole (review Fix 2): `SafeRealtimeLLMService.reset_conversation`
-- used by ConnectionRecovery for the 60-minute OpenAI Realtime session cap,
and for keepalive-death repair -- reconnects the SAME service instance in
place, without a new `StartFrame`. `ContextInitializer` only ever runs once,
on a brand-new client WebSocket connection, so it never re-fires here. The
old code cleared `_llm_needs_conversation_setup` (to stop a startup-style
double-response) but never re-sent `self._context`'s own messages onto the
freshly reconnected OpenAI-side session, which starts genuinely empty. A
comment on `reset_conversation` claimed "the live context is untouched (it's
restored by the SessionManager on the next real turn)" -- false: the
SessionManager's restore path doesn't apply to an in-place reconnect at all,
and clearing the setup flag actively prevents `_create_response()`'s own
lazy resend from ever running either. Net effect before this fix: every
60-minute cap silently wiped the model's memory of the conversation.

FIX ROUND 2 (review): the first version of this fix sent the re-seed
immediately once `_connect()` returned -- which only means the websocket
handshake finished, not that OpenAI's session is ready to accept
`conversation.item.create`. An unrecognised `error` event kills pipecat's
receive loop outright (`_receive_task_handler`'s `_handle_evt_error(evt);
return`), leaving the device deaf until the NEXT reconnect -- every hour, to
save one round trip. Fixed by deferring the actual send until
`_api_session_ready` is confirmed True (the same signal
`_create_response()` already waits on via `_run_llm_when_api_session_ready`
for the identical class of problem), via a `_handle_evt_session_updated`
override. These tests exercise BOTH cases: session already ready (send
happens immediately, unchanged observable behaviour) and session not yet
ready (send is deferred, and only happens once `_handle_evt_session_updated`
fires).

These tests drive `SafeRealtimeLLMService._reseed_context_after_reset`
directly (the surgical unit under test) and, separately, `reset_conversation`
itself with `_connect`/`_disconnect` stubbed out (so the test never touches
the network -- pipecat's own connect/disconnect machinery is not what this
fix changes) to prove the re-seed is actually wired into the reset path, not
just present as a dead method.
"""

import logging

import pytest
from pipecat.processors.aggregators.llm_context import LLMContext

from app.providers import OPENAI, ProviderOptions, build_service

RECONNECT_MESSAGES = [
    {"role": "user", "content": "what time is it in the kitchen"},
    {"role": "assistant", "content": "it is two o'clock"},
]


def _openai_options(**over):
    base = dict(
        api_key="sk-test",
        model="gpt-realtime-2",
        voice="cedar",
        instructions="You are Bjorn.",
        max_output_tokens=1024,
        speed=1.0,
        noise_reduction="",
        turn_detection_type="semantic_vad",
        vad_eagerness="medium",
        vad_threshold=0.5,
        vad_prefix_padding_ms=300,
        vad_silence_duration_ms=500,
        semantic_vad_create_response=True,
        interrupt_response=True,
        transcription_model="gpt-4o-transcribe",
        transcription_language="sv",
    )
    base.update(over)
    return ProviderOptions(**base)


class _FakeOpenAIWebSocket:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        pass


def _prepped_service(messages=None):
    """A built service with a fake websocket and a live `_context`, as if it
    had been running a real conversation right up to the reconnect. Callers
    set `_api_session_ready` themselves to pick which scenario they test."""
    service = build_service(OPENAI, _openai_options(), [])
    fake_ws = _FakeOpenAIWebSocket()
    service._websocket = fake_ws
    service._context = LLMContext(messages=list(messages if messages is not None else RECONNECT_MESSAGES))
    return service, fake_ws


def test_pending_reseed_attribute_is_declared_not_implicit():
    """FIX 4 (Minor, review): `_pending_reseed_messages` is real instance
    state a reconnect cycle sets, reads, and clears -- not incidental
    scratch space that happens to work through `getattr`'s default.
    Checking `vars(service)` (the instance's own `__dict__`), not
    `getattr(..., None)`, is what actually tells "declared and None" apart
    from "never set at all" -- `getattr` returns the same default either
    way."""
    service = build_service(OPENAI, _openai_options(), [])

    assert "_pending_reseed_messages" in vars(service), (
        "_pending_reseed_messages was never actually assigned in __init__ -- "
        "it only appears to exist via getattr's default"
    )
    assert service._pending_reseed_messages is None


@pytest.mark.asyncio
async def test_reseed_sends_immediately_when_session_already_ready():
    """If the session happens to already be confirmed ready (the ordinary,
    steady-state case once the handshake settles), the re-seed reaches the
    engine right away -- no artificial extra delay -- and nothing starts
    talking."""
    service, fake_ws = _prepped_service()
    service._api_session_ready = True

    await service._reseed_context_after_reset()

    assert len(fake_ws.sent) > 0, (
        "the live conversation never reached the reconnected session at all"
    )
    joined = " ".join(fake_ws.sent)
    assert "kitchen" in joined and "two o'clock" in joined, (
        "something was sent, but not the full conversation that was lost"
    )
    assert not any('"type": "response.create"' in payload for payload in fake_ws.sent), (
        "a response.create was sent -- the assistant would speak unprompted "
        "right after an hourly reconnect"
    )


@pytest.mark.asyncio
async def test_reseed_is_deferred_until_session_updated_confirms_ready():
    """FIX ROUND 2's core proof: right after a reconnect (`_api_session_ready`
    is False -- pipecat's own default, and what `_disconnect` always leaves
    it at), the re-seed must NOT send anything yet. It only reaches the
    engine once `_handle_evt_session_updated` -- the real, inherited method,
    driven by a dummy event since it never reads the event's fields -- fires
    and confirms the session is ready."""
    service, fake_ws = _prepped_service()
    assert service._api_session_ready is False  # sanity: real default

    await service._reseed_context_after_reset()

    assert fake_ws.sent == [], (
        "the re-seed sent immediately, before the session was confirmed "
        "ready -- exactly the race that can kill the receive loop"
    )
    assert service._pending_reseed_messages == RECONNECT_MESSAGES, (
        "the prepared re-seed was not stashed for later delivery"
    )

    await service._handle_evt_session_updated(object())

    assert service._api_session_ready is True
    assert len(fake_ws.sent) > 0, (
        "session.updated fired but the deferred re-seed was never delivered"
    )
    joined = " ".join(fake_ws.sent)
    assert "kitchen" in joined and "two o'clock" in joined
    assert service._pending_reseed_messages is None, (
        "the pending re-seed was not cleared after being delivered"
    )


@pytest.mark.asyncio
async def test_reseed_is_not_delivered_twice_on_a_second_session_updated():
    """`_handle_evt_session_updated` can fire more than once in a session's
    life (e.g. a later settings update mid-conversation) -- a second firing
    must not resend a re-seed that was already delivered."""
    service, fake_ws = _prepped_service()

    await service._reseed_context_after_reset()
    await service._handle_evt_session_updated(object())
    assert len(fake_ws.sent) == 1

    await service._handle_evt_session_updated(object())  # e.g. a later settings update

    assert len(fake_ws.sent) == 1, (
        "a second session.updated re-sent the already-delivered re-seed"
    )


@pytest.mark.asyncio
async def test_reseed_after_reset_strips_tool_plumbing():
    """Tool call ids belong to the conversation that just ended; the fresh
    session rejects a replayed id (same reasoning _strip_tool_plumbing
    exists for the client-reconnect path). A message that resolves to
    nothing once stripped (a bare tool result) must not block real
    surrounding content from still being sent."""
    service, fake_ws = _prepped_service(
        messages=[
            {"role": "user", "content": "set a timer"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_abc123", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call_abc123", "content": "done"},
            {"role": "assistant", "content": "timer set"},
        ]
    )
    service._api_session_ready = True

    await service._reseed_context_after_reset()

    joined = " ".join(fake_ws.sent)
    assert "call_abc123" not in joined, (
        "a stale tool_call_id from the pre-reset session was replayed -- "
        "OpenAI rejects the whole context with invalid_tool_call_id"
    )
    assert "timer set" in joined


@pytest.mark.asyncio
async def test_reseed_after_reset_caps_at_max_context_messages(monkeypatch):
    """FIX 2 (review): this path must be capped the same way SessionManager
    caps the client-reconnect restore, from the SAME MAX_CONTEXT_MESSAGES
    setting -- it fires every 60 minutes AND on every proactive refresh, so
    an uncapped conversation grows without bound over a long day."""
    monkeypatch.setenv("MAX_CONTEXT_MESSAGES", "2")
    long_conversation = [
        {"role": "user", "content": "message one"},
        {"role": "assistant", "content": "message two"},
        {"role": "user", "content": "message three"},
        {"role": "assistant", "content": "message four (most recent)"},
    ]
    service, fake_ws = _prepped_service(messages=long_conversation)
    service._api_session_ready = True

    await service._reseed_context_after_reset()

    joined = " ".join(fake_ws.sent)
    assert "message four" in joined and "message three" in joined, (
        "the most recent messages within the cap were dropped"
    )
    assert "message one" not in joined and "message two" not in joined, (
        "the cap (MAX_CONTEXT_MESSAGES=2) was not applied -- the full, "
        "uncapped conversation was sent"
    )


@pytest.mark.asyncio
async def test_reseed_after_reset_is_a_quiet_noop_with_no_context(caplog):
    """A service that was reset before any conversation ever happened (or
    whose context was never pre-seeded) has nothing to re-seed -- this must
    not raise, must not send anything, and (FIX ROUND 3, review) must not
    even LOG a warning: the whole method's body sits in one broad
    try/except (deliberately, per round 2's Minor fix, so a real surprise
    can't escape into `reset_conversation` and cost the router a false
    strike), which means asserting only `fake_ws.sent == []` proves nothing
    -- deleting the `if context is None: return` guard would make
    `context.get_messages()` raise `AttributeError`, get swallowed by that
    same blanket handler into a warning, and `sent` would still be `[]`.
    Asserting no warning was logged is what actually distinguishes "quiet
    no-op" from "quietly caught a bug"."""
    import logging

    service, fake_ws = _prepped_service()
    service._api_session_ready = True
    service._context = None

    with caplog.at_level(logging.WARNING, logger="app.providers.openai_realtime"):
        await service._reseed_context_after_reset()  # must not raise

    assert fake_ws.sent == []
    assert not any(r.levelno >= logging.WARNING for r in caplog.records), (
        "a genuine no-op logged a warning -- this path is catching an "
        "actual error, not doing nothing gracefully"
    )


@pytest.mark.asyncio
async def test_stale_pending_reseed_is_cleared_when_a_later_reset_has_no_context():
    """FIX 3 (Minor, review): if a LATER `_reseed_context_after_reset` call
    (e.g. a second dead-socket repair before the first reset's
    session.updated ever arrived) finds no context at all, any PREVIOUSLY
    stashed pending re-seed must be cleared too -- otherwise a later
    session.updated would still deliver messages from a reset attempt this
    later one explicitly superseded."""
    service, fake_ws = _prepped_service()
    await service._reseed_context_after_reset()  # stashes (not ready yet)
    assert service._pending_reseed_messages == RECONNECT_MESSAGES

    service._context = None  # the second reset found nothing to re-seed
    await service._reseed_context_after_reset()

    assert service._pending_reseed_messages is None, (
        "a stale pending re-seed from an earlier reset survived a later "
        "reset that found no context"
    )

    await service._handle_evt_session_updated(object())
    assert fake_ws.sent == [], "the stale re-seed was delivered anyway"


@pytest.mark.asyncio
async def test_stale_pending_reseed_is_cleared_when_a_later_reset_has_nothing_after_stripping():
    """Same guard, the other early-return: a later reset whose context
    survives `_strip_tool_plumbing`/`_cap_restored_messages` down to nothing
    (e.g. only a bare tool result) must also clear a previous stash."""
    service, fake_ws = _prepped_service()
    await service._reseed_context_after_reset()  # stashes (not ready yet)
    assert service._pending_reseed_messages == RECONNECT_MESSAGES

    # A bare tool result strips to nothing at all.
    service._context = LLMContext(
        messages=[{"role": "tool", "tool_call_id": "call_x", "content": "done"}]
    )
    await service._reseed_context_after_reset()

    assert service._pending_reseed_messages is None, (
        "a stale pending re-seed from an earlier reset survived a later "
        "reset whose context stripped down to nothing"
    )

    await service._handle_evt_session_updated(object())
    assert fake_ws.sent == [], "the stale re-seed was delivered anyway"


@pytest.mark.asyncio
async def test_reseed_failure_after_ready_is_logged_loudly(caplog):
    """FIX ROUND 2 (review): the old code only ever logged success. If the
    session claims to be ready but the send still doesn't go through (no
    websocket, or restore_context_silently declining for some other
    reason), that must be logged where someone will see it -- silently
    reopening the hourly hole a second time (this time past the ordering
    fix) is the exact invisibility this whole task exists to end."""
    service, _fake_ws = _prepped_service()
    service._api_session_ready = True
    service._websocket = None  # session claims ready, but nothing to send to

    with caplog.at_level(logging.WARNING, logger="app.providers.openai_realtime"):
        await service._reseed_context_after_reset()

    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("NOT sent" in m for m in warnings), (
        "a failed re-seed produced no log line at all -- the hourly hole "
        "reopens invisibly"
    )


@pytest.mark.asyncio
async def test_reset_conversation_actually_calls_the_reseed():
    """Wiring proof: reset_conversation() itself -- not just the helper in
    isolation -- must trigger the re-seed (deferred, since a fresh reconnect
    always starts with `_api_session_ready = False`). Stubs pipecat's own
    _connect/_disconnect (not what this fix touches, and _connect would
    otherwise dial the real OpenAI API) so this never touches the network."""
    service = build_service(OPENAI, _openai_options(), [])
    fake_ws = _FakeOpenAIWebSocket()

    async def _fake_disconnect():
        service._websocket = None

    async def _fake_connect():
        service._websocket = fake_ws

    service._connect = _fake_connect
    service._disconnect = _fake_disconnect
    # The conversation this service was tracking right up to the 60-minute
    # cap -- reset_conversation() must not lose it.
    service._context = LLMContext(messages=list(RECONNECT_MESSAGES))

    await service.reset_conversation()

    assert fake_ws.sent == [], (
        "reset_conversation() sent immediately -- before session.updated -- "
        "which is exactly the race this fix round closes"
    )
    assert service._pending_reseed_messages == RECONNECT_MESSAGES, (
        "reset_conversation() ran but never prepared a re-seed -- the "
        "re-seed helper exists but isn't actually wired in"
    )

    # The real signal a reconnected session sends once ready.
    await service._handle_evt_session_updated(object())

    assert len(fake_ws.sent) > 0, (
        "session.updated fired but the re-seed prepared by reset_conversation() "
        "was never delivered"
    )
    joined = " ".join(fake_ws.sent)
    assert "kitchen" in joined and "two o'clock" in joined
    assert not any('"type": "response.create"' in payload for payload in fake_ws.sent)
    assert service._llm_needs_conversation_setup is False
    assert service._run_llm_when_api_session_ready is False


@pytest.mark.asyncio
async def test_pending_response_at_session_updated_finds_memory_already_there():
    """FIX ROUND 3 (review): if `_run_llm_when_api_session_ready` is already
    True by the time `session.updated` arrives (something tried to create a
    response while we were still waiting -- e.g. a completed tool result),
    the OLD ordering (`await super()` first) let that `_create_response()`
    fire over a conversation `reset_conversation` had already stripped of
    `_llm_needs_conversation_setup`, landing our own re-seed's items only
    AFTER that bare response.create -- a turn answered with no memory, plus
    items arriving mid-response. The re-seed must land BEFORE any such
    response."""
    service, fake_ws = _prepped_service()
    service._run_llm_when_api_session_ready = True
    service._llm_needs_conversation_setup = False  # reset_conversation already did this

    await service._reseed_context_after_reset()
    assert fake_ws.sent == []  # stashed, not sent yet (_api_session_ready is False)

    await service._handle_evt_session_updated(object())

    assert service._api_session_ready is True
    assert service._run_llm_when_api_session_ready is False, (
        "super() never ran (or never saw the flag), so the pending response "
        "request was dropped entirely instead of being served"
    )
    reseed_index = next(
        (i for i, p in enumerate(fake_ws.sent) if '"type": "conversation.item.create"' in p),
        None,
    )
    response_index = next(
        (i for i, p in enumerate(fake_ws.sent) if '"type": "response.create"' in p),
        None,
    )
    assert reseed_index is not None, "the restored conversation was never sent at all"
    assert response_index is not None, "the pending response was never created"
    assert reseed_index < response_index, (
        "the response.create landed before (or without) the restored "
        "context -- exactly the no-memory-turn this fix closes"
    )
