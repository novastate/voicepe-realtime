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

These tests drive `SafeRealtimeLLMService._reseed_context_after_reset`
directly (the surgical unit under test) and, separately, `reset_conversation`
itself with `_connect`/`_disconnect` stubbed out (so the test never touches
the network -- pipecat's own connect/disconnect machinery is not what this
fix changes) to prove the re-seed is actually wired into the reset path, not
just present as a dead method.
"""

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


@pytest.mark.asyncio
async def test_reseed_after_reset_reaches_the_engine_without_starting_a_turn():
    """The core proof: after a reset, the SAME service's own live context
    reaches the reconnected session again, and nothing starts talking."""
    service = build_service(OPENAI, _openai_options(), [])
    fake_ws = _FakeOpenAIWebSocket()
    service._websocket = fake_ws
    # Simulates an ongoing conversation pipecat has been tracking on this
    # SAME service instance for the last 59 minutes, right up to the cap.
    service._context = LLMContext(messages=list(RECONNECT_MESSAGES))

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
async def test_reseed_after_reset_strips_tool_plumbing():
    """Tool call ids belong to the conversation that just ended; the fresh
    session rejects a replayed id (same reasoning _strip_tool_plumbing
    exists for the client-reconnect path). A message that resolves to
    nothing once stripped (a bare tool result) must not block real
    surrounding content from still being sent."""
    service = build_service(OPENAI, _openai_options(), [])
    fake_ws = _FakeOpenAIWebSocket()
    service._websocket = fake_ws
    service._context = LLMContext(
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

    await service._reseed_context_after_reset()

    joined = " ".join(fake_ws.sent)
    assert "call_abc123" not in joined, (
        "a stale tool_call_id from the pre-reset session was replayed -- "
        "OpenAI rejects the whole context with invalid_tool_call_id"
    )
    assert "timer set" in joined


@pytest.mark.asyncio
async def test_reseed_after_reset_is_a_quiet_noop_with_no_context():
    """A service that was reset before any conversation ever happened (or
    whose context was never pre-seeded) has nothing to re-seed -- this must
    not raise and must not send anything."""
    service = build_service(OPENAI, _openai_options(), [])
    fake_ws = _FakeOpenAIWebSocket()
    service._websocket = fake_ws
    service._context = None

    await service._reseed_context_after_reset()  # must not raise

    assert fake_ws.sent == []


@pytest.mark.asyncio
async def test_reset_conversation_actually_calls_the_reseed():
    """Wiring proof: reset_conversation() itself -- not just the helper in
    isolation -- must trigger the re-seed. Stubs pipecat's own _connect/
    _disconnect (not what this fix touches, and _connect would otherwise
    dial the real OpenAI API) so this never touches the network."""
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

    assert len(fake_ws.sent) > 0, (
        "reset_conversation() ran but never re-seeded the reconnected "
        "session -- the re-seed helper exists but isn't actually wired in"
    )
    joined = " ".join(fake_ws.sent)
    assert "kitchen" in joined and "two o'clock" in joined
    assert not any('"type": "response.create"' in payload for payload in fake_ws.sent)
    assert service._llm_needs_conversation_setup is False
    assert service._run_llm_when_api_session_ready is False
