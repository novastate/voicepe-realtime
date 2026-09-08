"""Telling the model who is speaking must work on both engines -- or, where
it genuinely cannot, must fail LOUDLY and SILENTLY-TO-THE-USER rather than
either vanishing quietly or making the assistant talk unprompted.

FIX ROUND 1. The first version of this test replaced the context aggregator
with a stub whose push_frame just appended to a list -- green whether the
name reached either engine or vanished into nothing. Traced directly against
the installed pipecat 0.0.97 source (not assumed): FrameProcessor.push_frame
forwards a frame to `self._next` -- it does NOT invoke the target
processor's own process_frame. Calling it on `context_aggregator.user()`
sends the frame straight past that aggregator to whatever is next in the
pipeline (the LLM service), and neither OpenAIRealtimeLLMService nor
GeminiLiveLLMService does anything with LLMMessagesUpdateFrame -- both just
forward it on unchanged. Nothing was ever sent on either websocket.

FIX ROUND 2. Round 1's fix for Gemini (`LLMMessagesAppendFrame` ->
`_create_single_response`) DID reach the model, but that method hardcodes
`turn_complete=True`, so Gemini spoke an unprompted reply every time a voice
was recognised -- worse than the silent loss this task started from. The
only other channel, `send_client_content(turn_complete=False)`, is real and
reachable (pipecat's own `_create_initial_response` uses it) but is silently
IGNORED by the model unless explicitly closed by a later
`send_client_content(turn_complete=True)` tied to the service's own
`UserStoppedSpeakingFrame` handling -- state this caller has no supported way
to hook. Verified against `GeminiLiveLLMService._handle_user_stopped_speaking`
directly (see its "without this, the model ignores the context" comment).
No channel exists that both reaches the model and doesn't force a reply, so
Gemini now gets NO model-facing injection at all -- these tests prove that
choice: Gemini sends nothing to its session and logs one clear warning
instead of a per-call "no-op"; OpenAI is provably unaffected.

These tests build REAL service instances the same way production code does
(app.providers.build_service), fake only the websocket/session (no network),
and assert on the real outbound behaviour -- what was sent, and what wasn't.
"""

import pytest

from app.providers import GEMINI, OPENAI, ProviderOptions, build_service, supports_client_events
from app.device_registry import DeviceConnection


def test_only_openai_takes_raw_client_events():
    assert supports_client_events("openai") is True
    assert supports_client_events("gemini") is False


def test_an_unknown_engine_is_refused_loudly():
    with pytest.raises(ValueError, match="unknown provider"):
        supports_client_events("claude")


def _openai_options(**over):
    base = dict(
        api_key="sk-test",
        model="gpt-realtime-2",
        voice="cedar",
        instructions="Du är Björn.",
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


def _gemini_options(**over):
    base = dict(
        api_key="AIza-test",
        model="models/gemini-3.1-flash-live-preview",
        voice="Charon",
        instructions="Du är Björn.",
        max_output_tokens=1024,
        language="sv-SE",
    )
    base.update(over)
    return ProviderOptions(**base)


class _FakeOpenAIWebSocket:
    """Stands in for the OpenAI Realtime websocket connection.

    Records what would have gone out; nothing touches the network. The real
    OpenAIRealtimeLLMService only calls `.send()` on this when `_websocket`
    is truthy (see `_ws_send` in the installed pipecat source) -- an
    unconnected service (the default; we never call `_connect()`) silently
    does nothing without this stand-in, which is exactly the trap the
    previous round's stub test fell into with the aggregator.
    """

    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)


class _FakeGeminiSession:
    """Stands in for the Gemini Live `AsyncSession`.

    Records what would have gone out; nothing touches the network. The real
    GeminiLiveLLMService only calls this when `_session` is truthy (see
    `_create_single_response` in the installed pipecat source) -- an
    unconnected service (the default; we never call `_connect()`) silently
    returns without this stand-in.
    """

    def __init__(self):
        self.calls = []

    async def send_client_content(self, **kwargs):
        self.calls.append(kwargs)


@pytest.mark.asyncio
async def test_the_speaker_name_reaches_a_real_openai_service():
    """Drives make_speaker_note's OpenAI branch against a real, unconnected
    OpenAIRealtimeLLMService (built via app.providers.build_service, exactly
    as production does) with a faked websocket. This is send_client_event --
    a direct method call, unchanged from what this code did before this task
    existed -- not a frame, so there is no queue/process_frame indirection to
    drive here; the assertion is on the real outbound payload."""
    from app.websocket_handler import make_speaker_note

    service = build_service(OPENAI, _openai_options(), [])
    fake_ws = _FakeOpenAIWebSocket()
    service._websocket = fake_ws

    connection = DeviceConnection(device_id="kitchen", websocket=object())
    connection.provider = OPENAI

    await make_speaker_note(connection, service)("male", "Henrik", 132.0)

    assert len(fake_ws.sent) == 1
    assert "Henrik" in fake_ws.sent[0]
    assert '"role": "system"' in fake_ws.sent[0]
    assert '"type": "conversation.item.create"' in fake_ws.sent[0]


@pytest.mark.asyncio
async def test_gemini_never_sends_anything_to_its_session():
    """ROUND 2's hard requirement: Gemini must not speak unprompted when a
    voice is recognised. Drives make_speaker_note's Gemini branch against a
    real, unconnected GeminiLiveLLMService with a faked session across
    several verdicts (guest, confident match, ambiguous match) and asserts
    the session's `send_client_content` -- the ONLY real send method
    reachable from this service, per the installed source -- is never called
    at all. A regression back to round 1's `_create_single_response` path
    (or any other path that reaches the session) would show up here as a
    non-empty `calls` list."""
    from app.websocket_handler import make_speaker_note

    service = build_service(GEMINI, _gemini_options(), [])
    fake_session = _FakeGeminiSession()
    service._session = fake_session

    connection = DeviceConnection(device_id="kitchen", websocket=object())
    connection.provider = GEMINI

    note = make_speaker_note(connection, service)
    await note("unknown", None, 0.0)
    await note("male", "Henrik", 132.0)
    await note("uncertain", None, 90.0)

    assert fake_session.calls == []


@pytest.mark.asyncio
async def test_gemini_logs_one_clear_warning_not_a_per_call_no_op(caplog):
    """The skip must be loud and unambiguous -- not worded like the "no-op"
    log lines that let the original bug hide for however long it shipped --
    and it must not spam once per wake for the life of a connection (the
    speaker probe fires on every wake)."""
    import logging
    from app.websocket_handler import make_speaker_note

    service = build_service(GEMINI, _gemini_options(), [])
    service._session = _FakeGeminiSession()

    connection = DeviceConnection(device_id="kitchen", websocket=object())
    connection.provider = GEMINI

    note = make_speaker_note(connection, service)
    with caplog.at_level(logging.WARNING, logger="app.websocket_handler"):
        await note("male", "Henrik", 132.0)
        await note("male", "Henrik", 132.0)
        await note("male", "Henrik", 132.0)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].message.lower()
    assert "no-op" not in message
    assert "gemini" in message


@pytest.mark.asyncio
async def test_openai_is_unaffected_by_the_gemini_change():
    """ROUND 2 touched only the `else` (non-OpenAI) branch of note() --
    confirm the OpenAI path still reaches a real, unconnected
    OpenAIRealtimeLLMService exactly as before, via the same
    send_client_event call, with no dependency introduced on anything Gemini-
    specific (this test never constructs a Gemini service at all)."""
    from app.websocket_handler import make_speaker_note

    service = build_service(OPENAI, _openai_options(), [])
    fake_ws = _FakeOpenAIWebSocket()
    service._websocket = fake_ws

    connection = DeviceConnection(device_id="kitchen", websocket=object())
    connection.provider = OPENAI

    note = make_speaker_note(connection, service)
    await note("male", "Henrik", 132.0)
    await note("male", "Henrik", 132.0)

    # Unlike Gemini, OpenAI sends on every call -- no once-per-connection
    # suppression, because there is nothing to warn about.
    assert len(fake_ws.sent) == 2
    assert all("Henrik" in payload for payload in fake_ws.sent)


@pytest.mark.asyncio
async def test_a_guest_voice_still_gets_a_context_note_on_openai():
    """The old callback never guarded on `name` being truthy -- the "this
    voice matches nobody enrolled" case fires with name=None, and the model
    still needs to be told to stay neutral (no names, no sir/ma'am). A naive
    `if not name: return` guard would silently drop this branch exactly the
    way the OpenAI-only event silently dropped everything on Gemini."""
    from app.websocket_handler import make_speaker_note

    service = build_service(OPENAI, _openai_options(), [])
    fake_ws = _FakeOpenAIWebSocket()
    service._websocket = fake_ws

    connection = DeviceConnection(device_id="kitchen", websocket=object())
    connection.provider = OPENAI

    await make_speaker_note(connection, service)("unknown", None, 0.0)

    assert len(fake_ws.sent) == 1
    assert "guest" in fake_ws.sent[0].lower()


@pytest.mark.asyncio
async def test_an_ambiguous_match_still_gets_a_context_note_on_openai():
    """The "not confidently matched" fallback also fires with name=None and
    must still reach the model, naming the household candidates from the
    probe -- another name=None case an early `if not name: return` would
    have silently eaten."""
    from types import SimpleNamespace
    from app.websocket_handler import make_speaker_note

    service = build_service(OPENAI, _openai_options(), [])
    fake_ws = _FakeOpenAIWebSocket()
    service._websocket = fake_ws

    connection = DeviceConnection(device_id="kitchen", websocket=object())
    connection.provider = OPENAI

    probe = SimpleNamespace(male_name="Henrik", female_name="Anna")
    await make_speaker_note(connection, service, probe)("uncertain", None, 90.0)

    assert len(fake_ws.sent) == 1
    assert "Henrik" in fake_ws.sent[0] and "Anna" in fake_ws.sent[0]


@pytest.mark.asyncio
async def test_a_failed_send_is_logged_not_raised():
    """The outer try/except in note() must keep swallowing delivery
    failures (carried over from the original callback) -- a dead connection
    must not blow up the wake-handling path that fires this callback.

    Raising from `send_client_event` itself (rather than from the fake
    websocket's `.send()`) is deliberate: `_ws_send` in the installed
    pipecat source already has its own internal try/except around the
    socket write, so an exception raised there never reaches our code at
    all -- that would make this test pass whether or not note()'s own
    try/except exists. Patching `send_client_event` directly exercises the
    try/except this code actually owns."""
    from app.websocket_handler import make_speaker_note

    service = build_service(OPENAI, _openai_options(), [])

    async def _boom(_event):
        raise ConnectionError("socket is gone")

    service.send_client_event = _boom

    connection = DeviceConnection(device_id="kitchen", websocket=object())
    connection.provider = OPENAI

    # Must not raise.
    await make_speaker_note(connection, service)("male", "Henrik", 132.0)
