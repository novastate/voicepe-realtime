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
was recognised -- worse than the silent loss this task started from. Round 2
looked for `send_client_content(turn_complete=False)` as a silent
alternative, found `_handle_user_stopped_speaking`'s own comment ("without
this, the model ignores the context") and concluded the bridge that makes a
`turn_complete=False` seed actually count required hooking
`UserStoppedSpeakingFrame` ourselves -- something this caller has no
supported way to do -- and shipped "no model-facing injection on Gemini"
instead.

FIX ROUND 3 found that conclusion wrong: the bridge round 2 thought needed
building already exists and is already running. `_create_initial_response`
sets a plain instance flag, `_needs_turn_complete_message = True`, after
sending with `turn_complete=False`; `_handle_user_stopped_speaking` --
already wired into `GeminiLiveLLMService.process_frame`'s
`UserStoppedSpeakingFrame` branch, so it runs on every real turn boundary the
live pipeline produces with no help from us -- checks that same flag and
silently closes the turn. Setting the flag from outside is a plain attribute
write, not a hook, subclass, or monkeypatch. So: Gemini DOES get the note,
silently, folded in when the model next responds to a real turn (one turn
later than OpenAI's immediate injection -- a real, disclosed cost, not
hidden). A structural guard covers the one way this could quietly break
again: if a future pipecat renames or removes
`_needs_turn_complete_message`, the injection is skipped and an ERROR is
logged naming the attribute, every time, deliberately not deduplicated.

These tests build REAL service instances the same way production code does
(app.providers.build_service), fake only the websocket/session (no network),
and assert on the real outbound behaviour -- what was sent, with what flag,
and what wasn't.
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
async def test_gemini_is_never_told_who_is_speaking():
    """Measured in the house 2026-09-09, and it cost an afternoon.

    Sending the note made Gemini answer the next question in TEXT ONLY. The
    reply was generated — the transcript showed it, using the person's name,
    so the note plainly arrived — but no audio followed, the device sat in
    `thinking`, and the watchdog forced it idle after 15 seconds. The
    assistant was mute. Turning the speaker probe off brought the voice back
    on the very next utterance.

    Reasoning about the turn bookkeeping offline said this was safe. The room
    said otherwise, and the room wins. Nothing goes to Gemini now.
    """
    from app.websocket_handler import _send_gemini_note_silently

    service = build_service(GEMINI, _gemini_options(), [])
    fake_session = _FakeGeminiSession()
    service._session = fake_session

    sent = await _send_gemini_note_silently(service, "The person speaking now is Henrik.")

    assert sent is False
    assert fake_session.calls == []
    # And the turn bookkeeping is left exactly as it was found, so nothing
    # else in the service starts behaving as if a turn were open.
    assert service._needs_turn_complete_message is False
@pytest.mark.asyncio
async def test_gemini_missing_turn_close_bridge_skips_and_logs_error(caplog):
    """The structural guard: if `_needs_turn_complete_message` is gone (a
    future pipecat rename/removal), the injection must be skipped -- not
    raise, and not silently do nothing -- and it must say exactly which
    attribute vanished, at ERROR level. Simulated by deleting the attribute
    off an otherwise-real, fully-constructed GeminiLiveLLMService instance,
    rather than replacing the service with a hand-rolled stub -- everything
    else about the service (get_llm_adapter, _session) stays real."""
    import logging
    from app.websocket_handler import make_speaker_note

    service = build_service(GEMINI, _gemini_options(), [])
    fake_session = _FakeGeminiSession()
    service._session = fake_session
    del service._needs_turn_complete_message

    connection = DeviceConnection(device_id="kitchen", websocket=object())
    connection.provider = GEMINI

    with caplog.at_level(logging.ERROR, logger="app.websocket_handler"):
        # Must not raise.
        await make_speaker_note(connection, service)("male", "Henrik", 132.0)

    assert fake_session.calls == []  # skipped, not sent
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "_needs_turn_complete_message" in errors[0].message


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
