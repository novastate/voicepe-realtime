"""Telling the model who is speaking must work on both engines.

The house has a name configured for one voice. Before this, the name reached
the model through an OpenAI-only client event, which the Gemini service does
not have -- the call fell into its except branch and the model simply never
learned who was talking. Silently: the log line said "no-op"."""

import pytest

from app.providers import supports_client_events


def test_only_openai_takes_raw_client_events():
    assert supports_client_events("openai") is True
    assert supports_client_events("gemini") is False


def test_an_unknown_engine_is_refused_loudly():
    with pytest.raises(ValueError, match="unknown provider"):
        supports_client_events("claude")


@pytest.mark.asyncio
async def test_the_speaker_name_reaches_the_model_on_both_engines():
    """The injected note must be pushed as a pipecat frame, which both
    services understand, not as an engine-specific event."""
    from app.websocket_handler import make_speaker_note

    pushed = []

    class FakeAggregator:
        async def push_frame(self, frame):
            pushed.append(frame)

    await make_speaker_note(FakeAggregator())("male", "Henrik", 132.0)
    assert len(pushed) == 1
    text = str(pushed[0])
    assert "Henrik" in text


@pytest.mark.asyncio
async def test_a_guest_voice_still_gets_a_context_note():
    """The old callback never guarded on `name` being truthy -- the "this
    voice matches nobody enrolled" case fires with name=None, and the model
    still needs to be told to stay neutral (no names, no sir/ma'am). A naive
    `if not name: return` guard (as a simpler-looking rewrite might add)
    would silently drop this branch exactly the way the OpenAI-only event
    silently dropped everything on Gemini."""
    from app.websocket_handler import make_speaker_note

    pushed = []

    class FakeAggregator:
        async def push_frame(self, frame):
            pushed.append(frame)

    await make_speaker_note(FakeAggregator())("unknown", None, 0.0)
    assert len(pushed) == 1
    assert "guest" in str(pushed[0]).lower()


@pytest.mark.asyncio
async def test_an_ambiguous_match_still_gets_a_context_note():
    """The "not confidently matched" fallback also fires with name=None and
    must still reach the model, naming the household candidates from the
    probe -- another name=None case an early `if not name: return` would
    have silently eaten."""
    from types import SimpleNamespace
    from app.websocket_handler import make_speaker_note

    pushed = []

    class FakeAggregator:
        async def push_frame(self, frame):
            pushed.append(frame)

    probe = SimpleNamespace(male_name="Henrik", female_name="Anna")
    await make_speaker_note(FakeAggregator(), probe)("uncertain", None, 90.0)
    assert len(pushed) == 1
    text = str(pushed[0])
    assert "Henrik" in text and "Anna" in text
