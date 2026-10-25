"""US-189: journal each turn through pipecat's real dispatcher."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
import logging

import pytest
from google.genai import types
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

from app.providers import build_service
from test_gemini_provider import OPENAI_SHAPE, _options


def message(**content):
    return types.LiveServerMessage(server_content=types.LiveServerContent(**content),
                                   usage_metadata=types.UsageMetadata())


@pytest.fixture
def service(caplog):
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    service.push_frame = AsyncMock()
    caplog.set_level(logging.INFO, logger="app.providers.gemini_live")
    return service


async def receive_messages(service, monkeypatch, messages):
    """Use the actual if/elif dispatch with a finite offline socket."""
    async def receive():
        for msg in messages:
            yield msg
        service._disconnecting = True
        raise RuntimeError("end of test socket")
    session = SimpleNamespace(receive=receive)
    async def ready(self, session):
        self._session = session
    monkeypatch.setattr(GeminiLiveLLMService, "_handle_session_ready", ready)
    class Connection:
        async def __aenter__(self):
            return session
        async def __aexit__(self, *args):
            pass
    service._client = SimpleNamespace(aio=SimpleNamespace(live=SimpleNamespace(
        connect=lambda **kwargs: Connection())))
    await service._connection_task_handler(None)


@pytest.mark.asyncio
async def test_ac1_one_answer_including_tool_followup(service, monkeypatch, caplog):
    monkeypatch.setattr(GeminiLiveLLMService, "_handle_msg_tool_call", AsyncMock())
    monkeypatch.setattr(GeminiLiveLLMService, "_handle_msg_model_turn", AsyncMock())
    await receive_messages(service, monkeypatch, [
        message(model_turn=types.Content(parts=[types.Part(inline_data=types.Blob(
            data=b"\x00\x00", mime_type="audio/pcm;rate=24000"))]),
                output_transcription=types.Transcription(text="Jag kollar. ")),
        types.LiveServerMessage(tool_call=types.LiveServerToolCall(function_calls=[
            types.FunctionCall(id="lamp", name="list_timers", args={})])),
        message(turn_complete=True),  # empty tool boundary must not flush the preamble
        message(output_transcription=types.Transcription(text="Lampan är tänd."), turn_complete=True),
        message(turn_complete=True),
    ])
    assert [r.message for r in caplog.records if "💬 Björn sa:" in r.message] == [
        "💬 Björn sa: Jag kollar. Lampan är tänd."]
    assert service._svar_text == ""


@pytest.mark.asyncio
async def test_ac2_heard_once_before_answer_next_turn_clean(service, monkeypatch, caplog):
    await receive_messages(service, monkeypatch, [
        message(input_transcription=types.Transcription(text="Är lampan ")),
        message(input_transcription=types.Transcription(text="tänd?"),
                output_transcription=types.Transcription(text="Ja.")),
        message(turn_complete=True),
        message(input_transcription=types.Transcription(text="Tack.")),
        message(output_transcription=types.Transcription(text="Varsågod.")),
        message(turn_complete=True),
    ])
    assert [r.message for r in caplog.records if "Hörde" in r.message or "Björn sa:" in r.message] == [
        "🎤 Hörde: Är lampan tänd?", "💬 Björn sa: Ja.",
        "🎤 Hörde: Tack.", "💬 Björn sa: Varsågod."]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["1011 Internal error", "1008 None. BidiGenerateContent session not found or expired"])
async def test_ac3_close_flushes_interrupted_text_and_clears_buffers(service, monkeypatch, caplog, error):
    monkeypatch.setattr(GeminiLiveLLMService, "_handle_connection_error", AsyncMock(return_value=True))
    service._fresh_session = AsyncMock()
    service._session_resumption_handle = "expired"
    await service._handle_msg_input_transcription(message(input_transcription=types.Transcription(text="Hur är läget?")))
    await service._handle_msg_output_transcription(message(output_transcription=types.Transcription(text="Jo tack")))
    service._answer_after_tool = True
    await service._handle_connection_error(RuntimeError(error))
    lines = [r.message for r in caplog.records if "avbruten" in r.message]
    assert len(lines) == 2
    assert "Hörde" in lines[0] and "Hur är läget?" in lines[0]
    assert "Björn sa" in lines[1] and "Jo tack" in lines[1]
    assert service._tur_text == service._svar_text == service._transkr_start == ""
    assert not service._transkr_kollad and not service._answer_after_tool
    caplog.clear()
    await service._handle_connection_error(RuntimeError(error))
    assert not [r for r in caplog.records if "avbruten" in r.message]
    await service._handle_msg_input_transcription(message(input_transcription=types.Transcription(text="Hej igen")))
    await service._handle_msg_output_transcription(message(output_transcription=types.Transcription(text="Hej!")))
    await service._handle_msg_turn_complete(message(turn_complete=True))
    assert [r.message for r in caplog.records if "Hörde:" in r.message or "Björn sa:" in r.message] == [
        "🎤 Hörde: Hej igen", "💬 Björn sa: Hej!"]


@pytest.mark.asyncio
async def test_late_preamble_while_tool_unanswered_does_not_flush_early(service, monkeypatch, caplog):
    """US-189 F2: a preamble transcript arriving after the tool_call keeps the empty boundary swallowed."""
    monkeypatch.setattr(GeminiLiveLLMService, "_handle_msg_tool_call", AsyncMock())
    monkeypatch.setattr(GeminiLiveLLMService, "_handle_msg_model_turn", AsyncMock())

    async def tool_call_then_wait(self, message):
        self._answer_after_tool = True
        self._unanswered_tools = {"t1"}
    monkeypatch.setattr(type(service), "_handle_msg_tool_call", tool_call_then_wait, raising=True)
    await receive_messages(service, monkeypatch, [
        types.LiveServerMessage(tool_call=types.LiveServerToolCall(function_calls=[
            types.FunctionCall(id="t1", name="list_timers", args={})])),
        message(output_transcription=types.Transcription(text="Jag kollar. ")),  # late preamble
        message(turn_complete=True),  # empty tool boundary
    ])
    assert not [r for r in caplog.records if "💬 Björn sa:" in r.message]
    service._unanswered_tools.discard("t1")
    await service._handle_msg_output_transcription(message(output_transcription=types.Transcription(text="Tänd.")))
    await service._handle_msg_turn_complete(message(turn_complete=True))
    assert [r.message for r in caplog.records if "💬 Björn sa:" in r.message] == ["💬 Björn sa: Jag kollar. Tänd."]
