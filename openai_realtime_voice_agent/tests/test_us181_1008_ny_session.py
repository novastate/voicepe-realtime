"""US-181: a resume rejected after connect must not kill the reader."""
from unittest.mock import AsyncMock

import pytest
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from app.providers import build_service
from test_gemini_provider import OPENAI_SHAPE, _options


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["connect", "receive"])
async def test_expired_resume_starts_fresh_and_next_phrase_can_be_sent(monkeypatch, path):
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    service._session_resumption_handle = "expired"
    service._session = None
    service._connection_task = object()
    async def refused(self, config):
        raise RuntimeError("1008 None. BidiGenerateContent session not found or expired")
    monkeypatch.setattr(GeminiLiveLLMService, "_connection_task_handler", refused)
    sessions = []
    async def connect(handle):
        sessions.append(handle)
        service._session = AsyncMock()
    service._connect = connect
    if path == "connect":
        await service._connection_task_handler(None)
    else:
        assert await service._handle_connection_error(
            RuntimeError("1008 None. BidiGenerateContent session not found or expired")) is False
    assert sessions == [None]
    assert service._session_resumption_handle is None
    await service.mata_text("Hej igen")
    service._session.send_realtime_input.assert_awaited_once_with(text="Hej igen")


@pytest.mark.asyncio
async def test_expired_error_without_resume_is_not_retried_forever(monkeypatch):
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    async def refused(self, config):
        raise RuntimeError("1008 None. BidiGenerateContent session not found or expired")
    monkeypatch.setattr(GeminiLiveLLMService, "_connection_task_handler", refused)
    with pytest.raises(RuntimeError):
        await service._connection_task_handler(None)
