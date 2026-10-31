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


@pytest.mark.asyncio
@pytest.mark.parametrize("activity_open", [True, False])
async def test_expired_resume_mid_turn_tells_him_the_turn_was_lost(activity_open):
    """Codex 2026-10-31: the expired-resume shortcut returned before the turn-lost notice."""
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    service._session_resumption_handle = "expired"
    service._activity_open = activity_open
    lost = []
    async def on_lost():
        lost.append(1)
    service.on_turn_lost = on_lost
    async def connect(handle):
        service._session = AsyncMock()
    service._connect = connect
    assert await service._handle_connection_error(
        RuntimeError("1008 None. BidiGenerateContent session not found or expired")) is False
    import asyncio
    await asyncio.sleep(0)
    assert lost == ([1] if activity_open else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("age,expect", [(0.1, 1), (10_000.0, 0)])
async def test_expired_resume_awaited_reply_window(age, expect):
    import asyncio, time
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    service._session_resumption_handle = "expired"
    service._activity_open = False
    service._reply_awaited_at = time.monotonic() - age
    lost = []
    async def on_lost():
        lost.append(1)
    service.on_turn_lost = on_lost
    async def connect(handle):
        service._session = AsyncMock()
    service._connect = connect
    await service._handle_connection_error(RuntimeError("1008 None. BidiGenerateContent session not found or expired"))
    await asyncio.sleep(0)
    assert len(lost) == expect and service._reply_awaited_at is None
