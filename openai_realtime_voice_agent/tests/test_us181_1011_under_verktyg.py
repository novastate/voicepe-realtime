"""US-181: an unanswered call is not an idle house."""
import time
from unittest.mock import AsyncMock

import pytest
from pipecat.services.google.gemini_live.llm import MAX_CONSECUTIVE_FAILURES
from pipecat.services.llm_service import LLMService, FunctionCallFromLLM

from app.providers import build_service
from test_gemini_provider import OPENAI_SHAPE, _options


@pytest.mark.asyncio
async def test_unanswered_call_preserves_failures_until_existing_fatal_threshold(monkeypatch):
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    monkeypatch.setattr(LLMService, "run_function_calls", AsyncMock())
    await service.run_function_calls([FunctionCallFromLLM(
        context=None, tool_call_id="pending", function_name="l", arguments={})])
    service.push_error = AsyncMock()
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
    monkeypatch.setattr(GeminiLiveLLMService, "_handle_session_ready", AsyncMock())
    for n in range(MAX_CONSECUTIVE_FAILURES):
        await service._handle_session_ready(AsyncMock())
        service._connection_start_time = time.time() - 94
        retry = await service._handle_connection_error(RuntimeError("1011 Internal error"))
        assert service._consecutive_failures == n + 1
    assert retry is False
    service.push_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_answered_tool_and_fresh_session_release_pending_state(monkeypatch):
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    monkeypatch.setattr(LLMService, "run_function_calls", AsyncMock())
    call = FunctionCallFromLLM(context=None, tool_call_id="a", function_name="styr", arguments={})
    await service.run_function_calls([call])
    service._session = AsyncMock()
    await service._tool_result("a", "styr", {"ok": True})
    assert not service._unanswered_tools
    await service.run_function_calls([call])
    monkeypatch.setattr(GeminiLiveLLMService, "_handle_session_ready", AsyncMock())
    monkeypatch.setattr(GeminiLiveLLMService, "_connect", AsyncMock())
    service._session = None
    await service._connect(None)
    assert not service._unanswered_tools


@pytest.mark.asyncio
async def test_failed_tool_response_stays_unanswered(monkeypatch):
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    monkeypatch.setattr(LLMService, "run_function_calls", AsyncMock())
    await service.run_function_calls([FunctionCallFromLLM(
        context=None, tool_call_id="failed", function_name="styr", arguments={})])
    service._session = AsyncMock()
    service._session.send_tool_response.side_effect = RuntimeError("socket closed")
    service._handle_send_error = AsyncMock()
    await service._tool_result("failed", "styr", {"ok": True})
    assert service._unanswered_tools == {"failed"}


@pytest.mark.asyncio
async def test_cancelled_tool_does_not_cause_false_failover_after_reconnect(monkeypatch):
    """B:s fynd: a tool that never answers must not stop the failure counter resetting on later idle closes."""
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    monkeypatch.setattr(LLMService, "run_function_calls", AsyncMock())
    await service.run_function_calls([FunctionCallFromLLM(
        context=None, tool_call_id="cancelled", function_name="styr", arguments={})])
    service.push_error = AsyncMock()
    monkeypatch.setattr(GeminiLiveLLMService, "_connect", AsyncMock())
    monkeypatch.setattr(GeminiLiveLLMService, "_handle_session_ready", AsyncMock())
    service._connection_start_time = time.time() - 94
    await service._handle_connection_error(RuntimeError("1011 Internal error"))  # counted: a tool was owed
    await service._connect("handle")  # the reconnect (with a resumption handle) clears what the old socket owed
    assert not service._unanswered_tools
    for _ in range(MAX_CONSECUTIVE_FAILURES + 2):  # ordinary idle closes afterwards are forgiven, never fatal
        service._connection_start_time = time.time() - 94
        await service._handle_connection_error(RuntimeError("1011 Internal error"))
        assert service._consecutive_failures <= 1
    service.push_error.assert_not_awaited()
