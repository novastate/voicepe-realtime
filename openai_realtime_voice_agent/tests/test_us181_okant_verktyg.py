"""US-181: pipecat's wildcard dispatch returns an error for unknown names."""
from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import FunctionCallResultFrame
from pipecat.services.llm_service import FunctionCallFromLLM
from app.providers import build_service
from test_gemini_provider import OPENAI_SHAPE, _options


@pytest.mark.asyncio
async def test_unknown_name_gets_result_through_real_pipecat_dispatch(caplog):
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    frames = []
    async def broadcast(kind, **kw):
        frames.append((kind, kw))
    service.broadcast_frame = broadcast
    service._call_event_handler = AsyncMock()
    # Run the real pipecat runner inline, without a pipeline task manager.
    async def parallel(items):
        for item in items:
            await service._run_function_call(item)
    service._run_parallel_function_calls = parallel
    await service.run_function_calls([FunctionCallFromLLM(
        context=None, tool_call_id="unknown-1", function_name="l", arguments={})])
    results = [kw for kind, kw in frames if kind is FunctionCallResultFrame]
    assert len(results) == 1
    assert results[0]["function_name"] == "l"
    assert results[0]["tool_call_id"] == "unknown-1"
    assert "okänt verktyg" in results[0]["result"]["error"]
    assert "l" in caplog.text and "okänt verktyg" in caplog.text
