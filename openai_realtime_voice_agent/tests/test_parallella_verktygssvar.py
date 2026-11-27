"""Rosten svarade två gånger på två samtidiga verktygsanrop (kontoret 2026-11-27 15:20):
Gemini svarar en gång per send_tool_response. Parallella svar från samma toolCall ska skickas i ETT."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService

from app.providers import build_service
from app.providers import gemini_live
from app.providers.tool_registration import _is_error
from test_gemini_provider import OPENAI_SHAPE, _options


def _anrop(*ids):
    return [FunctionCallFromLLM(context=None, tool_call_id=i, function_name="verktyg", arguments={}) for i in ids]


async def _tjanst(monkeypatch, *ids):
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    monkeypatch.setattr(LLMService, "run_function_calls", AsyncMock())
    await service.run_function_calls(_anrop(*ids))
    service._session = AsyncMock()
    return service


def _skickade(service):
    return [c.kwargs["function_responses"] for c in service._session.send_tool_response.await_args_list]


def _ids(skick):
    return [skick.id] if not isinstance(skick, list) else [r.id for r in skick]


@pytest.mark.asyncio
async def test_two_parallel_calls_give_one_send_with_two_responses(monkeypatch):
    service = await _tjanst(monkeypatch, "a", "b")
    await service._tool_result("a", "lage_drift", {"result": "ok"})
    assert service._session.send_tool_response.await_count == 0  # waits for b
    await service._tool_result("b", "hus_oversikt", {"result": "ok"})
    skick = _skickade(service)
    assert len(skick) == 1
    assert [r.id for r in skick[0]] == ["a", "b"]
    assert [r.name for r in skick[0]] == ["lage_drift", "hus_oversikt"]
    assert not service._unanswered_tools


@pytest.mark.asyncio
async def test_a_lone_call_is_sent_unchanged(monkeypatch):
    service = await _tjanst(monkeypatch, "a")
    await service._tool_result("a", "styr", {"ok": True})
    skick = _skickade(service)
    assert len(skick) == 1
    assert not isinstance(skick[0], list)  # as before: one FunctionResponse, not a list
    assert skick[0].id == "a" and skick[0].response == {"ok": True}
    assert not service._unanswered_tools


@pytest.mark.asyncio
async def test_a_slow_tool_does_not_block_forever(monkeypatch):
    monkeypatch.setattr(gemini_live, "PARALLELL_TAK_S", 0.05)
    service = await _tjanst(monkeypatch, "a", "b")
    await service._tool_result("a", "snabb", {"ok": 1})
    await asyncio.sleep(0.2)
    skick = _skickade(service)
    assert len(skick) == 1 and _ids(skick[0]) == ["a"]  # the cap sent what existed
    assert service._unanswered_tools == {"b"}
    await service._tool_result("b", "langsam", {"ok": 2})  # the rest goes as a new answer
    skick = _skickade(service)
    assert len(skick) == 2 and _ids(skick[1]) == ["b"]
    assert not service._unanswered_tools


@pytest.mark.asyncio
async def test_no_cap_send_after_everything_was_sent(monkeypatch):
    monkeypatch.setattr(gemini_live, "PARALLELL_TAK_S", 0.05)
    service = await _tjanst(monkeypatch, "a", "b")
    await service._tool_result("a", "x", {})
    await service._tool_result("b", "y", {})
    await asyncio.sleep(0.2)
    assert service._session.send_tool_response.await_count == 1


@pytest.mark.asyncio
async def test_failed_send_keeps_all_calls_unanswered(monkeypatch):
    service = await _tjanst(monkeypatch, "a", "b")
    service._session.send_tool_response.side_effect = RuntimeError("socket closed")
    service._handle_send_error = AsyncMock()
    await service._tool_result("a", "x", {})
    await service._tool_result("b", "y", {})
    assert service._unanswered_tools == {"a", "b"}


@pytest.mark.asyncio
async def test_new_connection_drops_held_responses(monkeypatch):
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
    monkeypatch.setattr(GeminiLiveLLMService, "_connect", AsyncMock())
    service = await _tjanst(monkeypatch, "a", "b")
    await service._tool_result("a", "x", {})
    await service._connect(None)
    await service._tool_result("b", "y", {})  # the old call is not ours anymore: sent alone, not held
    assert service._session.send_tool_response.await_count == 1


def test_fel_text_counts_as_error():
    assert _is_error("Fel [Comms, okant_verktyg]: Det verktyget finns inte för dig.")
    assert _is_error({"result": "Fel [Comms, okant_verktyg]: nej"})
    assert _is_error({"error": "x"})
    assert not _is_error("Klart: spelar P3 i Kontor.")
    assert not _is_error({"result": "Allt är Fel [ i mitten"})
    assert not _is_error({"result": "Klart"})
