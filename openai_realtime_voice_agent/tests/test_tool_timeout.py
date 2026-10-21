"""A tool that never answers must not keep the turn alive forever."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from app.phase_emitter import TurnLiveness
from app.providers import XAI, ProviderOptions, build_service


def _service():
    service = build_service(
        XAI,
        ProviderOptions(
            api_key="k",
            model="m",
            voice="rex",
            instructions="x",
            turn_detection_type="server_vad",
        ),
        [],
    )
    service.turn_liveness = TurnLiveness()
    return service


async def _call(service, name):
    results = []

    async def result_callback(result, *args, **kwargs):
        results.append(result)

    await service._functions[name].handler(
        SimpleNamespace(arguments={}, result_callback=result_callback, function_name=name)
    )
    return results


@pytest.mark.asyncio
async def test_hanging_tool_times_out_and_releases_liveness(monkeypatch):
    monkeypatch.setenv("TOOL_TIMEOUT_SECONDS", "0.2")
    service = _service()

    async def hanging_handler(params):
        await asyncio.sleep(5)

    service.register_function("hanging_tool", hanging_handler)
    started = time.monotonic()

    results = await asyncio.wait_for(_call(service, "hanging_tool"), timeout=1.0)

    assert time.monotonic() - started < 1.0
    assert results == [{
        "error": (
            "Verktyget svarade inte inom 0.2 sekunder. Säg att det inte gick att "
            "bekräfta att det blev gjort, och påstå aldrig att det är klart."
        )
    }]
    assert service.turn_liveness.in_flight == 0


@pytest.mark.asyncio
async def test_fast_tool_still_returns_its_result(monkeypatch):
    monkeypatch.setenv("TOOL_TIMEOUT_SECONDS", "0.2")
    service = _service()

    async def fast_handler(params):
        await params.result_callback({"result": "klart"})

    service.register_function("fast_tool", fast_handler)

    assert await _call(service, "fast_tool") == [{"result": "klart"}]
    assert service.turn_liveness.in_flight == 0
