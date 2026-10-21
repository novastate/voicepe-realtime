"""A real barge-in cancels tools, while Gemini's emulated start does not."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pipecat.frames.frames import UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frame_processor import FrameProcessor

from app.phase_emitter import PhaseEmitter, TurnLiveness
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


def _params(name, arguments, results):
    props = []

    async def result_callback(result, *args, **kwargs):
        results.append(result)
        props.append(kwargs.get("properties"))

    return SimpleNamespace(
        arguments=arguments,
        result_callback=result_callback,
        function_name=name,
        props=props,
    )


async def _interrupt(liveness, *, emulated):
    async def send_phase(_value):
        return None

    emitter = PhaseEmitter(send_phase, idle_debounce_s=0, liveness=liveness)
    emitter.push_frame = AsyncMock()
    with patch.object(FrameProcessor, "process_frame", new=AsyncMock()):
        await emitter.process_frame(
            UserStartedSpeakingFrame(emulated=emulated), FrameDirection.DOWNSTREAM
        )
    await emitter.close()


async def _assert_cancelled(task):
    done, _ = await asyncio.wait({task}, timeout=0.2)
    if task not in done:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        pytest.fail("the running tool was not cancelled by the real interruption")
    await task  # a real barge-in is swallowed: the call is closed, nothing propagates


@pytest.mark.asyncio
async def test_real_interruption_cancels_running_tool_without_result():
    service = _service()
    started = asyncio.Event()
    results = []

    async def sleeping_tool(params):
        started.set()
        await asyncio.sleep(10)
        await params.result_callback({"result": "sent"})

    service.register_function("HassTurnOn", sleeping_tool)
    params = _params("HassTurnOn", {"area": "kontoret", "domain": ["light"]}, results)
    task = asyncio.create_task(service._functions["HassTurnOn"].handler(params))
    await asyncio.wait_for(started.wait(), timeout=1.0)

    await _interrupt(service.turn_liveness, emulated=False)

    await _assert_cancelled(task)
    assert service.turn_liveness.in_flight == 0
    # pipecat keeps a call IN_PROGRESS until it gets a result: exactly one, and it must not make the model talk
    assert results == [{"result": "Avbröts av användaren. Det är inte bekräftat om det hann utföras."}]
    assert params.props[0].run_llm is False


@pytest.mark.asyncio
async def test_emulated_interruption_does_not_cancel_running_tool():
    service = _service()
    started = asyncio.Event()
    finish = asyncio.Event()
    results = []

    async def sleeping_tool(params):
        started.set()
        await finish.wait()
        await params.result_callback({"result": "klart"})

    service.register_function("HassTurnOn", sleeping_tool)
    task = asyncio.create_task(
        service._functions["HassTurnOn"].handler(
            _params("HassTurnOn", {"area": "kontoret", "domain": ["light"]}, results)
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)

    await _interrupt(service.turn_liveness, emulated=True)
    finish.set()
    await task

    assert results == [{"result": "klart"}]
    assert service.turn_liveness.in_flight == 0


@pytest.mark.asyncio
async def test_interrupted_call_is_refused_when_repeated_within_60_seconds():
    service = _service()
    started = asyncio.Event()
    runs = []

    async def tool(params):
        runs.append(params.arguments)
        started.set()
        await asyncio.sleep(10)

    service.register_function("HassTurnOn", tool)
    arguments = {"area": "kontoret", "domain": ["light"]}
    first = asyncio.create_task(
        service._functions["HassTurnOn"].handler(
            _params("HassTurnOn", arguments, [])
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await _interrupt(service.turn_liveness, emulated=False)
    await _assert_cancelled(first)

    results = []
    await service._functions["HassTurnOn"].handler(
        _params("HassTurnOn", arguments, results)
    )

    assert runs == [arguments]
    assert results == [{
        "result": "Anropet avbröts av användaren och körs inte igen. Fråga vad han vill."
    }]


@pytest.mark.asyncio
async def test_a_different_call_still_runs_after_interruption():
    service = _service()
    started = asyncio.Event()
    runs = []

    async def tool(params):
        runs.append(params.arguments)
        if params.arguments["area"] == "kontoret":
            started.set()
            await asyncio.sleep(10)
        else:
            await params.result_callback({"result": "klart"})

    service.register_function("HassTurnOn", tool)
    first = asyncio.create_task(
        service._functions["HassTurnOn"].handler(
            _params("HassTurnOn", {"area": "kontoret", "domain": ["light"]}, [])
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await _interrupt(service.turn_liveness, emulated=False)
    await _assert_cancelled(first)

    results = []
    await service._functions["HassTurnOn"].handler(
        _params("HassTurnOn", {"area": "köket", "domain": ["light"]}, results)
    )

    assert runs == [
        {"area": "kontoret", "domain": ["light"]},
        {"area": "köket", "domain": ["light"]},
    ]
    assert results == [{"result": "klart"}]


@pytest.mark.asyncio
async def test_real_interruption_cancels_all_parallel_tools():
    service = _service()
    both_started = asyncio.Event()
    started = 0

    async def tool(params):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.sleep(10)

    service.register_function("parallel_tool", tool)
    tasks = [
        asyncio.create_task(
            service._functions["parallel_tool"].handler(
                _params("parallel_tool", {"number": number}, [])
            )
        )
        for number in (1, 2)
    ]
    await asyncio.wait_for(both_started.wait(), timeout=1.0)

    await _interrupt(service.turn_liveness, emulated=False)

    for task in tasks:
        await _assert_cancelled(task)
    assert service.turn_liveness.in_flight == 0


@pytest.mark.asyncio
async def test_a_cancellation_that_is_not_a_barge_in_still_propagates():
    service = _service()
    started = asyncio.Event()

    async def sleeping_tool(params):
        started.set()
        await asyncio.sleep(10)

    service.register_function("slow_tool", sleeping_tool)
    results = []
    task = asyncio.create_task(
        service._functions["slow_tool"].handler(_params("slow_tool", {}, results))
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()  # the session closing, not the person speaking

    with pytest.raises(asyncio.CancelledError):
        await task
    assert results == []
    assert service.turn_liveness.in_flight == 0


@pytest.mark.asyncio
async def test_a_call_already_answered_is_closed_quietly_by_a_barge_in():
    service = _service()
    started = asyncio.Event()

    async def tool(params):
        await params.result_callback({"result": "ok"})
        started.set()
        await asyncio.sleep(10)

    service.register_function("answered_tool", tool)
    results = []
    task = asyncio.create_task(service._functions["answered_tool"].handler(_params("answered_tool", {}, results)))
    await asyncio.wait_for(started.wait(), timeout=1.0)

    await _interrupt(service.turn_liveness, emulated=False)

    await _assert_cancelled(task)  # finishes without raising
    assert results == [{"result": "ok"}]


@pytest.mark.asyncio
async def test_interrupted_text_does_not_claim_nothing_was_done():
    service = _service()
    started = asyncio.Event()

    async def tool(params):
        started.set()
        await asyncio.sleep(10)

    service.register_function("HassTurnOn", tool)
    results = []
    task = asyncio.create_task(service._functions["HassTurnOn"].handler(_params("HassTurnOn", {"a": 1}, results)))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await _interrupt(service.turn_liveness, emulated=False)
    await _assert_cancelled(task)
    assert "Inget mer gjordes" not in results[0]["result"]
