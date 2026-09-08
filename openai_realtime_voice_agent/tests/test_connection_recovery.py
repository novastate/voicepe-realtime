"""Verify connection teardown stops ConnectionRecovery background work."""
import asyncio
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.device_registry import DeviceConnection
from app.providers.openai_realtime import SafeRealtimeLLMService
from app.phase_emitter import PhaseEmitter
from app.websocket_handler import ConnectionRecovery, WebSocketHandler
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


async def main():
    recovery = ConnectionRecovery(object())
    refresh_task = asyncio.create_task(asyncio.Event().wait())
    recover_task = asyncio.create_task(asyncio.Event().wait())
    recovery._refresh_task = refresh_task
    recovery._recover_task = recover_task
    phase_emitter = PhaseEmitter(None)
    phase_task = asyncio.create_task(asyncio.Event().wait())
    phase_emitter._idle_task = phase_task

    connection = DeviceConnection(
        "kitchen", object(), recovery=recovery, phase_emitter=phase_emitter
    )
    await WebSocketHandler()._teardown(connection)

    assert refresh_task.cancelled()
    assert recover_task.cancelled()
    assert phase_task.cancelled()
    assert connection.recovery is None
    assert connection.phase_emitter is None

    class FakeRecovery:
        def __init__(self):
            self.reasons = []

        async def force_reconnect(self, reason):
            self.reasons.append(reason)

    handler = WebSocketHandler()
    handler.WEDGE_TIMEOUT_S = 0
    wedge_recovery = FakeRecovery()
    wedge_connection = DeviceConnection("kitchen", object(), recovery=wedge_recovery)
    wedge_phase = PhaseEmitter(None)
    await handler._wedge_check(wedge_connection, wedge_phase, 1.0)
    assert wedge_recovery.reasons == ["wedge: silent after wake"]

    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingService:
        async def reset_conversation(self):
            started.set()
            await release.wait()

    live_recovery = ConnectionRecovery(BlockingService())
    live_connection = DeviceConnection("office", object(), recovery=live_recovery)
    wedge_task = asyncio.create_task(
        handler._wedge_check(live_connection, PhaseEmitter(None), 1.0)
    )
    await started.wait()
    await handler._teardown(live_connection)
    try:
        await wedge_task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("wedge recovery survived connection teardown")

    # reset_conversation deliberately closes the old receive task. That must
    # not emit a second connection-death ErrorFrame into a stopped processor.
    service = object.__new__(SafeRealtimeLLMService)
    service._resetting_conversation = True
    errors = []

    async def receive_ended(_service):
        return None

    async def push_error(**kwargs):
        errors.append(kwargs)

    service.push_error = push_error
    with patch.object(OpenAIRealtimeLLMService, "_receive_task_handler", receive_ended):
        await service._receive_task_handler()
    assert errors == []
    print("ALL ASSERTIONS PASSED")


asyncio.run(main())
