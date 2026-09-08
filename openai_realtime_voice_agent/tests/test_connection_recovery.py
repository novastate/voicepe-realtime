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


"""Recovery must know which engine it is nursing, and when not to nurse."""

import pytest

from app.provider_router import ProviderRouter
from pipecat.frames.frames import ErrorFrame as _ErrorFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection


class FakeService:
    def __init__(self):
        self.resets = 0

    async def reset_conversation(self):
        self.resets += 1


def _recovery(provider, router, switched):
    async def on_failover():
        switched.append(True)

    return ConnectionRecovery(
        FakeService(),
        provider=provider,
        router=router,
        on_failover=on_failover,
    )


@pytest.mark.asyncio
async def test_a_dropped_socket_on_openai_is_repaired_in_place():
    switched = []
    router = ProviderRouter("openai", "gemini")
    rec = _recovery("openai", router, switched)
    await rec.handle_error("keepalive ping timeout")
    assert rec._service.resets == 1
    assert switched == []


@pytest.mark.asyncio
async def test_gemini_repairs_itself_so_we_keep_our_hands_off():
    switched = []
    router = ProviderRouter("gemini", "openai")
    rec = _recovery("gemini", router, switched)
    await rec.handle_error("keepalive ping timeout")
    assert switched == []


@pytest.mark.asyncio
async def test_out_of_money_is_never_repaired_only_switched():
    switched = []
    router = ProviderRouter("openai", "gemini")
    rec = _recovery("openai", router, switched)
    await rec.handle_error("You exceeded your current quota")
    assert rec._service.resets == 0
    assert switched == [True]
    assert router.current() == "gemini"


@pytest.mark.asyncio
async def test_a_tool_failure_neither_repairs_nor_switches():
    switched = []
    router = ProviderRouter("openai", "gemini")
    rec = _recovery("openai", router, switched)
    await rec.handle_error("play_media failed: 500 Internal Server Error")
    assert rec._service.resets == 0
    assert switched == []
    assert router.current() == "openai"


@pytest.mark.asyncio
async def test_a_second_dropped_socket_switches_engine():
    switched = []
    router = ProviderRouter("openai", "gemini")
    rec = _recovery("openai", router, switched)
    await rec.handle_error("keepalive ping timeout")
    rec._last_attempt = 0.0  # step past the flood cooldown
    await rec.handle_error("keepalive ping timeout")
    assert switched == [True]


@pytest.mark.asyncio
async def test_out_of_money_reaches_the_router_through_an_ordinary_error_frame():
    """Ruling 2, made concrete: an out-of-money error carries none of the
    close-socket signatures (no "client event", no "session_expired", no
    "realtime receive loop") that used to gate whether an ErrorFrame got any
    decision at all. If process_frame still gated on those signatures before
    calling handle_error -- as the brief's own draft wiring did -- this
    ErrorFrame would fall straight to the plain idle-unstick and the router
    would never be asked, so switched would stay empty. Going through
    process_frame itself (not calling handle_error directly, like the tests
    above) is what actually exercises that wiring."""
    switched = []
    router = ProviderRouter("openai", "gemini")
    rec = _recovery("openai", router, switched)
    try:
        await rec.process_frame(
            _ErrorFrame("You exceeded your current quota"), FrameDirection.UPSTREAM
        )
        await rec._recover_task
        assert switched == [True]
        assert router.current() == "gemini"
    finally:
        await rec.close()


@pytest.mark.asyncio
async def test_a_finished_turn_resets_the_strike_budget_so_two_rare_hiccups_do_not_switch():
    """Ruling 3, made concrete: two dropped sockets with a real finished turn
    in between are two FIRST strikes, not a second-in-a-row, so the engine
    must stay put. Mutation check: delete the note_success call this proves
    exists, and this test starts failing (switched becomes [True]) exactly
    like test_a_second_dropped_socket_switches_engine above -- confirming the
    assertion actually depends on the reset, not on some incidental default."""
    switched = []
    router = ProviderRouter("openai", "gemini")
    rec = _recovery("openai", router, switched)
    try:
        await rec.handle_error("keepalive ping timeout")
        rec._last_attempt = 0.0  # step past the flood cooldown

        # A turn finishes cleanly on this engine in between the two hiccups.
        await rec.process_frame(
            TranscriptionFrame(text="hej", user_id="", timestamp="now"),
            FrameDirection.UPSTREAM,
        )

        await rec.handle_error("keepalive ping timeout")

        assert switched == []
        assert router.current() == "openai"
        assert rec._service.resets == 2
    finally:
        await rec.close()


@pytest.mark.asyncio
async def test_a_transcription_on_the_other_engine_does_not_clear_this_ones_strikes():
    """note_success must be keyed by the engine that is actually running, not
    fired blindly -- a late transcript from a stale session must not reset
    the NEW engine's clean slate. (ProviderRouter.note_success is itself
    keyed by provider name; this proves ConnectionRecovery passes its own
    _provider, not something else.)"""
    switched = []
    router = ProviderRouter("openai", "gemini")
    rec = _recovery("gemini", router, switched)  # this connection IS gemini
    router._strikes["gemini"] = 1  # simulate an earlier hiccup already charged
    try:
        await rec.process_frame(
            TranscriptionFrame(text="hej", user_id="", timestamp="now"),
            FrameDirection.UPSTREAM,
        )
        assert "gemini" not in router._strikes
    finally:
        await rec.close()
