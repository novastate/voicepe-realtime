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
from pipecat.frames.frames import ErrorFrame as _ErrorFrame
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
    """Fix 6: the old version of this test asserted only `switched == []`,
    which still passes if the self_heals guard is deleted (nothing here has a
    backup to switch to, so switched stays empty regardless). The actual
    headline behaviour -- Gemini's own reconnect logic is left alone -- is
    that reset_conversation is never called; assert that directly."""
    switched = []
    router = ProviderRouter("gemini", "openai")
    rec = _recovery("gemini", router, switched)
    await rec.handle_error("keepalive ping timeout")
    assert switched == []
    assert rec._service.resets == 0


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
    rec._last_attempt = 0.0  # step past the flood-collapse cooldown
    await rec.handle_error("keepalive ping timeout")
    assert switched == [True]


@pytest.mark.asyncio
async def test_out_of_money_reaches_the_router_through_an_ordinary_error_frame():
    """Ruling 2, made concrete: an out-of-money error carries none of the
    close-socket signatures (no "client event", no "session_expired", no
    "realtime receive loop") that used to gate whether an ErrorFrame got any
    decision at all. If process_frame still gated on those signatures before
    calling handle_error, this ErrorFrame would fall straight to the plain
    idle-unstick and the router would never be asked, so switched would stay
    empty. Going through process_frame itself (not calling handle_error
    directly, like the tests above) is what actually exercises that wiring."""
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
async def test_a_rate_limit_gets_only_an_idle_nudge_never_a_reconnect():
    """Fix 1: classify() returns TRANSIENT for a rate limit -- and for any
    message it does not recognise at all -- not only for a dead socket. Using
    "not APP" as the repair trigger (what an earlier version of handle_error
    did) would call reset_conversation() on a merely-rate-limited but
    perfectly live session. Only an actual connection-death signature
    (_is_dead_socket) may repair; everything else must fall through to the
    exact old behaviour: reported to the router (so a money/auth failure
    hiding behind similar wording can still switch engines), never repaired,
    idle nudge fires so the device does not hang in `thinking`."""
    switched = []
    router = ProviderRouter("openai", None)
    rec = _recovery("openai", router, switched)
    idled = []

    async def fake_unstick(msg):
        idled.append(msg)

    rec._unstick_idle = fake_unstick
    try:
        await rec.process_frame(
            _ErrorFrame("Rate limit reached for requests"), FrameDirection.UPSTREAM
        )
        await rec._recover_task
        assert rec._service.resets == 0
        assert idled == ["Rate limit reached for requests"]
    finally:
        await rec.close()


@pytest.mark.asyncio
async def test_a_flood_of_identical_errors_is_handled_once():
    """Fix 2: a dead socket delivers the SAME message ~15x/s. In production
    process_frame's own `_reconnecting` guard stops most of that flood from
    ever reaching handle_error again -- but there is a real scheduling gap
    between creating the task and it actually setting `_reconnecting`, and
    calling handle_error directly (as this test does) removes that guard
    entirely. Without the window here, each call would re-report to the
    router -- accruing its own strike (switching engines after the first
    *detected* drop, not the second *real* one) and re-attempting
    reset_conversation for a connection we just finished reconnecting.
    Three identical calls in a row must be handled only once."""
    switched = []
    router = ProviderRouter("openai", "gemini")
    rec = _recovery("openai", router, switched)
    await rec.handle_error("keepalive ping timeout")
    await rec.handle_error("keepalive ping timeout")
    await rec.handle_error("keepalive ping timeout")
    # If each call had been handled separately, three TRANSIENT reports would
    # exceed the one-retry budget and switch engines, and reset_conversation
    # would have been called three times.
    assert switched == []
    assert router._strikes.get("openai") == 1
    assert rec._service.resets == 1


@pytest.mark.asyncio
async def test_a_different_message_right_after_is_not_deduped():
    """The flood-collapse window is keyed by message text, not merely by
    time, so a genuinely different failure right after a first one must still
    be reported (mutation check for "keyed by message": if the dedup ignored
    the message and only checked the time window, this second, different
    failure would be silently dropped too)."""
    switched = []
    router = ProviderRouter("openai", "gemini")
    rec = _recovery("openai", router, switched)
    await rec.handle_error("keepalive ping timeout")
    await rec.handle_error("You exceeded your current quota")
    assert switched == [True]
    assert router.current() == "gemini"


@pytest.mark.asyncio
async def test_a_failover_decision_with_no_switch_callback_yet_still_lets_the_idle_nudge_fire():
    """Fix 3: on_failover is None until Task 8 wires it in. If handle_error
    returned True whenever the router decided to switch -- regardless of
    whether anything could act on that decision -- the device would get
    neither an engine switch (nothing rebuilds the connection) nor an idle
    nudge (True suppresses it): stuck in `thinking` forever."""
    router = ProviderRouter("openai", "gemini")
    rec = ConnectionRecovery(FakeService(), provider="openai", router=router, on_failover=None)
    acted = await rec.handle_error("You exceeded your current quota")
    assert acted is False
    assert router.current() == "gemini"  # the router still switched...
    assert rec._service.resets == 0


@pytest.mark.asyncio
async def test_note_turn_success_resets_the_strike_budget_under_the_shipped_config():
    """Fix 4: with this house's shipped config (transcription_language empty)
    OpenAI never emits a TranscriptionFrame at all, so the reset signal
    cannot depend on one. note_turn_success() is what PhaseEmitter calls when
    a reply genuinely finishes (wired regardless of transcription config), so
    calling it directly here proves the reset works with no transcription
    frame anywhere in the picture -- exactly the shipped default. Mutation
    check: without note_turn_success clearing the strike, this test fails
    exactly like test_a_second_dropped_socket_switches_engine passes (switched
    becomes [True]) -- confirming the assertion depends on the reset."""
    switched = []
    router = ProviderRouter("openai", "gemini")
    rec = _recovery("openai", router, switched)
    await rec.handle_error("keepalive ping timeout")
    rec._last_attempt = 0.0  # step past the flood-collapse cooldown

    # A turn finishes cleanly on this engine in between the two hiccups --
    # never a TranscriptionFrame, exactly like OpenAI with transcription off.
    rec.note_turn_success()

    await rec.handle_error("keepalive ping timeout")

    assert switched == []
    assert router.current() == "openai"
    assert rec._service.resets == 2


@pytest.mark.asyncio
async def test_note_turn_success_only_clears_this_connections_own_engine():
    """note_success must be keyed by the engine that is actually running, not
    fired blindly -- a success signal from a stale/other connection must not
    reset an unrelated engine's strikes. (ProviderRouter.note_success is
    itself keyed by provider name; this proves ConnectionRecovery passes its
    own _provider, not something else.)"""
    switched = []
    router = ProviderRouter("openai", "gemini")
    rec = _recovery("gemini", router, switched)  # this connection IS gemini
    router._strikes["gemini"] = 1  # simulate an earlier hiccup already charged
    router._strikes["openai"] = 1  # a DIFFERENT engine's charge -- must survive
    rec.note_turn_success()
    assert "gemini" not in router._strikes
    assert router._strikes.get("openai") == 1


@pytest.mark.asyncio
async def test_a_successful_wedge_repair_resets_the_strike_budget():
    """Fix 4's second half: "let a successful repair reset the budget too."
    force_reconnect's wedge repair never goes through handle_error/
    report_failure at all (a wedge is detected positively, from silence, not
    from an ErrorFrame) -- so crediting it immediately, unlike a reactive
    handle_error repair, cannot erase a strike that a second real failure
    still needed to be measured against."""
    switched = []
    router = ProviderRouter("openai", "gemini")
    rec = _recovery("openai", router, switched)
    router._strikes["openai"] = 1  # a prior hiccup already charged
    rec._last_attempt = 0.0
    await rec.force_reconnect("wedge: silent after wake")
    assert "openai" not in router._strikes
    assert rec._service.resets == 1
