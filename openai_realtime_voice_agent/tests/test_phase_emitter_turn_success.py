"""Fix 4: the frame that means "the assistant finished answering" is
BotStoppedSpeakingFrame, but it never reaches ConnectionRecovery directly --
PhaseEmitter sits downstream of it in the pipeline and is the one that
already decides, with its debounce and in-flight-tool checks, when a reply is
genuinely over. This proves PhaseEmitter fires the success callback exactly
when that real end-of-turn idle happens, and never for a forced/aborted one.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.phase_emitter import PhaseEmitter, TurnLiveness


async def _noop(_value):
    return None


@pytest.mark.asyncio
async def test_a_clean_reply_signals_turn_success():
    calls = []
    pe = PhaseEmitter(_noop, idle_debounce_s=0)
    pe.set_turn_success_handler(lambda: calls.append(True))
    await pe._emit_idle_after_debounce()
    assert calls == [True]


@pytest.mark.asyncio
async def test_a_tool_still_running_does_not_signal_turn_success():
    """The debounce elapsing while a tool call is still in flight means the
    turn is NOT over -- PhaseEmitter shows `thinking` instead of `idle` here,
    so success must not fire either."""
    calls = []
    liveness = TurnLiveness()
    liveness.tool_started()
    pe = PhaseEmitter(_noop, idle_debounce_s=0, liveness=liveness)
    pe.set_turn_success_handler(lambda: calls.append(True))
    try:
        await pe._emit_idle_after_debounce()
        assert calls == []
    finally:
        # The "thinking" branch arms the watchdog task -- close() cancels it
        # so it doesn't stay pending after the test ends.
        await pe.close()


@pytest.mark.asyncio
async def test_a_forced_idle_does_not_signal_turn_success():
    """force_idle is the turn-DEATH path (rate limit, reconnect, watchdog) --
    the opposite of a clean finish. It must never be mistaken for success."""
    calls = []
    pe = PhaseEmitter(_noop, idle_debounce_s=0)
    pe.set_turn_success_handler(lambda: calls.append(True))
    await pe.force_idle("turn declared dead")
    assert calls == []
