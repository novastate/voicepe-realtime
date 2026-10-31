"""Kept (rescued) audio turns are purged by age on every store, also for devices that never return."""
import time
from types import SimpleNamespace

from app.websocket_handler import WebSocketHandler


def test_stada_raddade_drops_old_turns_of_other_devices():
    h = object.__new__(WebSocketHandler)
    nu = time.monotonic()
    h._raddade = {
        "borta": ([b"x"], nu - WebSocketHandler.RADDAD_TUR_MAX_ALDER_S - 1),
        "farsk": ([b"y"], nu - 1),
    }
    h._stada_raddade()
    assert list(h._raddade) == ["farsk"]


import pytest


@pytest.mark.asyncio
async def test_teardown_lagring_stadar_gamla_tur_hos_andra_enheter():
    nu = time.monotonic()
    tjanst = SimpleNamespace(sover=True, take_rescue=lambda: [b"a"])
    conn = SimpleNamespace(device_id="koket", openai_service=tjanst, task=None, ha_tools_task=None, recovery=None,
                           phase_emitter=None, maskin=None, pipeline=None, runner=None)
    hw = SimpleNamespace(_raddade={"borta": ([b"x"], nu - 999)}, RADDAD_TUR_MAX_ALDER_S=20.0)
    hw._stada_raddade = lambda: WebSocketHandler._stada_raddade(hw)
    await WebSocketHandler._teardown(hw, conn)
    assert list(hw._raddade) == ["koket"]
