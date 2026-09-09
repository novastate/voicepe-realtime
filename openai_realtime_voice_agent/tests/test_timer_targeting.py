"""Verify timer expiry stays scoped to the device that created it."""
import asyncio
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.timers as timers
from app.timers import TimerRegistry


async def main():
    os.environ["TIMER_RING_ENTITY"] = "switch.legacy_timer"
    os.environ["TIMER_RING_ENTITIES"] = (
        "kitchen=switch.kitchen_timer,office=switch.office_timer"
    )
    assert timers._ring_entity("office") == "switch.office_timer"
    assert timers._ring_entity("bedroom") == "switch.legacy_timer"
    assert timers._ring_entity("bedroom", allow_legacy=False) == ""

    registry = TimerRegistry()

    # Expiry is the bell alone, on the second: no spoken announcement, no
    # grace period. A 30 s kitchen timer used to be a voice at 30 s and a
    # chime at 50 s; the operator asked for the chime and nothing else.
    calls = []

    async def set_ring(on, device_id, allow_legacy=True):
        calls.append((on, device_id))
        return True

    timers._set_ring = set_ring
    timers.RING_AUTO_OFF_S = 0
    registry._timers[1] = {
        "device_id": "kitchen",
        "label": "pasta",
        "ends": time.monotonic(),
        "wall": time.time(),
        "task": asyncio.current_task(),
    }

    await registry._fire(1)

    assert calls == [(True, "kitchen"), (False, "kitchen")], calls
    assert registry._timers == {}

    # Voice tools must not expose or cancel timers from another room.
    first = registry.set_timer(60, "tea", device_id="kitchen")
    second = registry.set_timer(60, "coffee", device_id="office")
    kitchen_timers = registry.list_timers("kitchen")["timers"]
    assert len(kitchen_timers) == 1 and kitchen_timers[0]["id"] == first["id"]
    assert registry.cancel(None, "kitchen")["cancelled"] == first["id"]
    assert registry.cancel(second["id"], "kitchen") == {"error": f"no timer {second['id']}"}
    assert registry.list_timers("office")["timers"][0]["id"] == second["id"]
    registry.cancel(second["id"], "office")

    # A multi-device install must ring the timer's own device switch.
    ring_calls = []

    async def set_ring(on, device_id, allow_legacy=True):
        if not allow_legacy:
            return False
        ring_calls.append((on, device_id))
        return True

    timers._set_ring = set_ring
    timers.RING_AUTO_OFF_S = 0
    registry._timers[3] = {
        "device_id": "office", "label": "tea",
        "ends": time.monotonic(), "wall": time.time(), "task": asyncio.current_task(),
    }
    registry.announcer = None
    await registry._fire(3)
    assert ring_calls == [(True, "office"), (False, "office")]
    print("ring targeting -> timer ring stays in its originating room")

    registry.allow_legacy_ring = lambda _device_id: False
    registry._timers[4] = {
        "device_id": "bedroom", "label": "bread",
        "ends": time.monotonic(), "wall": time.time(), "task": asyncio.current_task(),
    }
    await registry._fire(4)
    assert ring_calls == [(True, "office"), (False, "office")]
    print("ring fallback  -> legacy switch is disabled with multiple devices")
    print("ALL ASSERTIONS PASSED")


asyncio.run(main())
