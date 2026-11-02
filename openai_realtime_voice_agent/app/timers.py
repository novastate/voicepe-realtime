"""Voice timers: backend-owned timer registry with on-device ringing.

The Voice PE firmware keeps its stock timer ring machinery (chime + LED,
silenced by the button or the "stop" word) behind a `timer_ringing` switch
that is exposed to Home Assistant. The backend owns the actual timers: the
model calls set/cancel/list tools, an asyncio task fires at expiry, and the
ring is triggered by flipping that switch via the HA API (TIMER_RING_ENTITY
option). If no entity is configured the assistant simply announces expiry is
unavailable rather than pretending.

Expiry is the bell alone, on the second — no spoken announcement.

Timers survive session refreshes and restarts using an atomic JSON store at
/data/timers.json (override with TIMER_STORE_PATH). Restore uses absolute wall
deadlines, ringing missed expiries up to 15 minutes late and dropping older ones.
If storage is unavailable, timers continue in memory.
"""
import asyncio
import json
import math
from pathlib import Path
import tempfile
import logging
import os
import time
from typing import Any, Awaitable, Callable, Dict, Optional, TYPE_CHECKING

import httpx

from app import ha_api

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

logger = logging.getLogger(__name__)

RESTORE_GRACE_S = 900
MAX_TIMERS = 10
MAX_DURATION_S = 24 * 3600
RING_AUTO_OFF_S = 120  # stop ringing after 2 min if nobody silences it


def _ring_entity(device_id: str, allow_legacy: bool = True) -> str:
    mappings = os.environ.get("TIMER_RING_ENTITIES", "")
    for mapping in mappings.split(","):
        key, separator, entity = mapping.partition("=")
        if separator and key.strip() == device_id:
            return entity.strip()
    return os.environ.get("TIMER_RING_ENTITY", "").strip() if allow_legacy else ""


async def _set_ring(on: bool, device_id: str = "", allow_legacy: bool = True) -> bool:
    entity = _ring_entity(device_id, allow_legacy)
    if not entity or not ha_api.configured():
        return False
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                ha_api.url(f"/services/switch/turn_{'on' if on else 'off'}"),
                headers=ha_api.headers(),
                json={"entity_id": entity},
            )
            r.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"⚠️ timer ring toggle failed ({entity}): {e!r}")
        return False


class TimerRegistry:
    def __init__(self, store_path=None):
        self.store_path = Path(store_path if store_path is not None else os.environ.get("TIMER_STORE_PATH", "/data/timers.json"))
        self._store_warned = False
        self._timers: Dict[int, dict] = {}
        self._next_id = 1
        # Wired by main.py. Takes the timer's device ID so an expiry rings in
        # the room where it was created.
        self.allow_legacy_ring = None

    def _save(self):
        temporary = None
        try:
            data = {"next_id": self._next_id, "timers": [
                {"id": tid, **{k: t[k] for k in ("device_id", "label", "wall")}}
                for tid, t in self._timers.items()]}
            with tempfile.NamedTemporaryFile(mode="w", dir=self.store_path.parent,
                                             prefix=self.store_path.name + ".", delete=False) as f:
                temporary = f.name
                json.dump(data, f, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.store_path)
        except OSError as exc:
            if not self._store_warned:
                logger.warning("Timer store unavailable (%s): %s; using memory", self.store_path, exc)
                self._store_warned = True
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def restore(self, now_wall=None):
        """Restore timers using wall deadlines; requires a running loop."""
        loop = asyncio.get_running_loop()
        for t in self._timers.values():
            t["task"].cancel()
        self._timers.clear()
        now = time.time() if now_wall is None else now_wall
        try:
            data = json.loads(self.store_path.read_text())
            if (not isinstance(data, dict) or not isinstance(data.get("timers"), list)
                    or type(data.get("next_id")) is not int or data["next_id"] < 1):
                raise ValueError("invalid timer store shape")
            ids = set()
            for t in data["timers"]:
                if (not isinstance(t, dict) or type(t.get("id")) is not int
                        or t["id"] < 1 or t["id"] in ids
                        or not isinstance(t.get("device_id"), str)
                        or not isinstance(t.get("label"), str)
                        or type(t.get("wall")) not in (int, float)
                        or not math.isfinite(t["wall"])):
                    raise ValueError("invalid stored timer")
                ids.add(t["id"])
        except FileNotFoundError:
            data = {"next_id": self._next_id, "timers": []}
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            logger.warning("Cannot restore timer store (%s): %s", self.store_path, exc)
            data = {"next_id": self._next_id, "timers": []}
        self._next_id = max(self._next_id, data["next_id"],
                            max((t["id"] + 1 for t in data["timers"]), default=1))
        for t in data["timers"]:
            remaining = t["wall"] - now
            if remaining < -RESTORE_GRACE_S:
                logger.info("Dropping old expired timer %s ('%s')", t["id"], t["label"])
                continue
            tid = t["id"]
            self._timers[tid] = {**t, "ends": time.monotonic() + max(0, remaining),
                                 "task": loop.create_task(self._fire(tid))}
        self._save()

    def _prune(self):
        now = time.monotonic()
        expired = [t for t, v in self._timers.items() if v["ends"] <= now and v["task"].done()]
        for tid in expired:
            self._timers.pop(tid, None)
        if expired:
            self._save()

    async def _fire(self, tid: int):
        t = self._timers.get(tid)
        if not t:
            return
        try:
            await asyncio.sleep(max(0.0, t["ends"] - time.monotonic()))
        except asyncio.CancelledError:
            return
        self._timers.pop(tid, None)
        self._save()
        logger.info(f"⏰ timer {tid} ('{t['label']}') expired")
        # The bell, on the second, and nothing else. There used to be a spoken
        # announcement first and a 20 s grace before the bell — so a 30 s
        # kitchen timer was a voice at 30 s and a chime at 50 s. Asked for and
        # removed 2026-09-09: "det räcker med chime på rätt tid". A timer is
        # the one thing in this house that must be exact, and a sentence read
        # by a different engine in a different voice was never the point.
        logger.info(f"⏰ timer {tid} escalating to ring")
        allow_legacy = (
            self.allow_legacy_ring(t["device_id"])
            if self.allow_legacy_ring is not None else True
        )
        if await _set_ring(True, t["device_id"], allow_legacy):
            await asyncio.sleep(RING_AUTO_OFF_S)
            await _set_ring(False, t["device_id"], allow_legacy)

    def set_timer(self, seconds: int, label: str, device_id: str = "") -> dict:
        self._prune()
        if sum(t["device_id"] == device_id for t in self._timers.values()) >= MAX_TIMERS:
            return {"error": "too many timers running"}
        seconds = max(5, min(int(seconds), MAX_DURATION_S))
        tid = self._next_id
        self._next_id += 1
        self._timers[tid] = {
            "device_id": device_id,
            "label": label or f"timer {tid}",
            "ends": time.monotonic() + seconds,
            "wall": time.time() + seconds,
            "task": asyncio.get_running_loop().create_task(self._fire(tid)),
        }
        self._save()
        logger.info(f"⏰ timer {tid} set: {seconds}s ('{label}')")
        try:
            import asyncio as _a
            from .ha_sensors import PUBLISHER
            _a.get_running_loop().create_task(PUBLISHER.timers(self))
        except Exception:
            pass
        return {"id": tid, "label": self._timers[tid]["label"], "seconds": seconds}

    def cancel(self, tid: Optional[int], device_id: str = "") -> dict:
        self._prune()
        timers = self.list_timers(device_id)["timers"]
        if tid is None:
            if len(timers) == 1:
                tid = timers[0]["id"]
            elif not timers:
                return {"error": "no timers running"}
            else:
                return {"error": "multiple timers running — need the id",
                        "timers": timers}
        t = self._timers.get(int(tid))
        if not t or t["device_id"] != device_id:
            return {"error": f"no timer {tid}"}
        del self._timers[int(tid)]
        t["task"].cancel()
        self._save()
        logger.info(f"⏰ timer {tid} cancelled")
        try:
            import asyncio as _a
            from .ha_sensors import PUBLISHER
            _a.get_running_loop().create_task(PUBLISHER.timers(self))
        except Exception:
            pass
        return {"cancelled": tid, "label": t["label"]}

    def list_timers(self, device_id: Optional[str] = None) -> dict:
        self._prune()
        now = time.monotonic()
        return {"timers": [
            {"id": tid, "label": t["label"], "seconds_left": int(t["ends"] - now)}
            for tid, t in sorted(self._timers.items())
            if device_id is None or t["device_id"] == device_id
        ]}


def get_timer_tool_definitions() -> list:
    return [
        {"type": "function", "name": "set_timer",
         "description": ("Set a countdown timer. The device will ring when it expires "
                         "(user silences it with the button or by saying 'stop'). "
                         "Use for any 'set a timer', 'remind me in N minutes' request."),
         "parameters": {"type": "object", "properties": {
             "seconds": {"type": "integer", "description": "Duration in seconds"},
             "label": {"type": "string", "description": "Short label, e.g. 'pasta'"}},
             "required": ["seconds"]}},
        {"type": "function", "name": "cancel_timer",
         "description": "Cancel a running timer. Omit id if only one is running.",
         "parameters": {"type": "object", "properties": {
             "id": {"type": "integer", "description": "Timer id (optional)"}}}},
        {"type": "function", "name": "list_timers",
         "description": "List running timers with time remaining.",
         "parameters": {"type": "object", "properties": {}}},
    ]


def register_timer_tools(llm, registry: "TimerRegistry", device_id: str) -> None:
    async def _set(params: "FunctionCallParams") -> None:
        a = params.arguments or {}
        await params.result_callback(registry.set_timer(
            a.get("seconds", 0), (a.get("label") or "").strip(), device_id
        ))

    async def _cancel(params: "FunctionCallParams") -> None:
        a = params.arguments or {}
        await params.result_callback(registry.cancel(a.get("id"), device_id))

    async def _list(params: "FunctionCallParams") -> None:
        await params.result_callback(registry.list_timers(device_id))

    llm.register_function("set_timer", _set)
    llm.register_function("cancel_timer", _cancel)
    llm.register_function("list_timers", _list)
