"""US-384: persisted timers retain their room and wall-clock deadline."""
import asyncio
import json
import logging
import time

import pytest

from app import timers


@pytest.fixture
def rings(monkeypatch):
    calls = []

    async def ring(on, device_id, allow_legacy=True):
        if on:
            calls.append((device_id, time.monotonic()))
        return True

    monkeypatch.setattr(timers, "_set_ring", ring)
    monkeypatch.setattr(timers, "RING_AUTO_OFF_S", 0)
    return calls


def stop(registry):
    for timer in registry._timers.values():
        timer["task"].cancel()


def test_ac1_deadline_survives_restart(tmp_path, monkeypatch, rings):
    async def scenario():
        path = tmp_path / "timers.json"
        monkeypatch.setattr(timers.time, "time", lambda: 1000.0)
        original = timers.TimerRegistry(store_path=path)
        first = original.set_timer(600, "pasta", "koket")
        stop(original)
        restored = timers.TimerRegistry(store_path=path)
        restored.restore(now_wall=1150)
        assert restored.list_timers("koket")["timers"][0]["seconds_left"] == pytest.approx(450, abs=1)
        assert restored._timers[first["id"]]["wall"] == 1600
        assert restored.set_timer(60, "tea", "koket")["id"] > first["id"]
        stop(restored)

        # Use the same persisted deadline with only one real second left.
        data = json.loads(path.read_text())
        data["timers"] = [t for t in data["timers"] if t["id"] == first["id"]]
        path.write_text(json.dumps(data))
        short = timers.TimerRegistry(store_path=path)
        start = time.monotonic()
        short.restore(now_wall=1599)
        await asyncio.wait_for(asyncio.gather(*(t["task"] for t in short._timers.values())), 3)
        assert len(rings) == 1 and rings[0][0] == "koket"
        assert rings[0][1] - start == pytest.approx(1, abs=2)
        assert short.list_timers()["timers"] == []
        assert json.loads(path.read_text())["timers"] == []
        again = timers.TimerRegistry(store_path=path)
        again.restore(now_wall=1601)
        await asyncio.sleep(0.01)
        assert len(rings) == 1
    asyncio.run(scenario())


def test_ac2_recent_expiry_rings_old_expiry_dropped(tmp_path, rings, caplog):
    async def scenario():
        path = tmp_path / "timers.json"
        path.write_text(json.dumps({"next_id": 9, "timers": [
            {"id": 2, "device_id": "koket", "label": "recent", "wall": 1880},
            {"id": 8, "device_id": "office", "label": "old", "wall": 1099},
        ]}))
        registry = timers.TimerRegistry(store_path=path)
        with caplog.at_level(logging.INFO, logger="app.timers"):
            registry.restore(now_wall=2000)
            await asyncio.gather(*(t["task"] for t in registry._timers.values()))
        assert [room for room, _ in rings] == ["koket"]
        assert registry.list_timers()["timers"] == []
        assert json.loads(path.read_text())["timers"] == []
        assert "8" in caplog.text and "drop" in caplog.text.lower()
    asyncio.run(scenario())


def test_ac3_cancel_persists(tmp_path, rings):
    async def scenario():
        path = tmp_path / "timers.json"
        registry = timers.TimerRegistry(store_path=path)
        timer = registry.set_timer(600, "pasta", "koket")
        registry.cancel(timer["id"], "koket")
        restored = timers.TimerRegistry(store_path=path)
        restored.restore(now_wall=time.time() + 601)
        await asyncio.sleep(0.01)
        assert restored.list_timers()["timers"] == []
        assert rings == []
    asyncio.run(scenario())


@pytest.mark.parametrize("broken", ["garbage", '{"timers":', '[]', '{"timers": [null]}'])
def test_ac4_corrupt_store_recovers_atomically(tmp_path, rings, caplog, broken):
    async def scenario():
        path = tmp_path / "timers.json"
        path.write_text(broken)
        registry = timers.TimerRegistry(store_path=path)
        with caplog.at_level(logging.WARNING, logger="app.timers"):
            registry.restore()
        assert registry.list_timers()["timers"] == []
        assert caplog.records
        registry.set_timer(60, "tea", "koket")
        data = json.loads(path.read_text())
        assert data["timers"][0]["device_id"] == "koket"
        assert set(data["timers"][0]) == {"id", "device_id", "label", "wall"}
        assert list(tmp_path.iterdir()) == [path]
        stop(registry)
    asyncio.run(scenario())


def test_unwritable_store_keeps_timers_in_memory(tmp_path, monkeypatch, rings, caplog):
    async def scenario():
        path = tmp_path / "timers.json"
        monkeypatch.setenv("TIMER_STORE_PATH", str(path))
        registry = timers.TimerRegistry()

        def denied(*args):
            raise PermissionError("read-only directory")

        monkeypatch.setattr(timers.os, "replace", denied)
        with caplog.at_level(logging.WARNING, logger="app.timers"):
            first = registry.set_timer(60, "tea", "koket")
            registry.set_timer(60, "pasta", "koket")
            registry.cancel(first["id"], "koket")
        assert len(registry.list_timers("koket")["timers"]) == 1
        assert len([r for r in caplog.records if "store unavailable" in r.message]) == 1
        assert list(tmp_path.iterdir()) == []
        stop(registry)
    asyncio.run(scenario())
