"""Exercise DeviceRegistry targeting, identity and frame formatting."""
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.device_registry import (
    DeviceConnection,
    DeviceRegistry,
    device_id_from_websocket,
    sanitize_device_id,
)
from app.session_manager import SessionManager


class FakeURL:
    def __init__(self, query):
        self.query = query


class FakeClient:
    def __init__(self, host):
        self.host = host


class FakeWS:
    def __init__(self, query="", host=None):
        self.url = FakeURL(query)
        self.client = FakeClient(host) if host else None
        self.sent = []

    async def send_text(self, data):
        self.sent.append(data)


async def main():
    # --- identity ---------------------------------------------------------
    assert device_id_from_websocket(FakeWS("device_id=kitchen", "10.0.3.9")) == "kitchen"
    assert device_id_from_websocket(FakeWS("", "10.0.3.9")) == "10.0.3.9"
    assert device_id_from_websocket(FakeWS("", None)) == "unknown"
    # injection / overlong ids are reduced, not trusted
    assert sanitize_device_id('kit chen"; drop') == "kitchen drop".replace(" ", "")
    assert len(sanitize_device_id("x" * 500)) == 64
    print("identity      -> query param wins, IP fallback, hostile ids sanitized")

    # --- targeting --------------------------------------------------------
    reg = DeviceRegistry()
    kitchen = DeviceConnection("kitchen", FakeWS())
    await reg.add(kitchen)
    assert reg.resolve() is kitchen, "sole connected device must work before first activity"
    office = DeviceConnection("office", FakeWS())
    await reg.add(office)
    assert len(reg) == 2
    assert reg.ids() == ["kitchen", "office"]
    assert reg.resolve() is None, "idle devices must not receive implicit sends"

    kitchen.touch()
    await asyncio.sleep(0.01)
    office.touch()
    assert reg.resolve() is office, "no id -> most recently active"
    kitchen.touch()
    assert reg.resolve() is kitchen, "activity moves the default target"
    assert reg.resolve("office") is office, "explicit id wins"
    assert reg.resolve("bedroom") is None, "unknown explicit id must NOT fall back"
    print("targeting     -> explicit id wins; default follows activity; idle -> None")

    # An idle reconnect must not take over the default target merely because it
    # was constructed more recently.
    reconnect = DeviceConnection("bedroom", FakeWS())
    await reg.add(reconnect)
    assert reg.resolve() is kitchen, "idle reconnect stole the active target"
    await reg.remove(reconnect)

    # --- reconnect replaces the stale entry -------------------------------
    kitchen2 = DeviceConnection("kitchen", FakeWS())
    displaced = await reg.add(kitchen2)
    assert displaced is kitchen
    assert reg.get("kitchen") is kitchen2
    assert len(reg) == 2, "reconnect must not duplicate the device"

    # a late disconnect for the OLD socket must not evict the new one
    assert await reg.remove(kitchen) is False
    assert reg.get("kitchen") is kitchen2, "stale disconnect evicted the live connection!"
    assert await reg.remove(kitchen2) is True
    assert reg.get("kitchen") is None
    print("lifecycle     -> reconnect replaces; late disconnect can't evict the live socket")

    # A late cleanup for the displaced session must not remove the replacement
    # from SessionManager's per-device service map.
    sessions = SessionManager()
    old_service = object()
    new_service = object()
    sessions.set_current_service("kitchen", old_service)
    sessions.set_current_service("kitchen", new_service)
    sessions.handle_client_disconnect("kitchen", old_service)
    assert sessions.get_current_service("kitchen") is new_service
    sessions.handle_client_disconnect("kitchen", new_service)
    assert sessions.get_current_service("kitchen") is None
    print("sessions      -> stale cleanup preserves the replacement service")

    # --- frame formatting -------------------------------------------------
    ws = FakeWS()
    conn = DeviceConnection("kitchen", ws)
    await conn.send_phase("listening")
    assert ws.sent == ['{"type":"phase","value":"listening"}'], ws.sent
    assert '"value":"listening"' in ws.sent[0], "firmware does a literal substring match"
    print(f"frames        -> compact separators: {ws.sent[0]}")

    # --- unicast, not broadcast -------------------------------------------
    reg2 = DeviceRegistry()
    a, b = DeviceConnection("a", FakeWS()), DeviceConnection("b", FakeWS())
    await reg2.add(a)
    await reg2.add(b)
    await a.send_phase("replying")
    assert len(a.websocket.sent) == 1 and len(b.websocket.sent) == 0, "phase leaked to the other device"
    assert await reg2.broadcast_json({"type": "hello"}) == 2
    print("isolation     -> phase goes to one device; broadcast still reaches all")

    print("\nALL ASSERTIONS PASSED")


asyncio.run(main())
