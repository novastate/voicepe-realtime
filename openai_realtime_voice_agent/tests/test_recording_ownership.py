"""Verify a failed connection setup releases its recording claim."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.websocket_handler import WebSocketHandler


class FakeURL:
    query = "device_id=kitchen"


class FakeWebSocket:
    url = FakeURL()
    client = SimpleNamespace(host="127.0.0.1")

    async def accept(self):
        return None


class FakeTransport:
    def event_handler(self, _name):
        def register(fn):
            return fn
        return register


class FakeRecordingService:
    def __init__(self):
        self.started = []
        self.stopped = 0

    def start_new_session(self, device_id):
        self.started.append(device_id)

    def stop_recording(self):
        self.stopped += 1


async def main():
    recorder = FakeRecordingService()
    handler = WebSocketHandler(audio_recording_service=recorder)
    handler.create_transport = lambda *_args: FakeTransport()

    async def factory(_connection):
        return object()

    def fail_after_claim(connection, _activity_callback):
        connection.records_audio = handler._claim_recording(connection.device_id)
        raise RuntimeError("pipeline setup failed")

    handler.openai_service_factory = factory
    handler.build_pipeline = fail_after_claim
    await handler.serve_connection(FakeWebSocket())

    assert recorder.started == ["kitchen"]
    assert recorder.stopped == 1
    assert handler._recording_owner is None
    print("ALL ASSERTIONS PASSED")


asyncio.run(main())
