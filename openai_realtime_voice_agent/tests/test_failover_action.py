"""Switching engines means tearing the pipeline down and letting the device
call back. The device reconnects in about a second -- the same path every
add-on update already uses -- and the session is rebuilt with the other
engine, with the conversation restored from cache."""

import asyncio

import pytest

from app.websocket_handler import make_failover


class FakeTask:
    def __init__(self):
        self.cancelled = 0

    async def cancel(self):
        self.cancelled += 1


class FakeConnection:
    def __init__(self):
        self.task = FakeTask()
        self.device_id = "10.30.0.81"
        self.phases = []

    async def send_phase(self, value):
        self.phases.append(value)
        return True


@pytest.mark.asyncio
async def test_the_device_is_unstuck_before_the_pipeline_is_torn_down():
    # Otherwise the LED keeps blinking through the gap and the user thinks it
    # is still listening.
    c = FakeConnection()
    await make_failover(c)()
    assert c.phases == ["idle"]
    assert c.task.cancelled == 1


@pytest.mark.asyncio
async def test_a_second_call_does_nothing():
    # Several error frames arrive for the same death. One teardown is enough.
    c = FakeConnection()
    failover = make_failover(c)
    await failover()
    await failover()
    assert c.task.cancelled == 1


@pytest.mark.asyncio
async def test_a_connection_with_no_task_does_not_explode():
    c = FakeConnection()
    c.task = None
    await make_failover(c)()  # must not raise
