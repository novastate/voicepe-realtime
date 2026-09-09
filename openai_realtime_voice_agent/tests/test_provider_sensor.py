"""A silent switch that spends money on the wrong account for three days is
exactly the kind of fault nobody notices. So it gets an entity.

Every assertion here is checked against what a broken or half-written
`SensorPublisher.provider()` would produce -- loose checks like "starts
with sensor.voicepe_" pass even when the entity id, the reason, or the
retry countdown are wrong, so the exact expected values are asserted
instead of just their shape.
"""

import pytest

from app import ha_sensors
from app.provider_router import ProviderRouter


class FakeClock:
    """A clock the test moves by hand, so no test depends on wall time."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def _capture(monkeypatch):
    """Patch ha_sensors._post and return the list of posts it records."""
    posts = []

    async def fake_post(entity, state, attrs):
        posts.append({"entity": entity, "state": state, "attrs": attrs})

    monkeypatch.setattr(ha_sensors, "_post", fake_post)
    return posts


@pytest.mark.asyncio
async def test_the_sensor_says_which_engine_and_why(monkeypatch):
    posts = _capture(monkeypatch)

    clock = FakeClock()
    router = ProviderRouter("gemini", "openai", cooldown_s=1800.0, clock=clock)
    router.report_failure("gemini", "insufficient_quota")
    await ha_sensors.SensorPublisher().provider(router.status())

    assert len(posts) == 1
    posted = posts[0]
    # Exact entity id, not just a prefix/suffix match -- a sensor published
    # under the wrong instance name would still satisfy startswith/endswith.
    assert posted["entity"] == f"sensor.voicepe_{ha_sensors._INST}_motor"
    assert posted["state"] == "openai"
    assert "quota" in posted["attrs"]["reason"]
    assert posted["attrs"]["primary"] == "gemini"
    assert posted["attrs"]["backup"] == "openai"
    # No time has passed on the fake clock, so the full cooldown is still
    # owed -- an exact figure, not just "some positive number" (a hardcoded
    # positive constant would also satisfy > 0).
    assert posted["attrs"]["retry_primary_in_s"] == 1800.0


@pytest.mark.asyncio
async def test_a_healthy_house_reports_no_reason(monkeypatch):
    posts = _capture(monkeypatch)

    await ha_sensors.SensorPublisher().provider(ProviderRouter("openai", "gemini").status())

    assert len(posts) == 1
    posted = posts[0]
    assert posted["state"] == "openai"
    assert posted["attrs"]["reason"] == ""
    assert posted["attrs"]["retry_primary_in_s"] == 0.0


@pytest.mark.asyncio
async def test_repeat_publish_with_nothing_changed_is_skipped(monkeypatch):
    """Session builds happen on every device (re)connect, which is often.

    Publishing to Home Assistant on every one of those, when the engine
    hasn't actually changed, would spam the state machine for no reason.
    The publisher should write once and then stay quiet until something
    about the engine/reason/switch actually moves.
    """
    posts = _capture(monkeypatch)

    clock = FakeClock()
    router = ProviderRouter("openai", "gemini", cooldown_s=1800.0, clock=clock)
    publisher = ha_sensors.SensorPublisher()

    await publisher.provider(router.status())
    await publisher.provider(router.status())
    await publisher.provider(router.status())

    assert len(posts) == 1


@pytest.mark.asyncio
async def test_a_real_switch_is_never_swallowed_by_the_dedup(monkeypatch):
    """The de-dup must not eat an actual engine change."""
    posts = _capture(monkeypatch)

    clock = FakeClock()
    router = ProviderRouter("gemini", "openai", cooldown_s=1800.0, clock=clock)
    publisher = ha_sensors.SensorPublisher()

    await publisher.provider(router.status())  # healthy: gemini
    router.report_failure("gemini", "insufficient_quota")
    await publisher.provider(router.status())  # switched: openai

    assert len(posts) == 2
    assert posts[0]["state"] == "gemini"
    assert posts[1]["state"] == "openai"
    assert posts[1]["attrs"]["reason"] != ""
