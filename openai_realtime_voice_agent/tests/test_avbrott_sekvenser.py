try:
    import pytest
except ModuleNotFoundError:  # allows direct python execution in minimal envs
    class _Mark:
        def __getattr__(self, _name):
            def deco(fn):
                return fn

            return deco

    class _Pytest:
        mark = _Mark()

    pytest = _Pytest()

from tools import avbrott_sekvenser as mod


class _FakeSession:
    def __init__(self):
        self.calls = []

    async def send_realtime_input(self, **kwargs):
        if "activity_start" in kwargs:
            self.calls.append("activity_start")
        elif "activity_end" in kwargs:
            self.calls.append("activity_end")
        elif "audio" in kwargs:
            self.calls.append("audio")
        elif "text" in kwargs:
            self.calls.append("text")
        else:
            self.calls.append("other")


async def _no_sleep(_seconds: float):
    return None


@pytest.mark.asyncio
async def test_s1_signal_order():
    s = _FakeSession()
    await mod.run_signal_sequence(s, "S1", sleep=_no_sleep)
    assert s.calls[0] == "activity_start"
    assert s.calls[-1] == "activity_end"
    assert s.calls.count("audio") == mod.PCM_CHUNKS_600MS
    assert s.calls == ["activity_start"] + ["audio"] * mod.PCM_CHUNKS_600MS + ["activity_end"]


@pytest.mark.asyncio
async def test_s2_double_start_before_end():
    s = _FakeSession()
    await mod.run_signal_sequence(s, "S2", sleep=_no_sleep)
    assert s.calls[:2] == ["activity_start", "activity_start"]
    assert s.calls[-1] == "activity_end"
    assert s.calls.count("activity_start") == 2
    assert s.calls.count("activity_end") == 1
    assert s.calls.count("audio") == mod.PCM_CHUNKS_600MS
    assert s.calls == ["activity_start", "activity_start"] + ["audio"] * mod.PCM_CHUNKS_600MS + ["activity_end"]


@pytest.mark.asyncio
async def test_s3_empty_activity():
    s = _FakeSession()
    await mod.run_signal_sequence(s, "S3", sleep=_no_sleep)
    assert s.calls == ["activity_start", "activity_end"]


@pytest.mark.asyncio
async def test_s4_control_no_signal():
    s = _FakeSession()
    await mod.run_signal_sequence(s, "S4", sleep=_no_sleep)
    assert s.calls == []
