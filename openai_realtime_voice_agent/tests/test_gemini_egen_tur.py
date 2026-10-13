"""GEMINI_EGEN_TUR=1 (raawr US-060): the engine's own activity detection runs the turns; ours is the gate and the guard."""
import pytest
from pipecat.frames.frames import InterruptionFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame

from app.providers import build_service
from test_gemini_provider import OPENAI_SHAPE, _ScriptedTurns, _frame, _options, _wire


def _bygg(monkeypatch, events):
    monkeypatch.setenv("GEMINI_EGEN_TUR", "1")
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    sent = _wire(service, None)
    service._fas_vad = _ScriptedTurns(events)
    service.sover = False
    pushed = []

    async def push(frame, direction=None):
        pushed.append(frame)
    service.push_frame = push
    return service, sent, pushed


def test_the_engine_gets_its_own_activity_detection(monkeypatch):
    monkeypatch.setenv("GEMINI_EGEN_TUR", "1")
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    assert service._turns is None and service._settings["vad"].disabled is not True
    monkeypatch.setenv("GEMINI_EGEN_TUR", "0")
    assert build_service("gemini", _options(), OPENAI_SHAPE)._settings["vad"].disabled is True
    monkeypatch.setenv("GEMINI_EGEN_TUR", "kontoret")   # one room: the others keep local turns and bana 0
    o = _options(); o.rum = "kontoret"
    assert build_service("gemini", o, OPENAI_SHAPE)._turns is None
    o = _options(); o.rum = "koket"
    assert build_service("gemini", o, OPENAI_SHAPE)._turns is not None


@pytest.mark.asyncio
async def test_every_frame_is_streamed_with_no_activity_markers(monkeypatch):
    service, sent, pushed = _bygg(monkeypatch, [None, "start", None, "end", None])
    for b in range(5):
        await service._send_user_audio(_frame(b + 1))
    assert len(sent) == 5 and all("activity_start" not in kw and "activity_end" not in kw for kw in sent)
    assert [type(f) for f in pushed] == [UserStartedSpeakingFrame, UserStoppedSpeakingFrame]  # only for the device's phases


@pytest.mark.asyncio
async def test_nothing_is_sent_when_nobody_woke_him(monkeypatch):
    service, sent, pushed = _bygg(monkeypatch, ["start", None])
    service.slapp_ljud = lambda: False
    await service._send_user_audio(_frame(1))
    await service._send_user_audio(_frame(2))
    assert sent == [] and pushed == []


@pytest.mark.asyncio
async def test_over_his_reply_only_the_engine_says_that_somebody_spoke(monkeypatch):
    service, sent, pushed = _bygg(monkeypatch, ["start", "end"])
    service.svarar = lambda: True
    await service._send_user_audio(_frame(1))
    await service._send_user_audio(_frame(2))
    assert len(sent) == 2 and pushed == []   # the engine hears it all; the local detector cuts nothing
    avbrott = []
    service.on_motor_avbrott = lambda: avbrott.append(1)
    await service._motor_avbrott()
    assert avbrott == [1]
    assert [type(f) for f in pushed if isinstance(f, (InterruptionFrame, UserStartedSpeakingFrame))] == [InterruptionFrame, UserStartedSpeakingFrame]


@pytest.mark.asyncio
async def test_interrupted_from_the_engine_reaches_the_handler(monkeypatch):
    service, sent, pushed = _bygg(monkeypatch, [])
    hort = []

    async def avbrott():
        hort.append(1)
    service._motor_avbrott = avbrott

    class M:
        def __init__(self, i):
            self.server_content = type("S", (), {"interrupted": i})()

    class Session:
        def receive(self):
            async def g():
                for m in (M(None), M(True), M(False)):
                    yield m
            return g()

    async def ingen(self, session):   # pipecat's own part needs a live socket
        return None
    monkeypatch.setattr("pipecat.services.google.gemini_live.llm.GeminiLiveLLMService._handle_session_ready", ingen)
    s = Session()
    await service._handle_session_ready(s)
    assert len([m async for m in s.receive()]) == 3 and hort == [1]


@pytest.mark.asyncio
async def test_a_sleeping_engine_is_woken_by_speech_and_gets_the_start_of_it(monkeypatch):
    service, sent, pushed = _bygg(monkeypatch, [None, "start", None])
    service.sover = True

    async def vakna():
        service.sover = False
        return True
    service.vakna = vakna
    for b in range(3):
        await service._send_user_audio(_frame(b + 1))
    assert not service.sover and len(sent) == 2
    assert len(sent[0]["audio"].data) == 2 * 640   # the frame before the speech and the first of it, in one piece
