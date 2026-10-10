"""OpenAI Live (raawr US-025): the protocol against a fake server, and the money rules.

Nothing here talks to api.openai.com: a websockets server on 127.0.0.1:0
plays OpenAI. Each test fails on 0.27.0 (the engine did not exist).
"""
import asyncio
import base64
import json

import pytest
from pipecat.frames.frames import TTSAudioRawFrame, LLMFullResponseEndFrame
from websockets.asyncio.server import serve

from app.providers import OPENAI_LIVE, PROVIDERS, ProviderOptions, build_service
from app.providers import openai_live
from app.providers.sovlage import Budget, SovlageMixin


@pytest.fixture(autouse=True)
def egna_liggare(tmp_path, monkeypatch):
    """Never the real ledgers; no lock left over from another test."""
    monkeypatch.setattr(SovlageMixin, "budget", Budget(path=str(tmp_path / "moln.json"), today=lambda: "d"))
    monkeypatch.setattr(openai_live, "OPENAI_BUDGET", Budget(path=str(tmp_path / "oa.json"), today=lambda: "d"))
    openai_live._Las.agare = None
    monkeypatch.setenv("OPENAI_LIVE_REPLY_GAP_MS", "100")
    yield
    openai_live._Las.agare = None


class FakeLive:
    """Records every client event; answers session.start; can push server events."""

    def __init__(self, started=True):
        self.seen = []
        self.started = started
        self.conn = None

    async def handler(self, ws):
        self.conn = ws
        async for raw in ws:
            ev = json.loads(raw)
            self.seen.append(ev)
            if ev["type"] == "session.start" and self.started:
                await ws.send(json.dumps({"type": "session.started", "session": {"id": "sess_1"}}))

    def types(self):
        return [e["type"] for e in self.seen]


async def _server(fake):
    server = await serve(fake.handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"ws://127.0.0.1:{port}"


def _service(url, tools=()):
    opts = ProviderOptions(api_key="k", model="", voice="", instructions="Du är Björn.")
    s = build_service(OPENAI_LIVE, opts, list(tools))
    s._url = url
    s.frames = []

    async def push(frame, direction=None):
        s.frames.append(frame)

    s.push_frame = push
    # No pipeline runs here, so no pipecat task manager: plain asyncio tasks.
    s.create_task = lambda coro, name=None: asyncio.get_running_loop().create_task(coro)
    return s


async def _vant(cond, t=2.0):
    for _ in range(int(t / 0.02)):
        if cond():
            return True
        await asyncio.sleep(0.02)
    return cond()


def test_motorn_finns_och_tar_24_khz():
    from app.providers import input_sample_rate, supports_client_events, self_heals
    assert OPENAI_LIVE in PROVIDERS
    assert input_sample_rate(OPENAI_LIVE) == 24000
    assert not supports_client_events(OPENAI_LIVE) and self_heals(OPENAI_LIVE)


def test_session_start_har_modellen_och_verktygen():
    opts = ProviderOptions(api_key="k", model="", voice="", instructions="Hej.")
    cfg = openai_live.session_config(opts, [{"type": "function", "name": "HassTurnOn",
                                             "description": "tänd", "parameters": {"type": "object"}}])
    assert cfg["model"] == "gpt-live-1"
    assert cfg["delegation"]["type"] == "responses"
    assert cfg["delegation"]["responses"]["tools"][0]["name"] == "HassTurnOn"


@pytest.mark.asyncio
async def test_sover_tills_vakning_och_vantar_pa_session_started():
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        assert s.sover is True
        await s._connect()  # what pipecat's start() does: nothing while asleep
        assert fake.seen == []
        assert await s.vakna() is True
        assert fake.types() == ["session.start"]
        assert fake.seen[0]["session"]["model"] == "gpt-live-1"
        await s.sova("tyst")
        assert await _vant(lambda: "session.close" in fake.types())


@pytest.mark.asyncio
async def test_utan_session_started_somnar_den_igen():
    fake = FakeLive(started=False)
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        s._ar_uppkopplad_orig = s._ar_uppkopplad

        async def kort():
            return await s._ar_uppkopplad_orig(timeout=0.2)

        s._ar_uppkopplad = kort
        assert await s.vakna() is False
        assert s.sover is True and openai_live._Las.agare is None


@pytest.mark.asyncio
async def test_ljud_in_och_ut_och_svaret_tar_slut():
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        s._turns = None  # stream straight through
        done = []

        async def klar():
            done.append(1)

        s.set_turn_complete_handler(klar)
        await s.vakna()
        await s._append(b"\x01\x02" * 10)
        assert await _vant(lambda: "session.input_audio.append" in fake.types())
        ev = [e for e in fake.seen if e["type"] == "session.input_audio.append"][0]
        assert base64.b64decode(ev["audio"]) == b"\x01\x02" * 10
        await fake.conn.send(json.dumps({"type": "session.output_audio.delta",
                                         "delta": base64.b64encode(b"\x00\x01" * 240).decode()}))
        assert await _vant(lambda: any(isinstance(f, TTSAudioRawFrame) for f in s.frames))
        audio = [f for f in s.frames if isinstance(f, TTSAudioRawFrame)][0]
        assert audio.sample_rate == 24000
        assert await _vant(lambda: done == [1])
        assert any(isinstance(f, LLMFullResponseEndFrame) for f in s.frames)
        await s.sova("klar")


@pytest.mark.asyncio
async def test_verktyg_via_delegering_och_resultat_tillbaka():
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        anrop = []

        async def handler(params):
            anrop.append((params.function_name, params.arguments))
            await params.result_callback({"ok": True})

        s.register_function("vaderprognos", handler)
        await s.vakna()
        await fake.conn.send(json.dumps({"type": "response.event", "delegation_id": "d1", "event": {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": "c1", "name": "vaderprognos", "arguments": "{\"dag\": 1}"}}}))
        assert await _vant(lambda: "response.create" in fake.types())
        out = [e for e in fake.seen if e["type"] == "response.item.create"][0]["item"]
        assert out == {"type": "function_call_output", "call_id": "c1", "output": "{\"ok\": true}"}
        assert anrop == [("vaderprognos", {"dag": 1})]
        await s.sova("klar")


class Turer:
    def __init__(self, events):
        self.events = list(events)

    def feed(self, pcm, rate=None):
        return self.events.pop(0) if self.events else None

    def reset(self):
        pass


def _frame(i):
    from pipecat.frames.frames import InputAudioRawFrame
    return InputAudioRawFrame(audio=bytes([i]) * 320, sample_rate=16000, num_channels=1)


@pytest.mark.asyncio
async def test_bana0_traff_nar_aldrig_live_miss_gor_det():
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        await s.vakna()
        beslut = {}

        async def tur_slut():
            if beslut["traff"]:
                await s.drop_turn()
            else:
                await s.answer_turn()

        s.on_user_turn_end = tur_slut
        beslut["traff"] = True
        s._turns = Turer(["start", None, "end"])
        for i in range(3):
            await s._send_user_audio(_frame(i + 1))
        await asyncio.sleep(0.2)
        assert "session.input_audio.append" not in fake.types()  # Live never heard the order

        beslut["traff"] = False
        s._turns = Turer(["start", None, "end"])
        for i in range(3):
            await s._send_user_audio(_frame(i + 1))
        assert await _vant(lambda: fake.types().count("session.input_audio.append") >= 3)
        await s.sova("klar")


@pytest.mark.asyncio
async def test_en_session_at_gangen():
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        a, b = _service(url), _service(url)
        assert await a.vakna() is True
        assert await b.vakna() is False and b.vagran == "las"
        await a.sova("klar")
        assert await b.vakna() is True
        await b.sova("klar")


@pytest.mark.asyncio
async def test_openai_taket_sager_nej(monkeypatch):
    monkeypatch.setenv("OPENAI_MAX_MINUTER_PER_DAG", "6")
    openai_live.OPENAI_BUDGET.lagg_till(6 * 60)
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        assert await s.vakna() is False and s.vagran == "budget"
        assert fake.seen == []


@pytest.mark.asyncio
async def test_sova_raknar_openai_minuter():
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        await s.vakna()
        s._uppkopplad_sedan -= 120
        await s.sova("klar")
        assert 119 < openai_live.OPENAI_BUDGET.anvant() < 125


def test_kedjan_tar_openai_live(monkeypatch):
    from app.main import build_router
    monkeypatch.setenv("VOICE_PROVIDERS", "gemini,openai_live,xai")
    assert build_router().chain == ["gemini", "openai_live", "xai"]


@pytest.mark.asyncio
async def test_nekad_vakning_flyttar_hogtalaren_till_nasta_motor():
    from unittest.mock import AsyncMock
    from app.provider_router import ProviderRouter
    from app.websocket_handler import ConnectionRecovery

    class Nekad(SovlageMixin):
        sover = True
        vagran = None

        async def vakna(self):
            self.vagran = "las"
            return False

    router = ProviderRouter("openai_live", "gemini", probe=lambda e: True)
    flytt = AsyncMock()
    r = ConnectionRecovery(Nekad(), provider="openai_live", router=router, on_failover=flytt)
    await r.vakna()
    await asyncio.sleep(0.05)
    flytt.assert_awaited_once()
    assert router.current() == "gemini"


# --- review of US-025 (adversarial, 2026-10-04) ---

@pytest.mark.asyncio
async def test_produktionsvagen_somnar_nar_session_started_uteblir():
    """Through ConnectionRecovery.vakna with its real 5 s limit: the engine's own
    3 s wait must end first, so the engine is asleep and the lock free again."""
    from app.websocket_handler import ConnectionRecovery

    fake = FakeLive(started=False)
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        r = ConnectionRecovery(s, provider=OPENAI_LIVE)
        await r.vakna()
        assert s.sover is True and openai_live._Las.agare is None


@pytest.mark.asyncio
async def test_yttre_tidsgrans_soever_motorn():
    """Even if the outer limit fires first, the engine goes back to sleep."""
    from app.websocket_handler import ConnectionRecovery

    fake = FakeLive(started=False)
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        r = ConnectionRecovery(s, provider=OPENAI_LIVE)
        r.VAKNA_TIMEOUT_S = 0.3  # shorter than the engine's own 3 s wait
        await r.vakna()
        assert s.sover is True and openai_live._Las.agare is None


@pytest.mark.asyncio
async def test_nedrivning_raknar_minuterna():
    from app.device_registry import DeviceConnection
    from app.websocket_handler import WebSocketHandler

    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        assert await s.vakna() is True
        s._uppkopplad_sedan -= 120  # two minutes connected
        conn = DeviceConnection(device_id="kontoret", websocket=object(), serializer=None)
        conn.openai_service = s
        await WebSocketHandler()._teardown(conn)
        assert openai_live._Las.agare is None  # the lock is released; minutes counted (the machine closes it)
        assert openai_live.OPENAI_BUDGET.anvant() >= 119


@pytest.mark.asyncio
async def test_svaret_tar_inte_slut_medan_ett_verktyg_kor():
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        s._turns = None
        done = []

        async def klar():
            done.append(1)

        s.set_turn_complete_handler(klar)
        await s.vakna()
        s._verktyg_pagar = 1  # a delegated tool is running
        await fake.conn.send(json.dumps({"type": "session.output_audio.delta",
                                         "delta": base64.b64encode(b"\x00\x01" * 240).decode()}))
        assert await _vant(lambda: any(isinstance(f, TTSAudioRawFrame) for f in s.frames))
        await asyncio.sleep(0.3)  # past the (test) 100 ms gap
        assert done == []
        await s.sova("klar")


def test_sessionen_startas_med_det_riktiga_endpointen_godkanner(monkeypatch):
    """Probed against /v1/live/sessions 2026-10-10: one audio.format for both directions, no audio.input."""
    from types import SimpleNamespace

    opts = SimpleNamespace(model="gpt-live-1", instructions="x", voice="marin", max_output_tokens=None)
    verktyg = [{"name": "HassTurnOn", "description": "d", "parameters": {"type": "object", "properties": {}}}]
    monkeypatch.delenv("OPENAI_LIVE_DELEGATION", raising=False)
    cfg = openai_live.session_config(opts, verktyg)
    assert cfg["audio"]["format"] == {"type": "audio/pcm", "rate": 24000} and "input" not in cfg["audio"]
    assert cfg["delegation"]["type"] == "responses" and cfg["delegation"]["responses"]["tools"][0]["name"] == "HassTurnOn"
    monkeypatch.setenv("OPENAI_LIVE_DELEGATION", "client")
    assert openai_live.session_config(opts, verktyg)["delegation"] == {"type": "client"}


@pytest.mark.asyncio
async def test_tystnaden_i_stromen_ar_inget_svar():
    """Live streams zeros between answers (probed live 2026-10-10); only real sound opens a reply."""
    s = _service("ws://x")
    await s._audio_out(b"\x00" * 4800)
    assert s.frames == [] and s._reply_open is False
    await s._audio_out(bytes([5, 1]) * 2400)
    assert any(isinstance(f, TTSAudioRawFrame) for f in s.frames) and s._reply_open is True
    s._reply_end_task.cancel()
