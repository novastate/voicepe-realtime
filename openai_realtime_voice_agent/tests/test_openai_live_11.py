"""gpt-live-1.1 prototype (branch live-gpt-11): protocol against a fake Live server, delegation against a fake backend.

Nothing here talks to api.openai.com or to Comms. Each test names the requirement ID it covers
(Bjorn_Live_Voice_Requirements v2.0). All of them fail on main (the engine did not exist there).
"""
import asyncio
import base64
import json

import pytest
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
)
from websockets.asyncio.server import serve

from app import live_delegering
from app.live_delegering import Delegering
from app.providers import GEMINI, OPENAI_LIVE_11, PROVIDERS, ProviderOptions, build_service
from app.providers import openai_live, openai_live_11
from app.providers.sovlage import Budget, SovlageMixin


@pytest.fixture(autouse=True)
def egna_liggare(tmp_path, monkeypatch):
    monkeypatch.setattr(SovlageMixin, "budget", Budget(path=str(tmp_path / "moln.json"), today=lambda: "d"))
    monkeypatch.setattr(openai_live, "OPENAI_BUDGET", Budget(path=str(tmp_path / "oa.json"), today=lambda: "d"))
    openai_live._Las.agare = None
    monkeypatch.setenv("OPENAI_LIVE_REPLY_GAP_MS", "1500")   # production's value: the interruption runs as on the device
    monkeypatch.setenv("LIVE11_PREBUF_S", "0")
    monkeypatch.setenv("LIVE11_TEXT_VANTA_MS", "20")
    yield
    openai_live._Las.agare = None


class FakeLive:
    def __init__(self):
        self.seen = []
        self.conns = []

    @property
    def conn(self):
        return self.conns[-1]

    async def handler(self, ws):
        self.conns.append(ws)
        async for raw in ws:
            ev = json.loads(raw)
            self.seen.append(ev)
            if ev["type"] == "session.start":
                await ws.send(json.dumps({"type": "session.started", "session": {"id": f"s{len(self.conns)}"}}))

    def of(self, typ):
        return [e for e in self.seen if e["type"] == typ]

    async def send(self, **ev):
        await self.conn.send(json.dumps(ev))


async def _server(fake):
    server = await serve(fake.handler, "127.0.0.1", 0)
    return server, f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"


class FakeBackend:
    """Comms /fraga as SSE frames; `hall` holds the answer until released; counts the calls (side effects)."""

    def __init__(self, svar="Det är tolv grader.", aktioner=None, fel=None, hall=False):
        self.fragor = []
        self.svar, self.aktioner, self.fel = svar, aktioner or [], fel
        self.slapp = asyncio.Event()
        if not hall:
            self.slapp.set()
        self.avbrutna = 0

    async def __call__(self, fraga):
        self.fragor.append(fraga)
        try:
            await self.slapp.wait()
        except asyncio.CancelledError:
            self.avbrutna += 1
            raise
        if self.fel:
            yield "error", {"code": self.fel}
            return
        yield "token", {"text": self.svar}
        yield "done", {"response": {"text": self.svar}, "actions": self.aktioner}


def _service(url, backend=None):
    opts = ProviderOptions(api_key="k", model="", voice="", instructions="Du är Björn, husets björn.")
    s = build_service(OPENAI_LIVE_11, opts, [{"type": "function", "name": "HassTurnOn"}])
    s._url = url
    s.frames = []
    if backend is not None:
        s.delegering.backend = backend

    async def push(frame, direction=None):
        s.frames.append(frame)

    s.push_frame = push
    s.create_task = lambda coro, name=None: asyncio.get_running_loop().create_task(coro)
    return s


async def _vant(cond, t=2.0):
    for _ in range(int(t / 0.02)):
        if cond():
            return True
        await asyncio.sleep(0.02)
    return cond()


def _ljud(n=480, tyst=False):
    return base64.b64encode((b"\x00\x00" if tyst else b"\x10\x01") * n).decode()


def _mic(byte=1, n=480):
    return InputAudioRawFrame(audio=bytes([byte, 0]) * n, sample_rate=24000, num_channels=1)


# --- selection and config -------------------------------------------------------------------------------------------

def test_flaggan_finns_och_gemini_ar_orord():
    from app.providers import input_sample_rate, self_heals, supports_client_events
    assert OPENAI_LIVE_11 in PROVIDERS and input_sample_rate(OPENAI_LIVE_11) == 24000
    assert self_heals(OPENAI_LIVE_11) and not supports_client_events(OPENAI_LIVE_11)
    assert not hasattr(build_service(GEMINI, ProviderOptions(api_key="k", model="", voice="", instructions="x"), []),
                       "live11")


def test_device_providers_valjer_per_enhet(monkeypatch):
    """Henrik 2026-11-28: the office tries 1.1, the kitchen stays on Gemini."""
    from app import providers
    from app.providers import device_provider
    monkeypatch.setattr(providers, "_ENHET_FEL", {})   # no pause left over from another test's failed wake
    monkeypatch.setenv("DEVICE_PROVIDERS", "koket=gemini,kontoret=openai_live_11")
    assert device_provider("kontoret") == OPENAI_LIVE_11
    assert device_provider("koket") == GEMINI


def test_session_start_ar_live_11_med_klientdelegering():
    """Contract per the SDK types: model, one audio.format, delegation client, no tools, policies after the persona."""
    s = _service("ws://x")
    cfg = s.session_config()
    assert cfg["model"] == "gpt-live-1.1"
    assert cfg["audio"] == {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "marin"}}
    assert cfg["delegation"] == {"type": "client", "acknowledgement_delay_ms": 800}
    assert "tools" not in json.dumps(cfg["delegation"])
    assert cfg["instructions"].startswith("Du är Björn, husets björn.")
    for rubrik in ("Backchannel policy:", "Interruption policy:", "Delegation policy:"):
        assert rubrik in cfg["instructions"]
    assert "input" not in cfg   # nothing to seed on the first session
    # No Realtime-only controls copied into Live (requirements section 3).
    for falt in ("turn_detection", "input_audio_buffer", "truncat", "response.create"):
        assert falt not in json.dumps(cfg)


# --- AUDIO-01 / CONV-02: continuous audio, no local turn gate ----------------------------------------------------------

@pytest.mark.asyncio
async def test_mikrofonen_strommas_aven_medan_bjorn_pratar():
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        s._fas_vad = None
        await s.vakna()
        await fake.send(type="session.output_audio.delta", delta=_ljud())
        assert await _vant(lambda: s._reply_open)
        await s.process_frame(_mic(7), None)
        assert await _vant(lambda: any(base64.b64decode(e["audio"])[:1] == b"\x07"
                                       for e in fake.of("session.input_audio.append")))
        await s.sova("klar")


@pytest.mark.asyncio
async def test_utan_mikrofon_skickas_tystnad_i_realtid():
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        await s.vakna()
        await asyncio.sleep(0.45)
        tyst = [e for e in fake.of("session.input_audio.append") if not any(base64.b64decode(e["audio"]))]
        assert 2 <= len(tyst) <= 6   # ~100 ms each, never faster than real time
        await s.sova("klar")


@pytest.mark.asyncio
async def test_det_som_sades_under_uppkopplingen_kommer_fram():
    """T19: audio before session.started is held (bounded) and sent once the session runs."""
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        s._fas_vad = None
        await s.process_frame(_mic(9), None)   # asleep: held
        await s.vakna()
        assert await _vant(lambda: any(base64.b64decode(e["audio"])[:1] == b"\x09"
                                       for e in fake.of("session.input_audio.append")))
        await s.sova("klar")


@pytest.mark.asyncio
async def test_rummet_skickas_inte_nar_ingen_vackt_honom():
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        s._fas_vad = None
        s.slapp_ljud = lambda: False
        await s.vakna()
        await s.process_frame(_mic(5), None)
        await asyncio.sleep(0.1)
        assert all(not any(base64.b64decode(e["audio"])) for e in fake.of("session.input_audio.append"))
        await s.sova("klar")


# --- AUDIO-03 / CONV-01: the app's own queue is dropped when Live yields ----------------------------------------------

@pytest.mark.asyncio
async def test_avbrott_slanger_gammalt_ljud_nar_live_tystnar_och_anvandaren_talar(monkeypatch):
    monkeypatch.setenv("LIVE11_TEXT_EFTER_S", "0.05")
    monkeypatch.setenv("LIVE11_AVBROTT_TYST_MS", "50")
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        s._fas_vad = None
        await s.vakna()
        await fake.send(type="session.output_audio.delta", delta=_ljud())
        await fake.send(type="session.output_transcript.delta", delta="Det var en gång", start_ms=0, end_ms=1)
        assert await _vant(lambda: s._reply_open)
        await asyncio.sleep(0.1)
        await fake.send(type="session.input_transcript.delta", delta="nej vänta", start_ms=1, end_ms=2)
        await asyncio.sleep(0.07)
        await fake.send(type="session.output_audio.delta", delta=_ljud(tyst=True))
        assert await _vant(lambda: any(isinstance(f, InterruptionFrame) for f in s.frames))
        assert any(isinstance(f, UserStartedSpeakingFrame) for f in s.frames)   # device clears its ring
        assert not s._reply_open and s.matning.avbrott == 1
        assert s.historik[-1] == {"roll": "assistant", "text": "Det var en gång", "status": "avbrutet"}
        await s.sova("klar")


@pytest.mark.asyncio
async def test_fragans_sena_transkript_avbryter_inte_svaret_den_fick():
    """The user's own question transcribed just after the answer began is not an interruption (CONV-02)."""
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        s._fas_vad = None
        await s.vakna()
        await fake.send(type="session.output_audio.delta", delta=_ljud())
        await fake.send(type="session.input_transcript.delta", delta="vad är klockan", start_ms=0, end_ms=1)
        await asyncio.sleep(0.25)
        await fake.send(type="session.output_audio.delta", delta=_ljud(tyst=True))
        await asyncio.sleep(0.1)
        assert not any(isinstance(f, InterruptionFrame) for f in s.frames)
        await s.sova("klar")


@pytest.mark.asyncio
async def test_ett_mm_medan_live_pausar_kort_avbryter_inte():
    """T04, the review's scenario: 0.9 s into the reply he says "mm", Live pauses 300 ms (zeros) and carries on.
    Nothing is dropped. Without the listening-sound rule this cut the reply (LIVE11_AVBROTT_TYST_BC_MS=0 shows it)."""
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        s._fas_vad = None
        await s.vakna()
        await fake.send(type="session.output_audio.delta", delta=_ljud(24000))   # 1 s of reply
        await asyncio.sleep(0.9)
        await fake.send(type="session.input_transcript.delta", delta="mm", start_ms=900, end_ms=1000)
        await asyncio.sleep(0.3)
        await fake.send(type="session.output_audio.delta", delta=_ljud(tyst=True))
        await asyncio.sleep(0.05)
        await fake.send(type="session.output_audio.delta", delta=_ljud(2400))
        await asyncio.sleep(0.1)
        assert not any(isinstance(f, InterruptionFrame) for f in s.frames)
        assert s._reply_open
        await s.sova("klar")


@pytest.mark.asyncio
async def test_samma_paus_med_riktiga_ord_ar_ett_avbrott():
    """The counterpart: the same timing with words that take the floor drops the old reply (CONV-01)."""
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        s._fas_vad = None
        await s.vakna()
        await fake.send(type="session.output_audio.delta", delta=_ljud(48000))   # 2 s of reply
        await asyncio.sleep(0.9)
        await fake.send(type="session.input_transcript.delta", delta="nej, nästa fredag", start_ms=900, end_ms=1000)
        await asyncio.sleep(0.1)
        await fake.send(type="session.output_audio.delta", delta=_ljud(tyst=True))
        assert await _vant(lambda: any(isinstance(f, InterruptionFrame) for f in s.frames))
        await s.sova("klar")


@pytest.mark.asyncio
async def test_avbrott_utan_lokal_detektor_stanger_lyssnar_lage(monkeypatch):
    """After a cut on the transcript alone the device is not left in "listening" (Grok, review 2026-11-28)."""
    from pipecat.frames.frames import UserStoppedSpeakingFrame
    monkeypatch.setenv("LIVE11_TEXT_EFTER_S", "0.05")
    monkeypatch.setenv("LIVE11_AVBROTT_TYST_MS", "50")
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        s = _service(url)
        s._fas_vad = None
        await s.vakna()
        await fake.send(type="session.output_audio.delta", delta=_ljud())
        await asyncio.sleep(0.1)
        await fake.send(type="session.input_transcript.delta", delta="vänta lite nu", start_ms=1, end_ms=2)
        await asyncio.sleep(0.1)
        await fake.send(type="session.output_audio.delta", delta=_ljud(tyst=True))
        assert await _vant(lambda: any(isinstance(f, InterruptionFrame) for f in s.frames))
        assert await _vant(lambda: any(isinstance(f, UserStoppedSpeakingFrame) for f in s.frames), 3.0)
        await s.sova("klar")


# --- TASK-01..03: client delegation through the backend ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delegering_gar_till_backend_och_resultatet_till_live():
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        be = FakeBackend(aktioner=[{"tool": "HassTurnOn", "target": "light.kontoret", "utfall": "ok"}])
        s = _service(url, be)
        await s.vakna()
        await fake.send(type="session.input_transcript.delta", delta="tänd i kontoret", start_ms=0, end_ms=900)
        await fake.send(type="session.delegation.created", event_id="e1", offset_ms=950,
                        delegation={"id": "del_1", "target": "client", "type": "delegation"})
        assert await _vant(lambda: fake.of("session.commentary.append"))
        assert be.fragor == ["tänd i kontoret"]
        ev = fake.of("session.commentary.append")[0]
        assert ev["delegation_id"] == "del_1" and ev["event_id"]
        assert "Det är tolv grader." in ev["content"] and "HassTurnOn light.kontoret: lyckades" in ev["content"]
        assert s.delegering.uppdrag[-1].status == "klar"
        await s.sova("klar")


@pytest.mark.asyncio
async def test_samma_order_igen_ger_ingen_ny_sidoeffekt():
    """T17/TASK-03: a repeated or retried order is not sent to the backend twice."""
    sent = []
    be = FakeBackend(aktioner=[{"tool": "HassTurnOn", "target": "light.kok", "utfall": "ok"}])
    d = Delegering(lambda ev: _samla(sent, ev), backend=be)
    u = await d.skapad("d1", "Tänd i köket.")
    await u.task
    assert await d.skapad("d2", "tänd i köket") is None
    assert len(be.fragor) == 1
    assert "ska inte göras igen" in sent[-1]["content"] and "lyckades" in sent[-1]["content"]


async def _samla(lista, ev):
    lista.append(ev)


@pytest.mark.asyncio
async def test_misslyckat_verktyg_pastar_aldrig_att_det_gick():
    """T12."""
    sent = []
    d = Delegering(lambda ev: _samla(sent, ev), backend=FakeBackend(fel="tidsgrans"))
    u = await d.skapad("d1", "Öppna persiennen.")
    await u.task
    assert u.status == "fel"
    assert sent[-1]["type"] == "session.commentary.append"
    assert "påstå inte att något är gjort" in sent[-1]["content"] and "tidsgrans" in sent[-1]["content"]


@pytest.mark.asyncio
async def test_utfall_som_inte_namns_ar_okant_inte_lyckat():
    rad = live_delegering.aktionsrad([{"tool": "lock", "target": "front", "utfall": "konstigt"}, {"tool": "x"}])
    assert "lyckades" not in rad and rad.count("okänt utfall") == 2


@pytest.mark.asyncio
async def test_rattelse_under_arbetet_far_sammanhanget_och_det_gamla_svaret_tystnar():
    """T10/T20/TASK-02: the correction carries the running task, the stale result is told silently."""
    sent = []
    gammal = FakeBackend(svar="Källor: BBC, Reuters.", hall=True)
    ny = FakeBackend(svar="Svenska källor: SVT, DN.")
    anrop = []

    def backend(fraga):
        anrop.append(fraga)
        return (gammal if len(anrop) == 1 else ny)(fraga)

    d = Delegering(lambda ev: _samla(sent, ev), backend=backend)
    u1 = await d.skapad("d1", "Sök nyheter om elpriset.")
    await asyncio.sleep(0)
    u2 = await d.skapad("d2", "bara svenska källor")
    await u2.task
    assert "Sök nyheter om elpriset." in anrop[1] and "bara svenska källor" in anrop[1]
    assert sent[-1]["type"] == "session.commentary.append" and "SVT" in sent[-1]["content"]
    gammal.slapp.set()
    await u1.task
    assert u1.status == "ersatt" and u1.ersatt_av == u2.op
    assert sent[-1]["type"] == "session.thinking.append" and "tidigare fråga" in sent[-1]["content"]


@pytest.mark.asyncio
async def test_ersatt_uppgift_med_atgard_berattas_anda():
    """TASK-03: a side effect is always told, also when its answer is otherwise stale."""
    sent = []
    gammal = FakeBackend(svar="Tänt.", aktioner=[{"tool": "HassTurnOn", "utfall": "ok"}], hall=True)
    d = Delegering(lambda ev: _samla(sent, ev), backend=lambda f: gammal(f) if not gammal.fragor else FakeBackend()(f))
    u1 = await d.skapad("d1", "Tänd i hallen.")
    await asyncio.sleep(0)
    await d.skapad("d2", "och i köket")
    gammal.slapp.set()
    await u1.task
    assert sent[-1]["type"] == "session.commentary.append" and "lyckades" in sent[-1]["content"]


@pytest.mark.asyncio
async def test_tyst_ar_inte_avbryt_men_avbryt_sokningen_ar_det():
    """T11: 'avbryt sökningen' cancels the running operation and says what happened; speech alone cancels nothing."""
    sent = []
    be = FakeBackend(hall=True)
    d = Delegering(lambda ev: _samla(sent, ev), backend=be)
    u = await d.skapad("d1", "Sök efter tågtider till Uppsala.")
    await asyncio.sleep(0)
    await d.skapad("d2", "Avbryt sökningen.")
    with pytest.raises(asyncio.CancelledError):
        await u.task
    assert u.status == "avbruten" and be.avbrutna == 1 and len(be.fragor) == 1
    assert "avbryta" in sent[-1]["content"] and "okänt" in sent[-1]["content"]


@pytest.mark.asyncio
async def test_glom_inte_ar_ett_tillagg_inte_ett_avbrott():
    sent = []
    be = FakeBackend(hall=True)
    d = Delegering(lambda ev: _samla(sent, ev), backend=be)
    u1 = await d.skapad("d1", "Tänd i köket.")
    await asyncio.sleep(0)
    assert await d.skapad("dx", "avbryt inte, fortsätt") is not None   # not a cancel either
    u2 = await d.skapad("d2", "och glöm inte hallen")
    assert u2 is not None and "glöm inte hallen" in u2.fraga
    assert not u1.task.cancelled() and u1.kor()
    be.slapp.set()
    await u1.task
    await u2.task


@pytest.mark.asyncio
async def test_avbryt_stoppar_aven_en_ersatt_uppgift_som_kor():
    """Review: a replaced task still runs; 'avbryt' must stop it too, or the user is told something false."""
    sent = []
    be = FakeBackend(hall=True)
    d = Delegering(lambda ev: _samla(sent, ev), backend=be)
    u1 = await d.skapad("d1", "Sök tågtider till Uppsala.")
    await asyncio.sleep(0)
    u2 = await d.skapad("d2", "bara direkttåg")
    await asyncio.sleep(0)
    await d.skapad("d3", "Avbryt sökningen.")
    for u in (u1, u2):
        with pytest.raises(asyncio.CancelledError):
            await u.task
        assert u.status == "avbruten"
    assert d.pagaende() == []


@pytest.mark.asyncio
async def test_omforsok_efter_en_foljdfraga_kor_inte_atgarden_igen():
    """Review: 'Tänd i hallen.' -> 'och i köket' -> 'Tänd i hallen.' again: the hall is not asked for twice."""
    sent = []
    be = FakeBackend(aktioner=[{"tool": "HassTurnOn", "utfall": "ok"}], hall=True)
    d = Delegering(lambda ev: _samla(sent, ev), backend=be)
    await d.skapad("d1", "Tänd i hallen.")
    await asyncio.sleep(0)
    await d.skapad("d2", "och i köket")
    await asyncio.sleep(0.01)
    assert await d.skapad("d3", "Tänd i hallen.") is None
    assert len(be.fragor) == 2
    be.slapp.set()
    for u in d.uppdrag:
        await u.task


@pytest.mark.asyncio
async def test_tva_delegeringar_tatt_far_var_sina_ord():
    """Review: two delegations close together each take the words up to their own offset_ms."""
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        be = FakeBackend()
        s = _service(url, be)
        await s.vakna()
        await fake.send(type="session.input_transcript.delta", delta="tänd i köket", start_ms=100, end_ms=900)
        await fake.send(type="session.delegation.created", event_id="e1", offset_ms=1000,
                        delegation={"id": "del_1", "target": "client", "type": "delegation"})
        await fake.send(type="session.input_transcript.delta", delta=" och vad är klockan", start_ms=1800, end_ms=2600)
        await fake.send(type="session.delegation.created", event_id="e2", offset_ms=2700,
                        delegation={"id": "del_2", "target": "client", "type": "delegation"})
        assert await _vant(lambda: len(be.fragor) == 2)
        assert be.fragor[0] == "tänd i köket"
        assert "vad är klockan" in be.fragor[1] and "tänd i köket" not in be.fragor[1].split("gäller nu:")[-1]
        await s.sova("klar")


def test_ekovakten_ar_kopplad_till_hans_senaste_replik():
    """The real wiring (koppla_live11): his own last words heard back over his reply are not delegated."""
    import time as _t
    from types import SimpleNamespace
    from app.websocket_handler import koppla_live11
    s = _service("ws://x")
    s.historik = [{"roll": "assistant", "text": "Lampan i köket är tänd nu.", "status": "klar"}]
    pe = SimpleNamespace(phase="replying", idle_since=_t.monotonic())
    ser = SimpleNamespace(sedan_svar=lambda: 0.5)
    conn = SimpleNamespace(recovery=None, barge_in_t=None)
    koppla_live11(s, conn, pe, ser)
    assert s.vakt("Lampan i köket är tänd nu.")
    assert s.vakt("släck i hallen") is None
    s._bjorn = "Jag låser upp dörren"   # still being said: his own words over the mic are not an order
    assert s.vakt("jag låser upp dörren")
    assert s.slapp_ljud() is True and s.svarar() is True
    s.on_motor_avbrott()
    assert conn.barge_tur is True


@pytest.mark.asyncio
async def test_ekovakten_stoppar_en_delegering_av_hans_egna_ord():
    sent = []
    be = FakeBackend()
    d = Delegering(lambda ev: _samla(sent, ev), backend=be, vakt=lambda t: "ignored: own reply")
    assert await d.skapad("d1", "lås upp dörren") is None
    assert be.fragor == [] and sent[-1]["type"] == "session.thinking.append"


@pytest.mark.asyncio
async def test_delegering_utan_text_ber_om_upprepning():
    sent = []
    be = FakeBackend()
    d = Delegering(lambda ev: _samla(sent, ev), backend=be)
    assert await d.skapad("d1", "  ") is None
    assert be.fragor == [] and sent[-1]["delegation_id"] is None


@pytest.mark.asyncio
async def test_langsam_backend_ger_en_lagesrad(monkeypatch):
    monkeypatch.setattr(live_delegering, "LANGSAM_S", 0.05)
    sent = []
    be = FakeBackend(hall=True)
    d = Delegering(lambda ev: _samla(sent, ev), backend=be)
    u = await d.skapad("d1", "Hur går det för Hammarby?")
    await asyncio.sleep(0.15)
    assert [e["content"] for e in sent].count(live_delegering.LANGSAM_TEXT) == 1
    be.slapp.set()
    await u.task


# --- T13 / CTX-01: link loss mid-conversation --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bortfall_ateransluter_med_historiken_och_gammalt_resultat_blir_kontext():
    fake = FakeLive()
    server, url = await _server(fake)
    async with server:
        be = FakeBackend(svar="Tåget går 14.05.", hall=True)
        s = _service(url, be)
        s._fas_vad = None
        await s.vakna()
        await fake.send(type="session.input_transcript.delta", delta="när går tåget", start_ms=0, end_ms=1)
        await fake.send(type="session.output_audio.delta", delta=_ljud())
        await fake.send(type="session.output_transcript.delta", delta="Jag kollar.", start_ms=1, end_ms=2)
        await fake.send(type="session.delegation.created", event_id="e", offset_ms=1,
                        delegation={"id": "del_a", "target": "client", "type": "delegation"})
        assert await _vant(lambda: be.fragor)
        await fake.conn.close()
        assert await _vant(lambda: len(fake.of("session.start")) == 2, 4.0)
        seed = fake.of("session.start")[1]["session"].get("input") or []
        texter = [i["content"][0]["text"] for i in seed]
        assert "när går tåget" in texter and any(t.startswith("Jag kollar.") for t in texter)
        assert seed[-1]["role"] == "developer" and "pagar" in seed[-1]["content"][0]["text"]
        be.slapp.set()
        assert await _vant(lambda: fake.of("session.thinking.append"))
        sist = fake.of("session.thinking.append")[-1]
        assert sist["delegation_id"] is None and "förra anslutningen" in sist["content"] and "14.05" in sist["content"]
        assert len(be.fragor) == 1   # not run again
        assert s.sover is False
        await s.sova("klar")


@pytest.mark.asyncio
async def test_journalraderna_innehaller_inga_ord(caplog):
    """OPS-02 on the lines actually written: a whole delegation and an interruption, and no user or backend words."""
    import logging
    caplog.set_level(logging.INFO)
    sent = []
    d = Delegering(lambda ev: _samla(sent, ev), backend=FakeBackend(svar="Hemligt svar om banken."), tagg="kontoret")
    u = await d.skapad("d1", "Vad är koden till larmet?")
    await u.task
    text = caplog.text
    assert "live11[kontoret] delegering slut" in text
    for ord_ in ("koden", "larmet", "Hemligt", "banken"):
        assert ord_ not in text


def test_telemetrin_innehaller_inga_ord():
    """OPS-02: lengths and ids, never the words."""
    rad = live_delegering.logga_json({"type": "session.commentary.append", "content": "lås upp dörren", "delegation_id": "d"})
    assert "lås" not in rad and "<14 tecken>" in rad
    m = openai_live_11.Matning()
    m.svarstid += [0.5, 0.7, 0.9]
    assert "p50=700ms" in m.rad()
