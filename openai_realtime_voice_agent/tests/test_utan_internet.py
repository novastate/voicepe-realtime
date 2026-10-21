"""Without internet (raawr US-018): bana 0 still confirms, a question gets an honest no."""
import asyncio
from types import SimpleNamespace
import time

import pytest

from app import bana0


@pytest.fixture(autouse=True)
def fyllnadsreplikerna_pa(monkeypatch):
    """The fillers are off by default since 0.27.12; these tests are about the net
    around them (the silence ack, the local turn's ack) with them on."""
    monkeypatch.setenv("EARLY_ACK", "1")

PCM = b"\x00" * 3200


async def _stt(pcm, timeout):
    return "tänd kontoret"


def _tur(prova_svar, skapa_svar=None, efter_miss=None, efter_traff=None, monkeypatch=None):
    async def prova(text, timeout):
        return prova_svar

    monkeypatch.setattr(bana0, "prova", prova)

    async def skicka(text):
        pass

    async def skapa():
        pass

    return bana0.tur(
        PCM, stt=_stt, timeout_stt=1.0, timeout_comms=1.0,
        skicka_svar_till_modellen=skicka, skapa_svar=skapa_svar or skapa,
        efter_miss=efter_miss, efter_traff=efter_traff,
    )


@pytest.mark.asyncio
async def test_traff_vaktar_bekraftelsen(monkeypatch):
    vakt = []
    assert await _tur("Slog på lampan", efter_traff=lambda: vakt.append(1),
                      monkeypatch=monkeypatch) == "bana0"
    assert vakt == [1]


async def _bekrafta(claimed):
    said = []

    async def say_ok():
        said.append(bana0.OK_FALLBACK)

    sagt = await bana0.vakta_bekraftelse(vanta_s=0.01, claim=lambda: claimed, say_ok=say_ok)
    return sagt, said


@pytest.mark.asyncio
async def test_modellen_tyst_efter_traff_ger_klart():
    assert await _bekrafta(True) == (True, [bana0.OK_FALLBACK])


@pytest.mark.asyncio
async def test_modellen_bekraftade_sjalv_inget_klart():
    assert await _bekrafta(False) == (False, [])


@pytest.mark.asyncio
async def test_miss_dar_modellen_inte_nas_kraschar_inte_och_vaktar(monkeypatch):
    calls = []

    async def skapa():
        raise RuntimeError("no socket")

    bana = await _tur(None, skapa_svar=skapa,
                      efter_miss=lambda: calls.append("vakt"), monkeypatch=monkeypatch)
    assert bana == "modell"
    assert calls == ["vakt"]


async def _vakta(nere, claimed=True, comms=True):
    said = []

    async def natet_nere():
        if isinstance(nere, Exception):
            raise nere
        return nere

    async def say(text):
        said.append(text)

    async def comms_lever():
        if isinstance(comms, Exception):
            raise comms
        return comms

    sagt = await bana0.vakta_natet(
        natet_nere=natet_nere, claim=lambda: claimed, say=say, kolla_comms=comms_lever,
    )
    return sagt, said


@pytest.mark.asyncio
async def test_comms_uppe_ger_offlinevariant_som_forut():
    sagt, said = await _vakta(True, comms=True)
    assert sagt is True and len(said) == 1 and said[0] in bana0.OFFLINE_VARIANTER


@pytest.mark.asyncio
async def test_comms_nere_ger_inget_lofte_om_lamporna():
    sagt, said = await _vakta(True, comms=False)
    assert sagt is True and len(said) == 1
    assert not any(lofte in said[0].lower() for lofte in ("fungerar", "funkar", "går att styra"))


@pytest.mark.asyncio
async def test_comms_probe_som_hanger_ger_replik_inom_taket(monkeypatch):
    monkeypatch.setenv("HA_API_URL", "http://comms.test/kanal/rost/kontoret/api")
    monkeypatch.setenv("COMMS_NYCKEL", "k")
    async def hanger(*args, **kwargs):
        await asyncio.sleep(2)

    monkeypatch.setattr(bana0.httpx.AsyncClient, "get", hanger)
    start = time.monotonic()
    sagt, said = await _vakta(True, comms=await bana0.comms_lever(0.05))
    assert time.monotonic() - start < 0.5
    assert sagt is True and said[0] in bana0.OFFLINE_UTAN_LOFTE


@pytest.mark.asyncio
async def test_natet_uppe_ingen_replik():
    assert await _vakta(False) == (False, [])


@pytest.mark.asyncio
async def test_modellen_eller_ogonblick_tog_platsen_ingen_replik():
    assert await _vakta(True, claimed=False) == (False, [])


def test_replikerna_ar_inga_fragor():
    assert all("?" not in r for r in bana0.LOKALA_REPLIKER)


# --- natet_nere: every engine, in parallel, a hang is a dead network ---

@pytest.mark.asyncio
async def test_alla_motorer_nere_ar_natet_nere():
    assert await bana0.natet_nere(lambda e: False, ("xai", "gemini"), 0.5) is True


@pytest.mark.asyncio
async def test_en_motor_uppe_ar_inte_natet_nere():
    # xAI down, Gemini up: the router fails over; the room must not hear "no net".
    assert await bana0.natet_nere(lambda e: e == "gemini", ("xai", "gemini"), 0.5) is False


@pytest.mark.asyncio
async def test_probe_som_hanger_raknas_som_nere_inom_tiden():
    start = time.monotonic()
    assert await bana0.natet_nere(lambda e: time.sleep(2) or True, ("xai", "gemini"), 0.3) is True
    assert time.monotonic() - start < 1.0  # parallel, and bounded by the timeout


@pytest.mark.asyncio
async def test_probe_som_kraschar_raknas_som_nere():
    def probe(engine):
        raise OSError("dns")
    assert await bana0.natet_nere(probe, ("xai",), 0.5) is True


@pytest.mark.asyncio
async def test_inga_motorer_ar_inte_natet_nere():
    assert await bana0.natet_nere(lambda e: False, (None, ""), 0.5) is False


def test_main_kopplar_in_proben():
    # initialize() needs HA and the engines; its source is what wires the probe.
    import inspect
    import app.main as m
    assert "websocket_handler.engine_probe = probe_engine" in inspect.getsource(m.Application.initialize)


def test_proben_hinner_fore_tystnadskvittot():
    from app import websocket_handler
    from app.providers.tool_registration import _silence_ack_ms
    assert websocket_handler.OFFLINE_PROBE_S * 1000 < _silence_ack_ms()


def test_traffens_kvitto_tystas_inte_av_att_turen_avslutas():
    """Live 2026-10-04 17:14: force_idle after the hit marked the turn over,
    claim_silence_ack said no, and "Klart." never played."""
    import time as _t
    from app.phase_emitter import TurnLiveness
    liv = TurnLiveness()
    asked = _t.monotonic()
    liv.turn_over()  # what force_idle("bana0") does right after the hit
    assert bana0.ingen_bekraftelse_an(liv, asked) is True
    liv.bot_started()
    assert bana0.ingen_bekraftelse_an(liv, asked) is False


def test_ny_fras_efter_traffen_far_inget_klart():
    import time as _t
    from app.phase_emitter import TurnLiveness
    liv = TurnLiveness()
    asked = _t.monotonic()
    _t.sleep(0.01)
    liv.user_started()
    assert bana0.ingen_bekraftelse_an(liv, asked) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status,lever", [(200, True), (404, True), (401, True), (502, False), (503, False)])
async def test_comms_lever_raknar_varje_svar_under_500(monkeypatch, status, lever):
    monkeypatch.setenv("HA_API_URL", "http://comms.test/kanal/rost/kontoret/api")
    monkeypatch.setenv("COMMS_NYCKEL", "k")

    async def svar(*args, **kwargs):
        return SimpleNamespace(status_code=status)

    monkeypatch.setattr(bana0.httpx.AsyncClient, "get", svar)
    assert await bana0.comms_lever(0.5) is lever


@pytest.mark.asyncio
async def test_comms_fraga_som_kastar_ger_inget_lofte():
    sagt, said = await _vakta(True, comms=RuntimeError("boom"))
    assert sagt is True and said[0] in bana0.OFFLINE_UTAN_LOFTE


@pytest.mark.asyncio
async def test_vakta_natet_frågar_comms_själv_utan_injektion(monkeypatch):
    """The guard owns the Comms check: nothing at the call site can drop it."""
    anropad = []

    async def comms_nere(timeout):
        anropad.append(timeout)
        return False

    monkeypatch.setattr(bana0, "comms_lever", comms_nere)
    said = []

    async def natet_nere():
        return True

    async def say(text):
        said.append(text)

    sagt = await bana0.vakta_natet(natet_nere=natet_nere, claim=lambda: True, say=say)
    assert anropad == [bana0.COMMS_PROBE_S]
    assert sagt is True and said[0] in bana0.OFFLINE_UTAN_LOFTE


@pytest.mark.asyncio
async def test_comms_lever_ar_falskt_utan_konfiguration_aven_om_nagon_svarar(monkeypatch):
    monkeypatch.delenv("HA_API_URL", raising=False)

    async def svar(*args, **kwargs):
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(bana0.httpx.AsyncClient, "get", svar)
    assert await bana0.comms_lever(0.5) is False


# --- G:s slutpass: motorprovet och Comms-kontrollen går sida vid sida, och platsen tas sist ---

@pytest.mark.asyncio
async def test_motorprov_och_comms_kontroll_vantar_inte_efter_varandra():
    async def natet_nere():
        await asyncio.sleep(0.3)
        return True

    async def comms():
        await asyncio.sleep(0.3)
        return True

    start = time.monotonic()
    sagt = await bana0.vakta_natet(natet_nere=natet_nere, claim=lambda: True, say=lambda t: asyncio.sleep(0), kolla_comms=comms)
    assert sagt is True
    assert time.monotonic() - start < 0.5  # 0.6 s om de väntade efter varandra


@pytest.mark.asyncio
async def test_tystnadsplatsen_tas_efter_comms_kontrollen():
    handelser = []

    async def natet_nere():
        return True

    async def comms():
        await asyncio.sleep(0.1)
        handelser.append("comms klar")
        return True

    def claim():
        handelser.append("claim")
        return True

    async def say(t):
        handelser.append("say")

    await bana0.vakta_natet(natet_nere=natet_nere, claim=claim, say=say, kolla_comms=comms)
    assert handelser == ["comms klar", "claim", "say"]
