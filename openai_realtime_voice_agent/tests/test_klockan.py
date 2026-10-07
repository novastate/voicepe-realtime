"""The clock without the cloud (raawr US-032 AC-7): local STT text -> cached clips,
neither comms nor the model asked."""
from datetime import datetime, timezone

import httpx
import pytest

from app import bana0, klockan
from tests.test_bana0 import FakeService, _transcript, _wyoming, comms  # noqa: F401

FEM_SATT = [
    "Vad är klockan?",
    "Hur mycket är klockan?",
    "Var är klockan?",  # how Whisper heard it live, 2026-10-02
    "Vet du vad klockan är?",
    "Hej Björn, hur dags är det nu?",
]


@pytest.mark.parametrize("text", FEM_SATT + ["vad e klockan", "Klockan?"])
def test_klockfragor_kanns_igen(text):
    assert klockan.ar_klockfraga(text)


@pytest.mark.parametrize("text", [
    None, "", "ställ klockan på sju", "väck mig klockan sex", "tänd lampan i kontoret",
    "vad är klockan i New York", "sätt en timer på tio minuter",
])
def test_annat_ar_ingen_klockfraga(text):
    assert not klockan.ar_klockfraga(text)


@pytest.mark.parametrize("nu, sagt", [
    (datetime(2026, 10, 7, 14, 22, tzinfo=klockan.TZ), ["Klockan är fjorton", "och tjugotvå minuter"]),
    (datetime(2026, 10, 7, 9, 5, tzinfo=klockan.TZ), ["Klockan är nio", "och fem minuter"]),
    (datetime(2026, 10, 7, 0, 0, tzinfo=klockan.TZ), ["Klockan är noll"]),
    (datetime(2026, 10, 7, 7, 1, tzinfo=klockan.TZ), ["Klockan är sju", "och en minut"]),
    (datetime(2026, 10, 7, 23, 59, tzinfo=klockan.TZ), ["Klockan är tjugotre", "och femtionio minuter"]),
    # UTC in, Stockholm out: summer +2, winter +1.
    (datetime(2026, 10, 7, 12, 30, tzinfo=timezone.utc), ["Klockan är fjorton", "och trettio minuter"]),
    (datetime(2026, 12, 7, 12, 30, tzinfo=timezone.utc), ["Klockan är tretton", "och trettio minuter"]),
])
def test_ratt_timme_och_minut_i_stockholm(nu, sagt):
    assert klockan.delar(nu) == sagt


def test_varje_minut_har_ett_forrenderat_klipp():
    alla = set(klockan.alla_delar())
    assert len(alla) == 24 + 59
    for h in range(24):
        for m in range(60):
            assert set(klockan.delar(datetime(2026, 1, 1, h, m, tzinfo=klockan.TZ))) <= alla


FAST = datetime(2026, 10, 7, 14, 22, tzinfo=klockan.TZ)


async def _klocktur(port, service, said, klocka):
    async def stt(pcm, timeout):
        return await bana0.transkribera(pcm, "127.0.0.1", port, timeout)
    return await bana0.tur(
        b"\x00" * 3200, stt=stt, timeout_stt=1.0, timeout_comms=4.0,
        skicka_svar_till_modellen=lambda t: bana0.be_om_bekraftelse(service, t),
        skapa_svar=lambda: bana0.be_om_svar(service),
        efter_miss=lambda: said.append("miss"),
        klockan=klocka,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fraga", FEM_SATT)
async def test_klockan_svarar_utan_comms_och_modell(comms, fraga):
    server, port, _ = await _wyoming(_transcript(fraga))
    service, said = FakeService(), []

    async def klocka():
        said.extend(klockan.delar(FAST))

    async with server:
        assert await _klocktur(port, service, said, klocka) == "klockan"
    assert said == ["Klockan är fjorton", "och tjugotvå minuter"]
    assert comms.seen == [] and service.events == []


@pytest.mark.asyncio
async def test_annan_fraga_gar_vidare_som_i_dag(comms):
    comms.svar = httpx.Response(204)
    server, port, _ = await _wyoming(_transcript("hur varmt är det ute"))
    service, said = FakeService(), []

    async def klocka():
        said.append("klocka")

    async with server:
        assert await _klocktur(port, service, said, klocka) == "modell"
    assert said == ["miss"]
    assert len(comms.seen) == 1 and service.typer() == ["response.create"]


@pytest.mark.asyncio
async def test_klocka_som_fallerar_ger_vanlig_tur(comms):
    comms.svar = httpx.Response(204)
    server, port, _ = await _wyoming(_transcript("vad är klockan"))
    service, said = FakeService(), []

    async def klocka():
        raise RuntimeError("no clip")

    async with server:
        assert await _klocktur(port, service, said, klocka) == "modell"
    assert said == ["miss"] and service.typer() == ["response.create"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fel, vantat_forvantat", [
    # tre fails twice and is skipped; the rest go on, each render paced.
    ({"Klockan är tre": 2}, [8.0, 8.0, 8.0, 60, 8.0]),
    # three clips in a row fail twice: the engine is down, stop.
    ({"Klockan är noll": 2, "Klockan är ett": 2, "Klockan är två": 2}, [60, 8.0, 60, 8.0, 60]),
])
async def test_uppvarmningen_haller_googles_takt(monkeypatch, fel, vantat_forvantat):
    """10 TTS requests/min (429 on 2026-10-07): a render waits, the disk does not,
    and no clip is cached in another voice."""
    import app.main as main

    vantat = []

    async def sov(s):
        vantat.append(s)

    tider = iter(range(0, 100000, 1))
    monkeypatch.setattr(main.asyncio, "sleep", sov)
    monkeypatch.setattr(main.time, "monotonic", lambda: next(tider))  # every render "takes" 1 s

    class Agent:
        async def _ack_clip(self, provider, text, fallback=True):
            assert fallback is False
            if fel.get(text):
                fel[text] -= 1
                raise RuntimeError("no audio")
            return b"pcm"

    await main.Application._warm_klockan(Agent(), "gemini", takt_s=8.0)
    assert vantat[:len(vantat_forvantat)] == vantat_forvantat
    if len(fel) == 3:
        assert vantat == vantat_forvantat  # stopped after the third
    else:
        assert len(vantat) == 24 + 59 + 1  # every clip paced, plus the one minute wait
