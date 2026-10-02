"""Without internet (raawr US-018): bana 0 still confirms, a question gets an honest no."""
import pytest

from app import bana0

PCM = b"\x00" * 3200


async def _stt(pcm, timeout):
    return "tänd kontoret"


def _tur(prova_svar, say, skapa_svar=None, efter_miss=None, monkeypatch=None):
    async def prova(text, timeout):
        return prova_svar

    monkeypatch.setattr(bana0, "prova", prova)

    async def skicka(text):
        pass

    async def skapa():
        pass

    return bana0.tur(
        PCM, stt=_stt, timeout_stt=1.0, timeout_comms=1.0, say=say,
        skicka_svar_till_modellen=skicka, skapa_svar=skapa_svar or skapa,
        efter_miss=efter_miss,
    )


@pytest.mark.asyncio
async def test_traff_utan_molnrost_sager_klart(monkeypatch):
    said = []

    async def say(text):
        if text != bana0.OK_FALLBACK:
            raise RuntimeError("TTS nere: inget internet")
        said.append(text)

    assert await _tur("Slog på lampan", say, monkeypatch=monkeypatch) == "bana0"
    assert said == [bana0.OK_FALLBACK]


@pytest.mark.asyncio
async def test_miss_dar_modellen_inte_nas_kraschar_inte_och_vaktar(monkeypatch):
    calls = []

    async def say(text):
        calls.append(("say", text))

    async def skapa():
        raise RuntimeError("no socket")

    bana = await _tur(None, say, skapa_svar=skapa,
                      efter_miss=lambda: calls.append("vakt"), monkeypatch=monkeypatch)
    assert bana == "modell"
    assert calls == ["vakt"]


async def _vakta(nere, claimed=True):
    said = []

    async def natet_nere():
        if isinstance(nere, Exception):
            raise nere
        return nere

    async def say(text):
        said.append(text)

    sagt = await bana0.vakta_natet(natet_nere=natet_nere, claim=lambda: claimed, say=say)
    return sagt, said


@pytest.mark.asyncio
async def test_natet_nere_sager_det():
    assert await _vakta(True) == (True, [bana0.OFFLINE_LINE])


@pytest.mark.asyncio
async def test_probe_som_kraschar_raknas_som_nere():
    assert await _vakta(OSError("dns")) == (True, [bana0.OFFLINE_LINE])


@pytest.mark.asyncio
async def test_natet_uppe_ingen_replik():
    assert await _vakta(False) == (False, [])


@pytest.mark.asyncio
async def test_modellen_eller_ogonblick_tog_platsen_ingen_replik():
    assert await _vakta(True, claimed=False) == (False, [])


def test_replikerna_ar_inga_fragor():
    assert all("?" not in r for r in bana0.LOKALA_REPLIKER)
