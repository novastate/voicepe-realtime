"""Speak a Core answer while it is written (raawr US-047 AC-4)."""
import asyncio

import pytest

from app import core_strom
from app.core_strom import Meningsdelare, las_ramar, tala_strom


def ram(event, data):
    import json
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def test_tokens_blir_hela_meningar_och_forkortningar_delar_inte():
    d = Meningsdelare()
    ut = []
    for tok in ["Elpriset är lågt ", "just nu, t.ex. ", "under 30 öre.", " Det stiger ", "efter klockan ", "sex!", " Ja."]:
        ut += d.mata(tok)
    assert ut == ["Elpriset är lågt just nu, t.ex. under 30 öre.", "Det stiger efter klockan sex!"]
    assert d.flush() == ["Ja."]  # too short to go alone, but the stream is done


def test_en_kort_mening_vantar_pa_nasta():
    d = Meningsdelare()
    assert d.mata("Ja. Det går bra att göra så. Sedan fortsätter vi.") == ["Ja. Det går bra att göra så."]
    assert d.flush() == ["Sedan fortsätter vi."]


@pytest.mark.asyncio
async def test_forsta_meningen_talas_innan_stromen_ar_slut():
    """The point of AC-4: the first sentence does not wait for `done`."""
    slut = asyncio.Event()
    talat = []

    async def chunks():
        yield ram("start", {"v": 1})
        yield ram("token", {"text": "Det blir regn i morgon förmiddag. "})
        yield ram("token", {"text": "Sedan klarnar"})
        await slut.wait()  # the rest of the answer is not written yet
        yield ram("token", {"text": " det upp."})
        yield ram("done", {"response": {"type": "speech", "text": "x"}})

    async def say(t):
        talat.append(t)
        slut.set()  # the test lets the stream go on once the first sentence was spoken

    ut = await asyncio.wait_for(tala_strom(chunks(), say), 5)
    assert talat == ["Det blir regn i morgon förmiddag.", "Sedan klarnar det upp."]
    assert ut["slut"] == "done" and ut["meningar"] == 2 and ut["forsta_mening_s"] is not None


@pytest.mark.asyncio
async def test_deferred_talar_cores_egen_rad_och_fel_talar_en_fast_rad():
    talat = []

    async def say(t):
        talat.append(t)

    async def deferred():
        yield ram("start", {})
        yield ram("deferred", {"agent": "sixten"})
        yield ram("done", {"response": {"type": "deferred", "text": "Sixten jobbar på det."}})

    async def fel():
        yield ram("token", {"text": "Det här börjar bra men "})
        yield ram("error", {"code": "core_nere", "message": "hemligt felmeddelande"})

    assert (await tala_strom(deferred(), say))["slut"] == "deferred"
    assert (await tala_strom(fel(), say))["slut"] == "error"
    assert talat == ["Sixten jobbar på det.", core_strom.FEL_REPLIK]  # the error text itself is never spoken


@pytest.mark.asyncio
async def test_las_ramar_klarar_bitar_mitt_i_en_ram_och_hoppar_over_skrap():
    data = ram("token", {"text": "hej"}) + b"event: token\ndata: inte json\n\n" + ram("done", {"a": 1})
    delad = [data[:7], data[7:30], data[30:]]

    async def chunks():
        for c in delad:
            yield c

    assert [e async for e in las_ramar(chunks())] == [("token", {"text": "hej"}), ("done", {"a": 1})]


@pytest.mark.asyncio
async def test_en_ström_utan_done_talar_det_som_finns():
    talat = []

    async def say(t):
        talat.append(t)

    async def chunks():
        yield ram("token", {"text": "Svaret kom aldrig hela vägen"})

    ut = await tala_strom(chunks(), say)
    assert talat == ["Svaret kom aldrig hela vägen"] and ut["slut"] == "avbruten"


def test_flaggan_ar_av_som_standard(monkeypatch):
    monkeypatch.delenv("CORE_STREAM_TALA", raising=False)
    assert core_strom.core_stream_tala_paa() is False
    monkeypatch.setenv("CORE_STREAM_TALA", "1")
    assert core_strom.core_stream_tala_paa() is True


@pytest.mark.asyncio
async def test_ack_clip_utan_cache_laser_och_skriver_inget(monkeypatch):
    import app.main as main

    anrop = []

    async def gemini_tts(text, key, voice, model="", cache=True):
        anrop.append(cache)
        return b"pcm"

    monkeypatch.setattr(main, "gemini_tts", gemini_tts)

    class Agent:
        gemini_api_key, xai_api_key, gemini_voice, _ack_clips = "g", "", None, {}

    a = Agent()
    await main.Application._ack_clip(a, "gemini", "Ett privat svar.", fallback=False, cache=False)
    await main.Application._ack_clip(a, "gemini", "Ett privat svar.", fallback=False, cache=False)
    assert anrop == [False, False] and a._ack_clips == {}  # rendered twice, remembered never


def test_adressen_harleds_ur_rummets_comms_adress(monkeypatch):
    monkeypatch.delenv("CORE_STROM_URL", raising=False)
    monkeypatch.setenv("HA_API_URL", "http://10.10.0.118:3500/kanal/rost/kontoret/api")
    assert core_strom.strom_url() == "http://10.10.0.118:3500/kanal/rost/kontoret/fraga"
    monkeypatch.setenv("CORE_STROM_URL", "http://x/y")
    assert core_strom.strom_url() == "http://x/y"
    monkeypatch.delenv("CORE_STROM_URL")
    monkeypatch.delenv("HA_API_URL")
    assert core_strom.strom_url() == ""
