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

    async def gemini_tts(text, key, voice, model="", cache=True, ram=""):
        anrop.append((cache, ram))
        return b"pcm"

    monkeypatch.setattr(main, "gemini_tts", gemini_tts)

    class Agent:
        gemini_api_key, xai_api_key, gemini_voice, _ack_clips = "g", "", None, {}

    a = Agent()
    await main.Application._ack_clip(a, "gemini", "Ett privat svar.", fallback=False, cache=False)
    await main.Application._ack_clip(a, "gemini", "Ett privat svar.", fallback=False, cache=False)
    assert anrop == [(False, ""), (False, "")] and a._ack_clips == {}  # rendered twice, remembered never
    await main.Application._ack_clip(a, "gemini", "Ett annat privat svar.", fallback=False, cache=False, levande=True)
    assert anrop[-1][0] is False and "levande" in anrop[-1][1]  # the livelier frame for a streamed answer


def test_adressen_harleds_ur_rummets_comms_adress(monkeypatch):
    monkeypatch.delenv("CORE_STROM_URL", raising=False)
    monkeypatch.setenv("HA_API_URL", "http://10.10.0.118:3500/kanal/rost/kontoret/api")
    assert core_strom.strom_url() == "http://10.10.0.118:3500/kanal/rost/kontoret/fraga"
    monkeypatch.setenv("CORE_STROM_URL", "http://x/y")
    assert core_strom.strom_url() == "http://x/y"
    monkeypatch.delenv("CORE_STROM_URL")
    monkeypatch.delenv("HA_API_URL")
    assert core_strom.strom_url() == ""


def test_roststyrka_drag_hittar_tonhojden_i_en_ren_ton():
    import importlib.util
    import pathlib

    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "rostjamforelse", pathlib.Path(__file__).parent.parent / "tools" / "rostjamforelse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t = np.arange(mod.RATE) / mod.RATE
    pcm = (0.3 * np.sin(2 * np.pi * 150 * t) * 32767).astype(np.int16).tobytes()
    d = mod.drag(pcm)
    assert d["sekunder"] == 1.0 and abs(d["f0_median"] - 150) < 5


@pytest.mark.asyncio
async def test_token_fore_deferred_talas_inte_en_gang_till_i_done(  ):
    """B on #35: Core's `done` carries ALL the text, also what was spoken before `deferred`."""
    talat = []

    async def say(t):
        talat.append(t)

    async def chunks():
        yield ram("token", {"text": "Jag ber Sixten titta på det här åt dig nu. "})
        yield ram("deferred", {"agent": "sixten"})
        yield ram("done", {"response": {"type": "deferred", "text": "Jag ber Sixten titta på det här åt dig nu."}})

    ut = await tala_strom(chunks(), say)
    assert talat == ["Jag ber Sixten titta på det här åt dig nu."] and ut["slut"] == "deferred"


@pytest.mark.asyncio
async def test_ett_svar_har_tak_pa_meningar():
    talat = []

    async def say(t):
        talat.append(t)

    async def chunks():
        for i in range(12):
            yield ram("token", {"text": f"Det här är mening nummer {i} i ett mycket långt svar. "})
        yield ram("done", {"response": {"type": "speech", "text": "x"}})

    ut = await tala_strom(chunks(), say, max_meningar=3)
    assert len(talat) == 4 and talat[-1] == core_strom.TAK_REPLIK and ut["tak"] is True  # 3 sentences, then a word that there is more


@pytest.mark.asyncio
async def test_avbryt_stoppar_ett_pagaende_svar():
    import asyncio

    from app.core_strom import register_fraga_core

    class Llm:
        def register_function(self, namn, fn):
            self.fn = fn

    llm = Llm()
    avbryt = register_fraga_core(llm, lambda t: asyncio.sleep(0), lambda: {})
    assert callable(avbryt)
    avbryt()  # nothing running: harmless


@pytest.mark.asyncio
async def test_avbryt_stoppar_ett_pagaende_svar_mitt_i(monkeypatch):
    """B on #35: a running answer is cancelled, not just an empty avbryt()."""
    import asyncio

    from app.core_strom import register_fraga_core

    talat, fastnat = [], asyncio.Event()

    class Svar:
        status_code = 200

        async def aiter_bytes(self):
            yield ram("token", {"text": "Första meningen är hel nu och klar. "})
            yield ram("token", {"text": "Sedan börjar nästa"})  # the next word shows the first sentence is whole
            fastnat.set()
            await asyncio.Event().wait()  # the rest is never written
            yield ram("done", {"response": {"type": "speech", "text": "x"}})

    class Ström:
        async def __aenter__(self):
            return Svar()

        async def __aexit__(self, *a):
            return False

    class Klient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, *a, **k):
            return Ström()

    monkeypatch.setattr(core_strom.httpx, "AsyncClient", Klient)
    monkeypatch.setenv("CORE_STROM_URL", "http://x/fraga")

    class Llm:
        def register_function(self, namn, fn):
            self.fn = fn

    class Params:
        arguments = {"question": "vad vet du om mig?"}

        async def result_callback(self, r):
            self.svar = r

    async def say(t):
        talat.append(t)

    llm = Llm()
    avbryt = register_fraga_core(llm, say, lambda: {})
    p = Params()
    await llm.fn(p)
    assert "checking" in p.svar["status"]
    await asyncio.wait_for(fastnat.wait(), 5)
    await asyncio.sleep(0.05)
    assert talat == ["Första meningen är hel nu och klar."]
    taken_innan = [t for t in asyncio.all_tasks() if "kor" in repr(t)]
    assert taken_innan
    avbryt()
    await asyncio.sleep(0.05)
    assert all(t.done() for t in taken_innan) and talat == ["Första meningen är hel nu och klar."]


def test_daempa_sanker_niva_med_2_3_db_och_klipper_inte():
    import numpy as np

    from app.early_ack import STROM_NIVA_DB, daempa

    x = (np.sin(np.arange(24000) / 10) * 12000).astype(np.int16)
    ut = np.frombuffer(daempa(x.tobytes()), dtype=np.int16).astype(np.float32)
    db = 20 * np.log10(np.sqrt(np.mean(ut ** 2)) / np.sqrt(np.mean(x.astype(np.float32) ** 2)))
    assert abs(db - STROM_NIVA_DB) < 0.05 and len(ut) == len(x)
    hog = daempa((np.full(100, 32767, dtype=np.int16)).tobytes(), db=6)  # up 6 dB: clipped, not wrapped
    assert set(np.frombuffer(hog, dtype=np.int16).tolist()) == {32767}
