"""US-403: when Core is slow, the speaker says once that it is still checking, instead of going silent."""
import asyncio
import json

import pytest

from app import core_strom
from app.core_strom import register_fraga_core


def ram(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def uppsattning(monkeypatch, bytes_fn):
    class Svar:
        status_code = 200
        aiter_bytes = bytes_fn

    class Strom:
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
            return Strom()

    monkeypatch.setattr(core_strom.httpx, "AsyncClient", Klient)
    monkeypatch.setattr(core_strom, "strom_url", lambda: "http://x/fraga")

    class Llm:
        def register_function(self, namn, fn):
            self.fn = fn

    class Params:
        arguments = {"question": "vad kostar elen"}

        async def result_callback(self, r):
            self.svar = r

    talat = []

    async def say(t):
        talat.append(t)

    return Llm(), Params(), talat, say


@pytest.mark.asyncio
async def test_ingen_mening_inom_gransen_ger_ett_langsam_besked_en_gang(monkeypatch):
    slut = asyncio.Event()

    async def bytes_fn(self):
        yield ram("start", {"v": 1})
        await slut.wait()  # Core writes nothing
        yield ram("done", {"response": {"type": "speech", "text": "Elen kostar 40 öre per kilowattimme."}})

    llm, p, talat, say = uppsattning(monkeypatch, bytes_fn)
    avbryt = register_fraga_core(llm, say, lambda: {}, langsam_efter_s=0.05)
    await llm.fn(p)
    await asyncio.sleep(0.3)
    assert talat == [core_strom.LANGSAM_REPLIK]  # once, not repeated
    slut.set()
    await asyncio.sleep(0.1)
    assert talat == [core_strom.LANGSAM_REPLIK, "Elen kostar 40 öre per kilowattimme."]
    avbryt()


@pytest.mark.asyncio
async def test_snabbt_svar_far_inget_langsam_besked(monkeypatch):
    async def bytes_fn(self):
        yield ram("token", {"text": "Elen kostar 40 öre per kilowattimme just nu. "})
        yield ram("done", {"response": {"type": "speech", "text": "Elen kostar 40 öre per kilowattimme just nu."}})

    llm, p, talat, say = uppsattning(monkeypatch, bytes_fn)
    register_fraga_core(llm, say, lambda: {}, langsam_efter_s=0.2)
    await llm.fn(p)
    await asyncio.sleep(0.4)
    assert talat == ["Elen kostar 40 öre per kilowattimme just nu."]


@pytest.mark.asyncio
async def test_avbrutet_svar_far_inget_sent_langsam_besked(monkeypatch):
    async def bytes_fn(self):
        await asyncio.Event().wait()
        yield b""

    llm, p, talat, say = uppsattning(monkeypatch, bytes_fn)
    avbryt = register_fraga_core(llm, say, lambda: {}, langsam_efter_s=0.2)
    await llm.fn(p)
    await asyncio.sleep(0.05)
    avbryt()  # the owner starts to speak
    await asyncio.sleep(0.4)
    assert talat == []
