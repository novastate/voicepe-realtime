"""raawr US-542: the voice agent has no calendar-writing road past comms, and reports only what the STT heard."""
import asyncio

import pytest

from app import hord, tool_selection
from app.transcript_logger import TranscriptLogger
from tests.test_mcp_dorr import _Klient, _app, _schema


@pytest.mark.asyncio
@pytest.mark.parametrize("skript", ["script__kalender_andra", "script__kalender_ta_bort", "kalender_ta_bort", "kalender_andra"])
async def test_ha_kalenderskript_erbjuds_och_registreras_aldrig(skript):
    ha, dorr = _Klient(_schema(skript, "HassTurnOn")), _Klient(_schema("lage_drift"))
    app = _app(ha, dorr)
    app.mcp_tool_allowlist = ()  # even with no allow-list and TOOL_DENY="-" (below) nothing slips through
    schema = await app._fetch_ha_tools_schema()
    assert skript not in [t["name"] for t in app._ha_tool_definitions(schema)]
    await app._register_ha_handlers(object(), schema, "kontoret")
    assert skript not in [n for grupp in ha.registrerat for n in grupp]


@pytest.mark.asyncio
async def test_kalender_sok_och_dorren_ar_kvar(monkeypatch):
    monkeypatch.setenv("TOOL_DENY", "-")
    app = _app(_Klient(_schema("script__kalender_sok")), _Klient(_schema("lage_drift")))
    namn = [t["name"] for t in app._ha_tool_definitions(await app._fetch_ha_tools_schema())]
    assert "lage_drift" in namn and "script__kalender_sok" in namn  # reading and the door are untouched


def test_kalender_skript_traffar_bara_skrivande_skript_utanfor_dorren():
    assert tool_selection.kalender_skript("script__kalender_andra")
    assert not tool_selection.kalender_skript("kalender_andra", frozenset({"kalender_andra"}))
    assert not tool_selection.kalender_skript("script__kalender_sok")


@pytest.mark.asyncio
async def test_hord_rapporterar_stt_text_i_ordning_med_rummets_nyckel(monkeypatch):
    monkeypatch.setenv("CORE_HORT_URL", "http://comms.test/kanal/rost/kontoret/hort")
    monkeypatch.setenv("COMMS_NYCKEL", "rumsnyckel")
    skickat = []

    class _K:
        async def post(self, url, json, headers, timeout=None):
            await asyncio.sleep(0.02 if json["text"] == "flytta tandläkaren" else 0)
            skickat.append((url, json["text"], headers.get("X-Raawr-Nyckel")))

            class _R:
                status_code = 200
            return _R()

    monkeypatch.setattr(hord.httpx, "AsyncClient", lambda **kw: _Ctx(_K()))
    hord.rapportera("flytta tandläkaren")
    hord.rapportera("ja")
    await asyncio.gather(*list(hord._uppgifter))
    assert [t for _, t, _ in skickat] == ["flytta tandläkaren", "ja"]
    assert {n for _, _, n in skickat} == {"rumsnyckel"}


class _Ctx:
    def __init__(self, k): self.k = k
    async def __aenter__(self): return self.k
    async def __aexit__(self, *a): return False


def test_transkriptloggern_rapporterar_bara_stt_ramar():
    from pipecat.frames.frames import TranscriptionFrame, TextFrame
    import inspect
    k = inspect.getsource(TranscriptLogger.process_frame)
    assert "TranscriptionFrame" in k and "_on_user" in k and "TextFrame(" not in k
