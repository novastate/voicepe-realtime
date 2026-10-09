"""Noise heard as speech must not start music (kitchen 2026-10-09 21:17: "Play Game of Tips" from music noise)."""
from types import SimpleNamespace

import httpx
import pytest

from app import bana0, play_media_tool, sprakkoll


@pytest.mark.parametrize("text", ["Ett tack till mina supporters via Patreon!", "Tack för att ni tittade.",
                                  "Undertexter från Amara.org", "Thanks for watching!", "", "a", "...", None])
def test_kanda_pahitt_och_tomt_ar_brus(text):
    assert bana0.brus(text)


@pytest.mark.parametrize("text", ["Sätt på undertexter", "Jag vill prenumerera på P3", "Tänd köket", "Vad är klockan", "ja", "Spela P3", "Tack"])
def test_riktiga_kommandon_ar_inte_brus(text):
    assert not bana0.brus(text)


@pytest.mark.asyncio
async def test_pahitt_ger_ingen_traff_och_modellen_far_ljudet():
    anrop = []

    async def stt(pcm, t):
        return "Ett tack till mina supporters via Patreon!"

    async def comms(*a, **k):
        anrop.append("comms")

    async def modell():
        anrop.append("modell")

    orig = bana0.prova
    bana0.prova = comms
    try:
        res = await bana0.tur(b"x" * 100, stt=stt, timeout_stt=1, timeout_comms=1,
                              skicka_svar_till_modellen=comms, skapa_svar=modell)
    finally:
        bana0.prova = orig
    assert res == "modell" and anrop == ["modell"]  # comms never asked about the invention


def test_hel_tur_mest_pa_engelska_hors_men_svenska_med_engelska_namn_inte():
    assert sprakkoll.mest_annat("Play Game of Tips") == "engelska"
    assert sprakkoll.mest_annat("Spela Game of Thrones musiken") is None
    assert sprakkoll.mest_annat("Kan du sätta på P3") is None
    assert sprakkoll.mest_annat("") is None


def _korr(tur_text, hort, query="Game of Tips"):
    spelade, sagt = [], []

    def handler(request):
        p = request.url.path
        if p.endswith("/states"):
            return httpx.Response(200, json=[{"entity_id": "media_player.kok", "state": "idle",
                                              "attributes": {"friendly_name": "Kök", "mass_player_type": "player"}}])
        if p.endswith("/config_entries/entry"):
            return httpx.Response(200, json=[{"entry_id": "ma1"}])
        if p.endswith("/music_assistant/search"):
            return httpx.Response(200, json={"service_response": {"podcasts": [
                {"name": "Game of Tips", "uri": "podcast://1", "favorite": False}]}})
        if p.endswith("/music_assistant/play_media"):
            spelade.append(1)
        return httpx.Response(200, json=[])

    async def cb(text):
        sagt.append(text)

    nr = {"n": 0}
    h = play_media_tool.create_play_media_tool_handler("koket", tur_text=lambda: tur_text, hort=lambda: hort,
                                                       tur_nr=lambda: nr["n"])
    h.nr = nr
    return h, SimpleNamespace(arguments={"query": query}, result_callback=cb), spelade, sagt, handler


@pytest.fixture(autouse=True)
def _ha(monkeypatch):
    monkeypatch.setenv("HA_API_URL", "http://comms/")
    monkeypatch.setenv("COMMS_NYCKEL", "k")
    monkeypatch.setattr(play_media_tool, "_config_entry_id", None, raising=False)


async def _kor(monkeypatch, tur_text, hort, query="Game of Tips", gor_om=False):
    h, params, spelade, sagt, mock = _korr(tur_text, hort, query)
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(mock)}))
    await h(params)
    if gor_om == "samma tur":
        await h(params)
    elif gor_om:
        h.nr["n"] += 1  # the user answered: a new turn
        await h(params)
    return spelade, sagt


@pytest.mark.asyncio
async def test_engelsk_avskrift_spelar_inget_utan_fraga_forst(monkeypatch):
    spelade, sagt = await _kor(monkeypatch, "Play Game of Tips", "Du kan välja med tips?")
    assert spelade == [] and "Ask the user once" in sagt[0]


@pytest.mark.asyncio
async def test_lokala_stt_hor_inte_namnet_fragar_forst(monkeypatch):
    spelade, sagt = await _kor(monkeypatch, "Spela Game of Tips", "Hej hej hej", query="Game of Tips")
    assert spelade == [] and "Ask the user once" in sagt[0]


@pytest.mark.asyncio
async def test_samma_onskan_igen_spelas(monkeypatch):
    spelade, sagt = await _kor(monkeypatch, "Play Game of Tips", "Du kan välja med tips?", gor_om=True)
    assert spelade == [1] and "Ask the user once" in sagt[0] and sagt[1].startswith("Playing")


@pytest.mark.asyncio
async def test_klar_svensk_begaran_spelas_direkt(monkeypatch):
    spelade, sagt = await _kor(monkeypatch, "Spela Game of Tips", "Spela game of tips")
    assert spelade == [1] and sagt[0].startswith("Playing")


@pytest.mark.asyncio
async def test_modellen_kan_inte_ringa_om_i_samma_tur(monkeypatch):
    spelade, sagt = await _kor(monkeypatch, "Play Game of Tips", "Du kan välja med tips?", gor_om="samma tur")
    assert spelade == [] and all("Ask the user once" in t for t in sagt)
