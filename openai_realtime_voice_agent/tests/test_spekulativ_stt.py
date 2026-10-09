"""Bana 0's speech-to-text starts early, on "preend" (raawr US-032): the model still
never hears a hit, and a pause that ends in more speech throws the early text away."""
import asyncio

import pytest

from app import bana0
from app.providers.local_turns import LocalTurns
from tests.test_bana0 import _gemini_koppling, _speak, _transcript, _wyoming, ha_svarar  # noqa: F401


class FakeVad:
    """sherpa's detector: is_speech_detected follows a script, one entry per accepted chunk."""

    def __init__(self, script):
        self.script, self.i = list(script), 0

    def accept_waveform(self, samples):
        self.i += 1

    def empty(self):
        return True

    def is_speech_detected(self):
        return self.script[min(self.i, len(self.script)) - 1]

    def reset(self):
        pass


def _feed(turns, n):
    import numpy as np
    chunk = (np.zeros(512, dtype=np.int16)).tobytes()
    return [turns.feed(chunk) for _ in range(n)]


def test_preend_kommer_fore_end_och_bara_medan_turen_pagar():
    #           speech ... quiet 500 ms (pre) ... quiet 800 ms (main)
    main = FakeVad([False, True, True, True, True, False])
    pre = FakeVad([False, True, True, True, False, False])
    assert _feed(LocalTurns(main, 16000, pre), 6) == [None, "start", None, None, "preend", "end"]


def test_preend_igen_efter_en_paus_som_blev_tal():
    main = FakeVad([False, True, True, True, True, True, True, False])
    pre = FakeVad([False, True, False, True, True, False, False, False])
    handelser = _feed(LocalTurns(main, 16000, pre), 8)
    assert handelser.count("preend") == 2 and handelser[-1] == "end"


def test_utan_pre_ar_allt_som_forut():
    main = FakeVad([False, True, True, False])
    assert _feed(LocalTurns(main, 16000), 4) == [None, "start", None, "end"]


def _spela_in(monkeypatch, connection):
    """(STT calls' audio lengths, the turn's final audio length), to tell early from final."""
    anrop, slutlangd = [], []
    riktig = bana0.transkribera

    async def transkribera(pcm, host, port, timeout):
        anrop.append(len(pcm))
        return await riktig(pcm, host, port, timeout)

    monkeypatch.setattr(bana0, "transkribera", transkribera)
    ta = connection.serializer.take_turn_audio

    def ta_spar():
        pcm = ta()
        slutlangd.append(len(pcm))
        return pcm

    connection.serializer.take_turn_audio = ta_spar
    return anrop, slutlangd


def _stt_korningar(seen):
    return len([1 for t, _, _ in seen if t == "audio-stop"])


@pytest.mark.asyncio
async def test_en_paus_som_inte_blev_tal_ger_en_enda_stt_och_modellen_hor_inget_av_en_traff(ha_svarar, monkeypatch):
    from test_gemini_provider import _kinds

    ha_svarar.append("Släckte i kontoret")
    server, port, seen = await _wyoming(_transcript("släck kontoret"))
    connection, service, google, said, idle = _gemini_koppling(
        ("127.0.0.1", port), [None, "start", "preend", None, "end"])
    anrop, slutlangd = _spela_in(monkeypatch, connection)
    async with server:
        await _speak(connection, service, 5)
        await service._turn_end_task
        await asyncio.sleep(0.05)
    assert _stt_korningar(seen) == 1 and len(anrop) == 1  # the early one is used, not asked again
    assert anrop[0] < slutlangd[0]  # and it was asked on the audio up to "preend", not the whole turn
    assert len(said) == 1 and said[0][0] in bana0.OK_VARIANTER and said[0][1] == "kontoret" and idle == ["bana0"]
    assert _kinds(google) == []  # a hit: Google never heard the order


@pytest.mark.asyncio
async def test_tal_som_fortsatte_efter_pausen_kastar_den_tidiga_texten(ha_svarar, monkeypatch):
    ha_svarar.append(None)
    server, port, seen = await _wyoming(_transcript("vad är klockan i morgon"))
    connection, service, google, said, idle = _gemini_koppling(
        ("127.0.0.1", port), [None, "start", "preend", None, None, None, None, "end"])
    anrop, slutlangd = _spela_in(monkeypatch, connection)
    async with server:
        await _speak(connection, service, 8)
        await service._turn_end_task
    # The turn grew by far more than quiet since "preend": asked again on the whole turn.
    assert len(anrop) == 2 and anrop[0] < anrop[1] == slutlangd[0]


@pytest.mark.asyncio
async def test_utan_preend_fraga_bana_0_som_forut(ha_svarar):
    ha_svarar.append(None)
    server, port, seen = await _wyoming(_transcript("hur varmt är det"))
    connection, service, google, said, idle = _gemini_koppling(
        ("127.0.0.1", port), [None, "start", None, None, "end"])
    async with server:
        await _speak(connection, service, 5)
        await service._turn_end_task
    assert _stt_korningar(seen) == 1


def test_kontrollen_av_den_tidiga_texten_ar_pa_som_standard():
    from app import websocket_handler
    assert websocket_handler.SPEC_STT_KONTROLL is True


@pytest.fixture(autouse=True)
def _utan_kontroll(monkeypatch, request):
    """The check reads the turn a second time; the other tests count speech-to-text runs."""
    from app import websocket_handler
    if "kontroll" not in request.node.name:
        monkeypatch.setattr(websocket_handler, "SPEC_STT_KONTROLL", False)


@pytest.mark.asyncio
async def test_kontroll_varnar_nar_tidig_text_skiljer_sig_fran_hela_turen(ha_svarar, monkeypatch, caplog):
    import logging
    caplog.set_level(logging.INFO)
    ha_svarar.append("Släckte i kontoret")
    server, port, seen = await _wyoming(_transcript("släck kontoret"))
    connection, service, google, said, idle = _gemini_koppling(
        ("127.0.0.1", port), [None, "start", "preend", None, "end"])
    riktig = bana0.transkribera
    anrop = []

    async def transkribera(pcm, host, port_, timeout):
        anrop.append(len(pcm))
        text = await riktig(pcm, host, port_, timeout)
        return text if len(anrop) == 1 else "Så kan."  # the second reading of the turn differs

    monkeypatch.setattr(bana0, "transkribera", transkribera)
    async with server:
        await _speak(connection, service, 5)
        await service._turn_end_task
        await asyncio.sleep(0.3)
    assert len(anrop) == 2 and anrop[0] < anrop[1]
    assert any("differs from the whole turn 'Så kan.'" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_varning_nar_banans_ljud_och_motorns_skiljer_mer_an_0_3_s(ha_svarar, monkeypatch, caplog):
    import logging
    caplog.set_level(logging.INFO)
    ha_svarar.append("Släckte i kontoret")
    server, port, seen = await _wyoming(_transcript("släck kontoret"))
    connection, service, google, said, idle = _gemini_koppling(
        ("127.0.0.1", port), [None, "start", "preend", None, "end"])
    monkeypatch.setattr(type(service), "held_seconds", lambda self: 99.0, raising=False)
    async with server:
        await _speak(connection, service, 5)
        await service._turn_end_task
        await asyncio.sleep(0.05)
    assert any("the engine holds 99.00 s" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    assert any("turn audio starts, phase" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_motorn_med_mindre_ljud_an_banan_ar_normalt_och_ger_bara_info(ha_svarar, monkeypatch, caplog):
    import logging
    caplog.set_level(logging.INFO)
    ha_svarar.append("Släckte i kontoret")
    server, port, seen = await _wyoming(_transcript("släck kontoret"))
    connection, service, google, said, idle = _gemini_koppling(
        ("127.0.0.1", port), [None, "start", "preend", None, "end"])
    monkeypatch.setattr(type(service), "held_seconds", lambda self: 0.0, raising=False)
    async with server:
        await _speak(connection, service, 5)
        await service._turn_end_task
        await asyncio.sleep(0.05)
    rader = [r for r in caplog.records if "the engine holds 0.00 s" in r.getMessage()]
    assert rader and all(r.levelno == logging.INFO for r in rader)


@pytest.mark.asyncio
async def test_turens_ljud_sparas_bara_nar_katalogen_ar_satt(ha_svarar, monkeypatch, tmp_path):
    from app import websocket_handler
    import json as _json
    import wave

    async def kor(katalog):
        monkeypatch.setattr(websocket_handler, "TUR_LJUD_DIR", katalog)
        ha_svarar.append("Släckte i kontoret")
        server, port, seen = await _wyoming(_transcript("släck kontoret"))
        connection, service, google, said, idle = _gemini_koppling(
            ("127.0.0.1", port), [None, "start", "preend", None, "end"])
        async with server:
            await _speak(connection, service, 5)
            await service._turn_end_task
            await asyncio.sleep(0.05)

    await kor("")
    assert list(tmp_path.iterdir()) == []
    await kor(str(tmp_path))
    wavs = sorted(tmp_path.glob("*.wav"))
    assert len(wavs) == 1
    with wave.open(str(wavs[0])) as f:
        assert f.getframerate() == 16000 and f.getnframes() > 0
    assert wavs[0].with_suffix(".txt").read_text(encoding="utf-8") == "släck kontoret"
    assert "forrulle_bytes" in _json.loads(wavs[0].with_suffix(".json").read_text())


def test_sparade_turer_ar_bara_for_agaren_gamla_rensas_och_namnet_ar_rent(monkeypatch, tmp_path):
    import os
    import stat
    import time as _time
    from app import websocket_handler as wh

    katalog = tmp_path / "turer"
    monkeypatch.setattr(wh, "TUR_LJUD_DIR", str(katalog))
    gammal = tmp_path / "x"
    katalog.mkdir(mode=0o755)
    gammal = katalog / "20200101-000000-koket.wav"
    gammal.write_bytes(b"x")
    os.utime(gammal, (_time.time() - 8 * 86400,) * 2)
    wh._spara_tur("../../etc/koket", b"\x01\x00" * 1600, 100, "hej")
    assert not gammal.exists()  # older than a week: deleted at the next save
    assert stat.S_IMODE(os.stat(katalog).st_mode) == 0o700
    filer = sorted(katalog.iterdir())
    assert len(filer) == 3 and all(f.parent == katalog for f in filer)  # the name cannot climb out
    assert all(stat.S_IMODE(os.stat(f).st_mode) == 0o600 for f in filer)
    assert "/" not in filer[0].name and ".." not in filer[0].name


def test_jamforelseverktyget_skrubbar_nyckeln_ur_felen(monkeypatch):
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "gemini_jamforelse", pathlib.Path(__file__).parent.parent / "tools" / "gemini_jamforelse.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError:
        pytest.skip("google-genai not installed here")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyHEMLIG-nyckel_123")
    url = "ConnectionError('wss://x/v1beta/ws?key=AIzaSyHEMLIG-nyckel_123&alt=sse')"
    rent = mod.skrubba(url + " och AIzaSyHEMLIG-nyckel_123 en gång till")
    assert "AIzaSy" not in rent and "<dold>" in rent
