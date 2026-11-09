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


def _logger(**kw):
    """A TranscriptLogger whose downstream push is a no-op, so process_frame can be called on its own."""
    lg = TranscriptLogger(**kw)

    async def _push(frame, direction=None):
        pass

    lg.push_frame = _push
    return lg


@pytest.mark.asyncio
async def test_transkriptloggern_anropar_hord_for_stt_text_och_for_inget_annat(monkeypatch):
    from pipecat.frames.frames import TextFrame, TranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection
    skickat = []
    monkeypatch.setattr(hord, "rapportera", lambda t, rutt="hort": skickat.append((rutt, t)))
    lg = _logger(capture="user", on_user=hord.rapportera)
    await lg.process_frame(TranscriptionFrame(text="ja gör det", user_id="u", timestamp="t"), FrameDirection.UPSTREAM)
    await lg.process_frame(TextFrame(text="modellens egen text"), FrameDirection.DOWNSTREAM)
    await lg.process_frame(TranscriptionFrame(text="   ", user_id="u", timestamp="t"), FrameDirection.UPSTREAM)
    assert skickat == [("hort", "ja gör det")]


@pytest.mark.asyncio
async def test_transkriptloggern_rapporterar_assistentens_uttalade_svar_som_levererat(monkeypatch):
    from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, LLMFullResponseEndFrame, TTSTextFrame
    from pipecat.processors.frame_processor import FrameDirection
    skickat = []
    monkeypatch.setattr(hord, "rapportera", lambda t, rutt="hort": skickat.append((rutt, t)))
    lg = _logger(capture="assistant", on_assistant=hord.levererat)
    for bit in ("Ska jag ta bort ", "Tandläkare?"):
        await lg.process_frame(TTSTextFrame(text=bit, aggregated_by="sentence"), FrameDirection.DOWNSTREAM)
    await lg.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    assert skickat == []  # the text is finished but nothing has been heard yet
    await lg.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
    await lg.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    assert skickat == [("levererat", "Ska jag ta bort Tandläkare?")]


def test_websocket_handler_kopplar_bada_riktningarna():
    """The wiring itself: both taps carry their callback (the mutant below removes one)."""
    import inspect
    from app import websocket_handler
    k = inspect.getsource(websocket_handler)
    assert k.count("on_user=hord.rapportera") == 2 and k.count("on_assistant=hord.levererat") == 2


@pytest.mark.asyncio
async def test_avbruten_uppläsning_rapporteras_inte_som_levererad(monkeypatch):
    from pipecat.frames.frames import BotStoppedSpeakingFrame, LLMFullResponseEndFrame, StartInterruptionFrame, TTSTextFrame
    from pipecat.processors.frame_processor import FrameDirection
    skickat = []
    monkeypatch.setattr(hord, "rapportera", lambda t, rutt="hort": skickat.append((rutt, t)))
    lg = _logger(capture="assistant", on_assistant=hord.levererat)
    await lg.process_frame(TTSTextFrame(text="Ska jag ta bort Tand", aggregated_by="sentence"), FrameDirection.DOWNSTREAM)
    await lg.process_frame(StartInterruptionFrame(), FrameDirection.DOWNSTREAM)  # ägaren avbryter mitt i meningen
    await lg.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    await lg.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    assert not any("Tand" in t for _, t in skickat)  # the cut-off question is never reported


@pytest.mark.asyncio
async def test_avbrott_efter_att_texten_blev_klar_men_under_uppspelning_ger_ingen_leverans(monkeypatch):
    """Codex P2: End -> playback going on -> StartInterruption -> the owner says 'ja'. The question was never heard to its end."""
    from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, LLMFullResponseEndFrame,
                                       StartInterruptionFrame, TTSTextFrame, TranscriptionFrame)
    from pipecat.processors.frame_processor import FrameDirection
    skickat = []
    monkeypatch.setattr(hord, "rapportera", lambda t, rutt="hort": skickat.append((rutt, t)))
    lg = _logger(capture="both", on_user=hord.rapportera, on_assistant=hord.levererat)
    await lg.process_frame(TTSTextFrame(text="Ska jag ta bort Tandläkare?", aggregated_by="sentence"), FrameDirection.DOWNSTREAM)
    await lg.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)  # text done, audio still playing
    await lg.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
    await lg.process_frame(StartInterruptionFrame(), FrameDirection.DOWNSTREAM)    # the owner barges in
    await lg.process_frame(TranscriptionFrame(text="ja", user_id="u", timestamp="t"), FrameDirection.UPSTREAM)
    await lg.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    assert skickat == [("hort", "ja")]  # the yes is heard, the question is never reported as delivered


@pytest.mark.asyncio
async def test_svar_vars_ljud_redan_tagit_slut_for_texten_blev_klar_rapporteras_inte(monkeypatch):
    """Fail-closed: no BotStoppedSpeaking after the finished text means nothing proves it was heard."""
    from pipecat.frames.frames import LLMFullResponseEndFrame, TTSTextFrame
    from pipecat.processors.frame_processor import FrameDirection
    skickat = []
    monkeypatch.setattr(hord, "rapportera", lambda t, rutt="hort": skickat.append((rutt, t)))
    lg = _logger(capture="assistant", on_assistant=hord.levererat)
    await lg.process_frame(TTSTextFrame(text="Ska jag ta bort Tandläkare?", aggregated_by="sentence"), FrameDirection.DOWNSTREAM)
    await lg.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    assert skickat == []


def _ram(namn):
    import pipecat.frames.frames as f
    return getattr(f, namn)


async def _svar(lg, text):
    """One finished reply as the pipeline delivers it: Start, text, End."""
    from pipecat.processors.frame_processor import FrameDirection
    await lg.process_frame(_ram("LLMFullResponseStartFrame")(), FrameDirection.DOWNSTREAM)
    if text:
        await lg.process_frame(_ram("TTSTextFrame")(text=text, aggregated_by="sentence"), FrameDirection.DOWNSTREAM)
    await lg.process_frame(_ram("LLMFullResponseEndFrame")(), FrameDirection.DOWNSTREAM)


@pytest.mark.asyncio
async def test_tva_svar_i_ko_en_uppspelning_ger_ingen_leverans(monkeypatch):
    """Codex P2: 'Vill du ha kaffe?' + End, kalenderfråga + End, BotStopped. Which one ended? Neither is reported."""
    from pipecat.processors.frame_processor import FrameDirection
    skickat = []
    monkeypatch.setattr(hord, "rapportera", lambda t, rutt="hort": skickat.append((rutt, t)))
    lg = _logger(capture="assistant", on_assistant=hord.levererat)
    await _svar(lg, "Vill du ha kaffe?")
    # the next reply starts while the first still waits: ambiguous, the queue is dropped
    await _svar(lg, "Ska jag ta bort Tandläkare?")
    await lg.process_frame(_ram("BotStoppedSpeakingFrame")(), FrameDirection.UPSTREAM)
    assert skickat == []  # neither the coffee question nor the calendar question is proven heard


@pytest.mark.asyncio
async def test_kalenderfraga_sedan_ny_replik_och_sent_botstopped_ger_ingen_falsk_leverans(monkeypatch):
    """Codex P2: calendar question + End, a new reply 'Vill du ha kaffe?' starts, a late BotStopped arrives: the calendar
    question must NOT be reported as delivered while the new reply is going on."""
    from pipecat.processors.frame_processor import FrameDirection
    skickat = []
    monkeypatch.setattr(hord, "rapportera", lambda t, rutt="hort": skickat.append((rutt, t)))
    lg = _logger(capture="assistant", on_assistant=hord.levererat)
    await _svar(lg, "Ska jag ta bort Tandläkare?")
    await lg.process_frame(_ram("LLMFullResponseStartFrame")(), FrameDirection.DOWNSTREAM)  # the new reply begins
    await lg.process_frame(_ram("TTSTextFrame")(text="Vill du ha kaffe?", aggregated_by="sentence"), FrameDirection.DOWNSTREAM)
    await lg.process_frame(_ram("BotStoppedSpeakingFrame")(), FrameDirection.UPSTREAM)  # late/duplicate stop
    assert skickat == []


@pytest.mark.asyncio
async def test_ett_svar_utan_ord_men_med_ljud_rapporteras_som_segment(monkeypatch):
    """Codex P2: audio without text is still a delivered segment; Core must be told so it can invalidate a waiting question."""
    from pipecat.processors.frame_processor import FrameDirection
    skickat = []
    monkeypatch.setattr(hord, "rapportera", lambda t, rutt="hort": skickat.append((rutt, t)))
    lg = _logger(capture="assistant", on_assistant=hord.levererat)
    await _svar(lg, "")
    await lg.process_frame(_ram("BotStoppedSpeakingFrame")(), FrameDirection.UPSTREAM)
    assert skickat == [("levererat", "(ljud utan text)")]


@pytest.mark.asyncio
async def test_tva_slutade_svar_utan_start_emellan_ger_ingen_leverans(monkeypatch):
    """Two finished replies queued with no Start frame between them: one BotStopped cannot be attributed to the last one."""
    from pipecat.frames.frames import LLMFullResponseEndFrame, TTSTextFrame
    from pipecat.processors.frame_processor import FrameDirection
    skickat = []
    monkeypatch.setattr(hord, "rapportera", lambda t, rutt="hort": skickat.append((rutt, t)))
    lg = _logger(capture="assistant", on_assistant=hord.levererat)
    for text in ("Vill du ha kaffe?", "Ska jag ta bort Tandläkare?"):
        await lg.process_frame(TTSTextFrame(text=text, aggregated_by="sentence"), FrameDirection.DOWNSTREAM)
        await lg.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    await lg.process_frame(_ram("BotStoppedSpeakingFrame")(), FrameDirection.UPSTREAM)
    assert skickat == []
