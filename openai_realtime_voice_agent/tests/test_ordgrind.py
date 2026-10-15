"""GEMINI_ORDGRIND (raawr US-060, Hemma 2026-10-15): over his reply the words decide whether it is an interruption."""
import asyncio

import pytest
from pipecat.frames.frames import InputAudioRawFrame, UserStartedSpeakingFrame

from app import eko_vakt
from test_gemini_egen_tur import _bygg

HANS = "Bränsle och luft sugs in, komprimeras av kolven och tänds av ett tändstift. Smällen trycker ner kolven och driver vevaxeln runt."


@pytest.mark.parametrize("hort, ok, under, person", [
    ("Ja.", True, True, False),                  # a back-channel while he speaks: he goes on (AC-8)
    ("Okej.", True, True, False),
    ("Mm, precis.", True, True, False),
    ("Ja.", True, False, True),                  # just after he stopped it answers him
    ("Vänta.", True, True, True),
    ("Vad är status på huset?", True, True, True),
    ("Och smällen. Ja.", True, True, False),     # his own echo with a "ja" in it (06:51:24)
    ("Trycker ner kolven.", True, True, False),
    ("", True, True, False),
    ("Tack till elever och personal vid Värmland.", True, True, False),   # what the speech-to-text writes of noise
    (None, False, True, False),                  # no text while he speaks: he goes on
    (None, False, False, True),                  # ...just after: in doubt it is the person
])
def test_the_words_decide(hort, ok, under, person):
    assert eko_vakt.ord_dom(hort, ok, HANS, under) is person


# B on #79: a short command that shares a word with what he just said is a person, not his echo.
@pytest.mark.parametrize("hort", ["Tänd garaget.", "Släck garaget.", "Tänd badrummet.", "Släck köket.", "Tänd lampan.",
                                  "Släck hallen."])
def test_a_short_command_with_one_of_his_words_is_a_person(hort):
    assert eko_vakt.ord_dom(hort, True, "Jag tänder köket och släcker lampan i hallen.", True) is True


def test_over_him_is_decided_where_the_speech_began():
    assert eko_vakt.ord_under(True, 5.0)          # he was speaking
    assert eko_vakt.ord_under(False, 0.4)         # his last words are still in the device's buffer
    assert not eko_vakt.ord_under(False, 0.7)     # "Ja." 0.7 s after he stopped answers him (B on #79)
    assert eko_vakt.ord_dom("Ja.", True, HANS, eko_vakt.ord_under(False, 0.7)) is True


def test_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("GEMINI_ORDGRIND", raising=False)
    assert not eko_vakt.ordgrind("kontoret")
    monkeypatch.setenv("GEMINI_ORDGRIND", "kontoret")
    assert eko_vakt.ordgrind("kontoret") and not eko_vakt.ordgrind("koket")
    monkeypatch.setenv("GEMINI_ORDGRIND", "1")
    assert eko_vakt.ordgrind("koket")


def _ljud(amp, ms=20):
    import numpy as np
    return InputAudioRawFrame(audio=(np.ones(16 * ms, dtype=np.int16) * amp).tobytes(), sample_rate=16000, num_channels=1)


def _grind(monkeypatch, dom):
    service, sent, pushed = _bygg(monkeypatch, [None] * 1000)
    service.svarar = lambda: True
    asked, barge = [], []

    async def ordgrind(sek, sedan_start, svarade):
        asked.append(sek)
        return dom
    service.ordgrind_vakt = lambda: not barge   # as live: the cut reply (tysta_svar) ends the watch
    service.ordgrind = ordgrind
    service.on_barge_in = lambda: barge.append(1)
    return service, sent, pushed, asked, barge


async def _mata(service, frames):
    for f in frames:
        await service._send_user_audio(f)
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_over_his_reply_the_engine_hears_nothing_of_a_quiet_room(monkeypatch):
    service, sent, pushed, asked, barge = _grind(monkeypatch, True)
    await _mata(service, [_ljud(30)] * 100)   # his echo far under the floor, 2 s
    assert sent == [] and asked == [] and barge == []


@pytest.mark.asyncio
async def test_a_back_channel_is_held_judged_and_dropped(monkeypatch):
    service, sent, pushed, asked, barge = _grind(monkeypatch, False)
    await _mata(service, [_ljud(30)] * 20 + [_ljud(8000)] * 20 + [_ljud(30)] * 40)
    assert len(asked) == 1 and asked[0] >= service.ORD_FORE_S + service.ORD_FONSTER_S
    assert sent == [] and barge == [] and not any(isinstance(f, UserStartedSpeakingFrame) for f in pushed)


@pytest.mark.asyncio
async def test_a_person_cuts_him_and_the_engine_gets_all_of_it(monkeypatch):
    service, sent, pushed, asked, barge = _grind(monkeypatch, True)
    fore = [_ljud(30)] * 20
    tal = [_ljud(8000)] * 60
    await _mata(service, fore + tal)
    assert barge == [1] and any(isinstance(f, UserStartedSpeakingFrame) for f in pushed)
    # the 0.3 s before the speech, the speech up to the verdict and after it: nothing lost, and the fore-roll in front
    assert len(sent) == 15 + 60 and sent[0]["audio"].data == _ljud(30).audio


@pytest.mark.asyncio
async def test_a_failing_verdict_lets_him_go_on(monkeypatch):
    service, sent, pushed, asked, barge = _grind(monkeypatch, True)

    async def trasig(sek, sedan_start, svarade):
        raise RuntimeError("stt down")
    service.ordgrind = trasig
    await _mata(service, [_ljud(8000)] * 60)
    assert sent == [] and barge == []


@pytest.mark.asyncio
async def test_with_the_gate_off_everything_is_streamed_as_before(monkeypatch):
    service, sent, pushed = _bygg(monkeypatch, [None] * 1000)
    service.svarar = lambda: True
    await _mata(service, [_ljud(8000)] * 3)
    assert len(sent) == 3


@pytest.mark.asyncio
async def test_the_words_are_read_before_the_canceller():
    from app.raw_audio_serializer import RawAudioSerializer
    ser = RawAudioSerializer("kontoret", input_sample_rate=16000)

    class Tom:   # a canceller that takes everything out
        def process(self, pcm):
            return bytes(len(pcm))
    ser.aec = Tom()
    await ser.deserialize(b"\x10\x27" * 1600)
    assert ser.ra_mik(3200) == b"\x10\x27" * 1600


@pytest.mark.asyncio
async def test_after_his_reply_nothing_is_muted_when_a_person_is_let_in(monkeypatch):
    service, sent, pushed, asked, barge = _grind(monkeypatch, True)
    service.svarar = lambda: False     # he has stopped; only his tail is in the room
    await _mata(service, [_ljud(8000)] * 60)
    assert barge == [] and len(sent) > 0


@pytest.mark.asyncio
async def test_the_end_of_the_speech_ends_the_mute(monkeypatch):
    service, sent, pushed = _bygg(monkeypatch, ["start", None, "end"])
    service.svarar = lambda: False
    slut = []
    service.on_tal_slut = lambda: slut.append(1)
    await _mata(service, [_ljud(8000)] * 3)
    assert slut == [1]


def test_the_engine_taking_the_turn_ends_the_mute(monkeypatch):
    from test_gemini_egen_tur import _Ser, _koppla
    slut = []
    monkeypatch.setattr(_Ser, "tysta_slut", lambda self: slut.append(1))
    tjanst, conn, fas = _koppla()
    tjanst.on_motor_avbrott()
    assert slut == [1]


@pytest.mark.asyncio
async def test_after_him_is_measured_before_the_slow_speech_to_text():
    import time
    from app.websocket_handler import koppla_ordgrind

    class Ser:
        sist = time.monotonic()
        def sedan_svar(self): return time.monotonic() - self.sist
        def ra_mik(self, n): return b"\0" * n
        def tystad(self): return False
        def tysta_slut(self): pass

    async def langsam(pcm):
        await asyncio.sleep(0.5)
        return "Ja.", True
    service = type("S", (), {"svar": None})()
    ser = Ser()
    koppla_ordgrind(service, ser, type("F", (), {"phase": "idle"})(), langsam)
    # "Ja." began 0.3 s after his last audio and the verdict is asked for 1.0 s after it began
    ser.sist = time.monotonic() - 1.3
    assert await service.ordgrind(1.0, 1.0, False) is False   # over him: a back-channel, he goes on
