"""US-223 (D-96): a new phrase that begins while bana 0 still decides the one before is a turn of its own."""
import asyncio

import pytest


def _service(events):
    from test_gemini_provider import _ScriptedTurns, _options, _wire, OPENAI_SHAPE
    from app.providers import build_service

    service = build_service("gemini", _options(), OPENAI_SHAPE)
    _wire(service, _ScriptedTurns(events))
    return service


async def _first_turn_held(service, frisl, avgjorde, hit):
    async def on_end():
        avgjorde.append(list(service._held or []))
        if len(avgjorde) == 1:
            await frisl.wait()
            if hit:
                await service.drop_turn()  # a hit: Live never hears the turn
                return
        service._held = None  # a miss: answer_turn clears the held turn

    service.on_user_turn_end = on_end


EVENTS = ["start", None, "end", "start", None, "end"]


@pytest.mark.asyncio
async def test_ac1_hit_on_the_first_phrase_keeps_the_second():
    from test_gemini_provider import _frame

    service = _service(EVENTS)
    frisl, avgjorde = asyncio.Event(), []
    await _first_turn_held(service, frisl, avgjorde, hit=True)
    for b in range(3):
        await service._send_user_audio(_frame(b))
    await asyncio.sleep(0.05)
    assert len(avgjorde) == 1  # the hook is waiting for comms
    for b in range(3, 6):  # the new phrase begins during the decision
        await service._send_user_audio(_frame(b))
    frisl.set()
    await asyncio.sleep(0.1)
    assert len(avgjorde) == 2, "the second phrase must be decided on its own"
    second = b"".join(f.audio for f in avgjorde[1])
    assert _frame(3).audio in second and _frame(5).audio in second
    assert _frame(0).audio not in second


@pytest.mark.asyncio
async def test_ac2_miss_on_the_first_phrase_gives_two_turns_in_order():
    from test_gemini_provider import _frame

    service = _service(EVENTS)
    frisl, avgjorde = asyncio.Event(), []
    await _first_turn_held(service, frisl, avgjorde, hit=False)
    for b in range(6):
        await service._send_user_audio(_frame(b))
        if b == 2:
            await asyncio.sleep(0.05)
    frisl.set()
    await asyncio.sleep(0.1)
    assert len(avgjorde) == 2
    first = b"".join(f.audio for f in avgjorde[0])
    second = b"".join(f.audio for f in avgjorde[1])
    assert _frame(2).audio in first and _frame(3).audio not in first
    assert _frame(3).audio in second and _frame(2).audio not in second


@pytest.mark.asyncio
async def test_ac3_without_a_new_phrase_the_turn_is_decided_as_before():
    from test_gemini_provider import _frame

    service = _service(["start", None, "end"])
    frisl, avgjorde = asyncio.Event(), []
    frisl.set()
    await _first_turn_held(service, frisl, avgjorde, hit=True)
    for b in range(3):
        await service._send_user_audio(_frame(b))
    await asyncio.sleep(0.1)
    assert len(avgjorde) == 1 and len(avgjorde[0]) == 4  # the pre-roll frame plus three
    assert service._forts is None and service._held is None


@pytest.mark.asyncio
async def test_vanta_keeps_the_next_speech_in_the_same_held_turn():
    """Granskarens F1: after "vänta" (fortsatt_tur) the held turn stays held and what he says next joins it."""
    from test_gemini_provider import _frame

    service = _service(["start", None, "end", "start", None, "end"])
    service.on_user_turn_end = lambda: asyncio.sleep(10)
    for b in range(3):
        await service._send_user_audio(_frame(b))
    service._vantar = True  # bana 0's fortsatt_tur: the turn was only "vänta"
    held_before = service._held
    for b in range(3, 5):
        await service._send_user_audio(_frame(b))
    assert service._forts is None, "vänta must not split the speech into a second turn"
    assert service._held is held_before and _frame(4).audio in b"".join(f.audio for f in service._held)


@pytest.mark.asyncio
async def test_hit_while_the_second_phrase_is_still_spoken_keeps_it_as_preroll():
    """Granskarens F2: drop_turn arrives before the second phrase ends; what was said of it goes back as pre-roll."""
    from test_gemini_provider import _frame

    service = _service(["start", None, "end", "start", None])
    frisl, avgjorde = asyncio.Event(), []
    await _first_turn_held(service, frisl, avgjorde, hit=True)
    for b in range(5):
        await service._send_user_audio(_frame(b))
        if b == 2:
            await asyncio.sleep(0.05)
    assert service._forts is not None and len(service._forts) == 2
    await service.drop_turn()
    assert service._forts is None and service._held is None
    assert bytes(service._preroll) == _frame(3).audio + _frame(4).audio
