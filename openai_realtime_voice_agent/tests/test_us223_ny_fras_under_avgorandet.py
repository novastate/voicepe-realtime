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
