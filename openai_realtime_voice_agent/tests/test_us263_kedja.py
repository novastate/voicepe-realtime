"""US-263: a new turn does not reach the model in the middle of the earlier turn's tool chain."""
import asyncio
import time

import pytest

from app.providers import build_service
from app.providers import gemini_live as gl
from test_gemini_provider import OPENAI_SHAPE, _frame, _options, _wire


def _bygg():
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    sent = _wire(service, None)
    service.sover = False
    return service, sent


def _hall(service):
    gl._held_start(service, [_frame(1)], "test")


@pytest.mark.asyncio
async def test_en_ny_tur_vantar_tills_ordens_kedja_ar_klar():
    service, sent = _bygg()
    service._answer_after_tool = True        # the order's tool step is open
    service._kedja_t = time.monotonic()
    _hall(service)
    task = asyncio.create_task(service.answer_turn())
    await asyncio.sleep(0.4)
    assert sent == [] and not task.done()    # held: nothing reached the model
    service._answer_after_tool = False       # the model began to answer the order
    await asyncio.wait_for(task, 2)
    assert sent                              # now the next turn goes


@pytest.mark.asyncio
async def test_utan_oppen_kedja_gar_turen_direkt():
    service, sent = _bygg()
    _hall(service)
    await asyncio.wait_for(service.answer_turn(), 1)
    assert sent


@pytest.mark.asyncio
async def test_en_tyst_kedja_haller_inte_for_evigt(monkeypatch):
    service, sent = _bygg()
    monkeypatch.setattr(gl, "KEDJA_LUCKA_S", 0.3)
    service._answer_after_tool = True
    service._kedja_t = time.monotonic()
    _hall(service)
    await asyncio.wait_for(service.answer_turn(), 2)   # quiet for 0.3 s: the chain is dead, the turn goes
    assert sent


@pytest.mark.asyncio
async def test_ett_pagaende_verktyg_raknas_inte_som_tystnad(monkeypatch):
    service, sent = _bygg()
    monkeypatch.setattr(gl, "KEDJA_LUCKA_S", 0.2)
    monkeypatch.setattr(gl, "KEDJA_TAK_S", 1.0)
    service._answer_after_tool = True
    service._kedja_t = time.monotonic()
    service._unanswered_tools = {"call_1"}              # a tool is still running
    _hall(service)
    task = asyncio.create_task(service.answer_turn())
    await asyncio.sleep(0.5)
    assert sent == [] and not task.done()
    await asyncio.wait_for(task, 2)                      # the cap ends it
    assert sent


@pytest.mark.asyncio
async def test_texttur_vantar_ocksa():
    service, sent = _bygg()
    service._answer_after_tool = True
    service._kedja_t = time.monotonic()
    _hall(service)
    task = asyncio.create_task(service.answer_turn_text("Är det tänt i kontoret?"))
    await asyncio.sleep(0.3)
    assert sent == []
    service._answer_after_tool = False
    await asyncio.wait_for(task, 2)
    assert sent


@pytest.mark.asyncio
async def test_ett_nytt_verktygssteg_flyttar_tystnadsklockan(monkeypatch):
    service, sent = _bygg()
    monkeypatch.setattr(gl, "KEDJA_LUCKA_S", 0.5)
    service._answer_after_tool = True
    service._kedja_t = time.monotonic()
    _hall(service)
    task = asyncio.create_task(service.answer_turn())
    for _ in range(4):                       # a new tool step every 0.3 s: never quiet for 0.5 s
        await asyncio.sleep(0.3)
        service._kedja_t = time.monotonic()
    assert sent == [] and not task.done()
    await asyncio.wait_for(task, 2)          # quiet from now on: it goes
    assert sent


@pytest.mark.asyncio
async def test_den_vantande_turen_kan_raddas_om_lanken_bryts():
    service, sent = _bygg()
    service._answer_after_tool = True
    service._kedja_t = time.monotonic()
    _hall(service)
    task = asyncio.create_task(service.answer_turn())
    await asyncio.sleep(0.3)
    assert service.take_rescue()             # the next connection gets the waiting turn
    service._answer_after_tool = False
    await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_en_kastad_tur_skickas_aldrig_av_den_gamla_vantaren():
    service, sent = _bygg()
    service._answer_after_tool = True
    service._kedja_t = time.monotonic()
    _hall(service)
    task = asyncio.create_task(service.answer_turn_text("text B"))
    await asyncio.sleep(0.3)
    await service.drop_turn()                # B thrown away while it waits
    _hall(service)                           # C starts
    service._answer_after_tool = False
    await asyncio.wait_for(task, 2)
    assert sent == [] and service._held      # B sent nothing, and C is still held for its own sender


@pytest.mark.asyncio
async def test_texttur_efter_vantan_kan_raddas():
    service, sent = _bygg()
    service._answer_after_tool = True
    service._kedja_t = time.monotonic()
    _hall(service)
    task = asyncio.create_task(service.answer_turn_text("Är det tänt?"))
    await asyncio.sleep(0.3)
    service._answer_after_tool = False
    service._turn_rescue = None              # the earlier answer's first sound cleared it
    await asyncio.wait_for(task, 2)
    assert sent and service.take_rescue()


@pytest.mark.asyncio
async def test_sju_sekunders_lucka_mitt_i_en_order_slapper_inte_turen():
    """Kontor 2026-10-30: hitta_givare 13:35:31.6 -> styr 13:35:38.7 are two steps of one order."""
    service, sent = _bygg()
    service._answer_after_tool = True
    service._kedja_t = time.monotonic() - 7.1        # the last step is 7.1 s old
    _hall(service)
    task = asyncio.create_task(service.answer_turn())
    await asyncio.sleep(0.4)
    assert sent == [] and not task.done()
    service._kedja_t = time.monotonic()
    service._answer_after_tool = False
    await asyncio.wait_for(task, 2)
    assert sent


@pytest.mark.asyncio
async def test_tystnaden_raknas_fran_verktygets_resultat(monkeypatch):
    service, sent = _bygg()
    monkeypatch.setattr(gl, "KEDJA_LUCKA_S", 0.5)
    service._answer_after_tool = True
    service._kedja_t = time.monotonic() - 12            # the call came 12 s ago; the tool is still running
    service._unanswered_tools = {"call_1"}
    _hall(service)
    task = asyncio.create_task(service.answer_turn())
    await asyncio.sleep(0.3)
    service._unanswered_tools = set()                   # the result arrives
    await asyncio.sleep(0.25)
    assert sent == [] and not task.done()               # the model gets its time for the next step
    await asyncio.wait_for(task, 2)
    assert sent


@pytest.mark.asyncio
async def test_resultatet_flyttar_klockan_aven_for_en_tur_som_kommer_efter():
    service, sent = _bygg()
    service._answer_after_tool = True
    service._kedja_t = time.monotonic() - 12            # call made 12 s ago
    service._unanswered_tools = {"call_1"}

    async def svar(**kw):
        pass
    service._session.send_tool_response = svar
    await service._tool_result("call_1", "styr", {"ok": True})   # the result is sent now
    assert time.monotonic() - service._kedja_t < 0.5
    _hall(service)
    task = asyncio.create_task(service.answer_turn())
    await asyncio.sleep(0.3)
    assert not task.done()                               # held: the model gets time for the next step
    service._answer_after_tool = False
    await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_pingtexten_vantar_ocksa():
    service, sent = _bygg()
    service._answer_after_tool = True
    service._kedja_t = time.monotonic()
    _hall(service)
    task = asyncio.create_task(service.ping_and_answer("bana 0 gjorde: tände kontoret"))
    await asyncio.sleep(0.3)
    assert sent == [] and not task.done()
    service._answer_after_tool = False
    await asyncio.wait_for(task, 2)
    assert sent
