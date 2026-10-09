"""One timing line per turn (raawr US-032 / US-047 AC-1)."""
import logging

from app.turn_tider import TurnTider


class Klocka:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def rader(caplog):
    return [r.getMessage() for r in caplog.records if "⏱ tider" in r.getMessage()]


def test_en_rad_per_tur_med_stegen_i_millisekunder_efter_talets_slut(caplog):
    caplog.set_level(logging.INFO)
    k = Klocka()
    t = TurnTider("kontoret", k)
    t.start(0.8)  # the turn ended 0.8 s after the last word
    for steg, dt in (("stt", 0.19), ("comms", 0.15), ("modell", 1.2), ("enhet", 0.05)):
        k.t += dt
        t.mark(steg)
    assert rader(caplog) == ["⏱ tider kontoret turslut=800 stt=990 comms=1140 modell=2340 enhet=2390"]


def test_steg_som_inte_hande_utelamnas_och_verktyg_syns(caplog):
    caplog.set_level(logging.INFO)
    k = Klocka()
    t = TurnTider("koket", k)
    t.start(0.8)
    k.t += 0.5
    t.mark("verktyg")
    t.mark("verktyg")  # only the first counts
    k.t += 1.0
    t.mark("enhet")
    assert rader(caplog) == ["⏱ tider koket turslut=800 verktyg=1300 enhet=2300"]


def test_en_tur_utan_ljud_skrivs_vid_nasta_tur_och_raden_har_ingen_text(caplog):
    caplog.set_level(logging.INFO)
    k = Klocka()
    t = TurnTider("kontoret", k)
    t.start(0.8)
    k.t += 5
    t.start(0.8)  # the next turn: the first one never got sound
    t.mark("hemligt ord")  # an unknown mark is stored but never printed
    k.t += 1
    t.mark("enhet")
    r = rader(caplog)
    assert r[0].endswith("utan_ljud") and "enhet" not in r[0]
    assert all("hemligt" not in x for x in r)
    assert t.t is None


import pytest  # noqa: E402


@pytest.mark.asyncio
async def test_bana_0_klipp_som_gar_direkt_till_enheten_markerar_enhet_utan_utan_ljud(caplog):
    """B on #34: bana 0's own clips and the receipts do not pass the serializer."""
    from types import SimpleNamespace

    from app.websocket_handler import WebSocketHandler

    caplog.set_level(logging.INFO)
    skickat = []

    async def send(data):
        skickat.append(data)

    handler = WebSocketHandler()
    connection = SimpleNamespace(device_id="kontoret", transport=SimpleNamespace(client=SimpleNamespace(send=send)))
    k = Klocka()
    connection.tider = TurnTider("kontoret", k)
    handler.resolve_device = lambda device_id=None: connection
    connection.tider.start(0.8)  # a bana 0 turn: the speech-to-text is done, the clip goes out
    k.t += 0.3
    connection.tider.mark("stt")
    k.t += 0.2
    assert await handler.send_bytes_to(b"\x00\x00" * 100, "kontoret") is True
    k.t += 0.1
    await handler.send_bytes_to(b"\x00\x00" * 100, "kontoret")  # the second chunk of the clip: not a new line
    r = rader(caplog)
    assert r == ["⏱ tider kontoret turslut=800 stt=1100 enhet=1300"]
    assert not any("utan_ljud" in x for x in r) and len(skickat) == 2
