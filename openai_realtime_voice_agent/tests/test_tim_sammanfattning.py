"""One journal line per speaker and hour (raawr US-489)."""
import logging
import re

from app.tim_sammanfattning import TYST_TIMMAR, TimSammanfattning
from app.turn_tider import TurnTider


def rader(caplog):
    return [r.getMessage() for r in caplog.records if "📊 tim" in r.getMessage()]


def test_us489_tre_turer(caplog):
    caplog.set_level(logging.INFO)
    s = TimSammanfattning()
    s.anslutning("kontoret")
    s.tur("kontoret", 1000)
    s.tur("kontoret", 2400)
    s.tur("kontoret", None, utan_ljud=True)
    assert s.skriv_timme() == [
        "📊 tim kontoret turer=3 utan_ljud=1 median_ms=1700 max_ms=2400 anslutningar=1"]
    assert rader(caplog) == ["📊 tim kontoret turer=3 utan_ljud=1 median_ms=1700 max_ms=2400 anslutningar=1"]


def test_us489_tom_timme(caplog):
    caplog.set_level(logging.INFO)
    s = TimSammanfattning()
    s.anslutning("koket")
    s.skriv_timme()
    assert rader(caplog) == ["📊 tim koket turer=0 utan_ljud=0 median_ms=- max_ms=- anslutningar=1"]
    s.skriv_timme()  # the next hour: still a line, no new connection counted
    assert rader(caplog)[-1].endswith("turer=0 utan_ljud=0 median_ms=- max_ms=- anslutningar=0")


def test_us489_tyst_efter_dygn():
    s = TimSammanfattning()
    s.anslutning("koket")
    for _ in range(TYST_TIMMAR - 1):
        assert not s.skriv_timme()[0].endswith("tyst")
    assert s.skriv_timme()[0].endswith(" tyst")   # the 24th empty hour in a row
    s.tur("koket", 900)                            # a turn ends it
    assert not s.skriv_timme()[0].endswith("tyst")
    assert not s.skriv_timme()[0].endswith("tyst")  # and the count starts again from 1


def test_us489_ingen_text(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    s = TimSammanfattning()
    monkeypatch.setattr("app.turn_tider.SAMMANFATTNING", s)   # a real TurnTider feeds the counter
    k = iter(range(100, 200))
    t = TurnTider("kontoret", lambda: float(next(k)))
    t.start(0.8)
    t.mark("enhet")
    t.start(0.8)
    t.start(0.8)  # the second turn is written without sound
    s.skriv_timme()
    rad = rader(caplog)[0]
    assert rad.startswith("📊 tim kontoret turer=2 utan_ljud=1 ")
    assert re.fullmatch(r"📊 tim kontoret( (turer|utan_ljud|median_ms|max_ms|anslutningar)=(\d+|-))+( tyst)?", rad)
