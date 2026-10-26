"""US-229: with the skill catalogue on, the session still gets fraga_core (the only road to a fact under EN_HJARNA)."""
import pytest

from app import skills


@pytest.mark.parametrize("lage", ["b", "c"])
def test_ac1_fraga_core_behalls_i_skilllaget(monkeypatch, lage):
    monkeypatch.setenv("ROST_SKILLS", lage)
    assert skills.behall("fraga_core")
    assert skills.behall("set_timer") and skills.behall("las_skill")   # the ten of today are unchanged


@pytest.mark.parametrize("namn", ["web_search", "ask_openclaw", "recall_memory", "HassTurnOn", "webbsok", "fraga_huset"])
def test_ac2_andra_dolda_verktyg_forblir_dolda(monkeypatch, namn):
    monkeypatch.setenv("ROST_SKILLS", "c")
    assert not skills.behall(namn)
