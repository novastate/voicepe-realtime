"""US-060 layer 2: his own words heard back are ignored, other speech is an interruption."""
from app import eko_vakt as ev

BJORN = "Jag sätter på radion i kontoret nu. Det är P3 som spelar, och den går ganska högt."


def test_his_own_words_are_echo():
    assert ev.ar_eko("sätter på radion i kontoret", BJORN)
    assert ev.ar_eko("det är p3 som spelar", BJORN)


def test_an_inexact_local_transcript_of_his_voice_is_still_echo():
    assert ev.ar_eko("jag sätta på radio i kontor", BJORN)  # near enough, word by word


def test_other_speech_is_an_interruption():
    assert not ev.ar_eko("vänta stopp", BJORN)
    assert not ev.ar_eko("nej lägg till mjölk på listan", BJORN)


def test_nothing_intelligible_is_not_an_interruption():
    assert ev.ar_eko(None, BJORN) and ev.ar_eko("", BJORN) and ev.ar_eko("...", BJORN)


def test_before_he_has_said_anything_any_speech_is_a_person():
    assert not ev.ar_eko("hallå", "")


def test_a_mixed_utterance_below_the_share_is_an_interruption():
    # one of four words is his: 25 % < 60 %
    assert not ev.ar_eko("stopp vänta lite radion", BJORN)


def test_svar_keeps_the_current_reply_and_the_last_two():
    s = ev.Svar()
    for t in ("ett svar", "två svar", "tre svar"):
        s.mata(t)
        s.klar()
    s.mata("fyra pågår")
    assert "ett" not in s.text() and "två" in s.text() and "tre" in s.text() and "fyra" in s.text()


def test_the_tail_after_a_reply():
    assert ev.i_svans(100.0, nu=101.0) and not ev.i_svans(100.0, nu=102.0) and not ev.i_svans(0.0, nu=1.0)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("BARGE_IN", raising=False)
    assert not ev.paa()
    monkeypatch.setenv("BARGE_IN", "1")
    assert ev.paa()
    monkeypatch.setenv("ANTI_EKO", "0")
    assert not ev.nat_paa()
