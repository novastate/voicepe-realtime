"""The AC-10 measuring command (raawr US-032)."""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "matvarden", pathlib.Path(__file__).parent.parent / "tools" / "matvarden.py")
matvarden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(matvarden)


def rad(tid, text, mod="app.session_state"):
    return f"Oct 08 x env[1]: 2026-10-08 {tid} - {mod} - INFO - {text}"


LOGG = [
    rad("10:00:00,000", "🧭 kontoret: IDLE -> WAKE (wake: device wake)"),
    rad("10:00:01,000", "🧭 kontoret: WAKE -> LISTENING (listening: phase)"),
    rad("10:00:03,000", "🧭 kontoret: LISTENING -> THINKING (thinking: phase)"),
    rad("10:00:03,300", "bana0: hit 'tänd lampan hemligt ord' -> 'Tände'", "app.bana0"),
    rad("10:00:04,000", "🧭 kontoret: THINKING -> SPEAKING (replying: phase)"),
    rad("10:00:06,000", "🧭 kontoret: SPEAKING -> LISTENING (listening: phase)"),  # an interruption
    rad("10:00:08,000", "🧭 kontoret: LISTENING -> THINKING (thinking: phase)"),
    rad("10:00:11,000", "🧭 kontoret: THINKING -> SPEAKING (replying: phase)"),
    rad("10:00:20,000", "🧭 kontoret: SPEAKING -> CLOSING (close: quiet for 30s)"),
    rad("10:01:00,000", "🧭 kontoret: CLOSING -> IDLE (closed: quiet for 30s)"),
    rad("10:05:00,000", "🧭 koket: IDLE -> WAKE (wake: device wake)"),
    rad("10:05:05,000", "🧭 koket: WAKE -> CLOSING (close: wake without speech for 5s)"),
    rad("10:05:06,000", "🧭 koket: CLOSING -> IDLE (closed: wake without speech for 5s)"),
    rad("10:05:07,000", "bana0: heard 'hemligt ord'", "app.bana0"),
]


def test_per_rum_siffror_stammer_med_journalen():
    ut = matvarden.mat(LOGG)
    k = ut["kontoret"]
    assert k["sessioner"] == 1 and k["uppkopplade_minuter"] == 1.0
    assert k["avbrott"] == 1
    assert k["stangningsorsaker"] == {"quiet for 30s": 1}
    assert k["forsta_ljud_s"] == {"p50": 1.0, "p95": 3.0, "n": 2}  # 1 s and 3 s
    assert k["reflex_s"] == {"p50": 0.3, "p95": 0.3, "n": 1}
    q = ut["koket"]
    assert q["sessioner"] == 1 and q["stangningsorsaker"] == {"wake without speech for 5s": 1}
    assert q["forsta_ljud_s"]["n"] == 0 and q["forsta_ljud_s"]["p95"] is None


def test_inget_av_det_som_sades_hamnar_i_utskriften():
    import json
    assert "hemligt" not in json.dumps(matvarden.mat(LOGG), ensure_ascii=False)
    assert "tänd lampan" not in json.dumps(matvarden.mat(LOGG), ensure_ascii=False)


def test_percentiler_med_narmaste_rang():
    assert matvarden.pct([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert matvarden.pct([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert matvarden.pct([], 50) is None


def test_tider_raderna_ger_p50_p95_efter_talets_slut_och_verktygsturer_for_sig():
    logg = [
        rad("10:00:00,000", "⏱ tider kontoret turslut=800 stt=990 enhet=1500", "app.turn_tider"),
        rad("10:00:10,000", "⏱ tider kontoret turslut=800 stt=990 enhet=2500", "app.turn_tider"),
        rad("10:00:20,000", "⏱ tider kontoret turslut=800 verktyg=1300 enhet=5000", "app.turn_tider"),
        rad("10:00:30,000", "⏱ tider kontoret turslut=800 utan_ljud", "app.turn_tider"),
    ]
    e = matvarden.mat(logg)["kontoret"]["efter_talets_slut_s"]
    assert e["n"] == 2 and e["p50"] == 1.5 and e["p95"] == 2.5
    assert e["med_verktyg_n"] == 1 and e["med_verktyg_p50"] == 5.0
