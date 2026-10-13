"""Verktygets egen logik (tools/ac_tabeller.py): journalrader, manifest, tursortering och räkning av oberoende fall. Inget ljud, inget moln."""
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import ac_journal as J  # noqa: E402
import ac_manifest as M  # noqa: E402
import ac_tabeller_md as T  # noqa: E402

RAD_SESS = "2026-10-13 15:01:12,036 - app.session_state - INFO - 🧭 kontoret: THINKING -> SPEAKING (replying: phase)"


def test_journalrad_tid_och_typ():
    t = J.tid_ur_rad(RAD_SESS)
    assert t == J.tid_ur_rad("2026-10-13 15:01:12.036 | DEBUG    | x - y")      # båda formaten
    h = J.klassa_rad(t, RAD_SESS)
    assert (h["typ"], h["enhet"], h["till"], h["orsak"]) == ("sess", "kontoret", "SPEAKING", "phase")
    h = J.klassa_rad(1.0, "x - ⏱ tider kontoret turslut=800 stt=803 modell=4105 enhet=4111")
    assert h["steg"] == {"turslut": 800, "stt": 803, "modell": 4105, "enhet": 4111} and not h["utan_ljud"]
    assert J.klassa_rad(1.0, "2026-10-13 x - 🧪 rost-prov: the recorded voice starts in the mic now")["typ"] == "rostprov"
    assert J.klassa_rad(1.0, "ointressant rad") is None


def test_svarsfonster_och_pass(tmp_path):
    f = tmp_path / "j.txt"
    f.write_text("\n".join([
        "2026-10-13 15:00:00,000 - app.session_state - INFO - 🧭 kontoret: THINKING -> SPEAKING (replying: phase)",
        "2026-10-13 15:00:09,000 - app.session_state - INFO - 🧭 kontoret: SPEAKING -> IDLE (idle: phase)",
        "2026-10-13 15:00:09,000 - app.session_state - INFO - 🧭 kontoret: SPEAKING -> IDLE (idle: phase)",   # dubblett: räknas en gång
        "2026-10-13 17:00:00,000 - app.session_state - INFO - 🧭 kontoret: THINKING -> SPEAKING (replying: phase)",
        "2026-10-13 17:00:05,000 - app.session_state - INFO - 🧭 kontoret: SPEAKING -> LISTENING (listening: phase)",
    ]) + "\n")
    h = J.las_journal([str(f)])
    assert len(h) == 4 and h[0]["radnr"] == 1                       # spårbart: fil + radnummer
    fon = J.svarsfonster(h)
    assert [(round(w["t1"] - w["t0"])) for w in fon] == [9, 5]
    p = J.bygg_pass(h, [])
    assert len(p) == 2 and p[0]["id"] == "P01"                      # 2 h lucka = två pass
    assert J.pass_for(p, h[0]["t"] + 3)["id"] == "P01"


def test_manifest_dubbletter_arbetsfiler_och_justering(tmp_path):
    def wav(path, ref_s=2.0):
        a = np.zeros((16000 * 3, 2), np.int16)
        a[:int(16000 * ref_s), 1] = 1000
        with wave.open(str(path), "wb") as w:
            w.setnchannels(2), w.setsampwidth(2), w.setframerate(16000)
            w.writeframes(a.tobytes())
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    wav(a / "aec_20261013_150000.wav")
    wav(b / "aec_20261013_150000.wav")            # samma bytes: dubblett
    wav(a / "aec_20261013_150000_ren.wav", 1.0)   # arbetsfil, inget fall
    j = tmp_path / "j.txt"
    j.write_text("\n".join(f"2026-10-13 14:59:5{i},500 - app.raw_audio_serializer - INFO - ⏱ device: audio burst rx_ms={i} play_ms={i + 330} (rx->play 330 ms), first 1024 B" for i in range(0, 3)) + "\n")
    m = M.bygg([str(a), str(b)], [str(j)])
    assert len(m["inspelningar"]) == 1 and m["dubbletter"] == 1
    assert m["ignorerade"]["antal"]["arbetsfil (härledd/syntetisk/delad kanal)"] == 1
    r = m["inspelningar"][0]
    assert r["just"]["poang"] is not None and r["just"]["spelstart_ms"] == 330 and r["pass"] == "P01"


def test_oberoende_fall_raknas_per_inspelning():
    rader = [{"inspelning": "a", "klippt": True}, {"inspelning": "a", "klippt": True}, {"inspelning": "a", "klippt": True},
             {"inspelning": "b", "klippt": False}, {"inspelning": "b", "klippt": True}, {"inspelning": "c", "klippt": False}]
    o = J.oberoende(rader)
    assert o == {"korningar": 6, "inspelningar": 3, "klippt_korningar": 4, "klippt_alla": 1, "klippt_nagon": 2}


def test_tursort_delar_bara_pa_det_som_ar_kant_fore():
    ev = [{"t": 10.0, "till": "LISTENING", "orsak": "phase"}]
    klipp = [{"t": 8.0}]
    tider = lambda steg, t=12.0: {"t": t, "steg": steg}
    assert T.tursort([], tider({"enhet": 1}), ev, []) == "bana 0 (lokalt svar)"
    assert T.tursort([], tider({"modell": 1, "enhet": 2}), ev, []) == "modellfråga"
    assert T.tursort([], tider({"verktyg": 1, "modell": 2, "enhet": 3}), ev, []) == "verktyg"
    assert T.tursort([], tider({"modell": 1, "enhet": 2}), ev, klipp) == "avbrott (turen kom efter ett klipp)"
    ev2 = [{"t": 10.0, "till": "LISTENING", "orsak": "fraga-prov"}]
    assert T.tursort([], tider({"modell": 1, "enhet": 2}), ev2, klipp) == "provrunda (inlagd fråga)"


def test_erle_hinkar_och_median():
    assert [T.erle_hink(x) for x in (None, 5.0, 15.0, 25.0)] == ["ej mätt", "<10 dB", "10-20 dB", ">=20 dB"]
    assert J.median([3, 1, 2]) == 2 and J.median([1, 2, 3, 4]) == 2.5 and J.median([]) is None
