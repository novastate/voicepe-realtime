"""US-060: läs agentens journal och inspelningarna till händelser, pass och svarsfönster. Ren logik, inget ljud, inget moln.

Används av tools/ac_tabeller.py. Allt här går att prova utan inspelningar (tests/test_ac_tabeller.py).

Journalens tid är UTC (lokal tid minus 2 h i oktober). Raderna kommer i två format, båda med millisekunder:
    2026-10-13 15:01:03,783 - app.session_state - INFO - ...        (python logging)
    2026-10-13 15:01:03.783 | DEBUG    | pipecat... - ...            (loguru)
"""
import calendar
import hashlib
import os
import re
import wave
from typing import Dict, Iterable, List, Optional

import numpy as np

RATE = 16000
LOKAL_FORSKJUTNING_S = 2 * 3600          # lokal tid = UTC + 2 h (sommartid), bara för utskriften

TID = re.compile(r"^(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)[,.](\d{3})")
AEC_NAMN = re.compile(r"^aec_(\d{8})_(\d{6})\.wav$")
BURST = re.compile(r"audio burst rx_ms=(\d+) play_ms=(\d+) \(rx->play (\d+) ms\)")
SESS = re.compile(r"🧭 (\w+): (\w+) -> (\w+) \((\w+): ([^)]*)\)")
TURSTART = re.compile(r"📏 turn start level (-?[\d.]+) dBFS, (-?[\d.]+) s since idle, (-?[\d.]+) s since wake")
TIDER = re.compile(r"⏱ tider (\w+) (.*)$")
USER = re.compile(r"🗣️ user: (.*)$")
HEARD = re.compile(r"over his reply: heard (.*?) (?:\(|;|->)")
VERKTYG = re.compile(r"Calling function \[(\w+):")
BANA0_HORT = re.compile(r"bana0: heard (.*)$")


def tid_ur_rad(rad: str) -> Optional[float]:
    m = TID.match(rad)
    if not m:
        return None
    y, mo, d, h, mi, s, ms = (int(x) for x in m.groups())
    return calendar.timegm((y, mo, d, h, mi, s)) + ms / 1000.0


def tid_text(t: float, lokal: bool = False) -> str:
    import time
    g = t + (LOKAL_FORSKJUTNING_S if lokal else 0)
    return time.strftime("%m-%d %H:%M:%S", time.gmtime(g)) + f".{int((g % 1) * 1000):03d}"


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------------- journal

def las_journal(paths: Iterable[str]) -> List[dict]:
    """Alla journalrader som händelser {t, typ, rad, ...}, i tidsordning, utan dubbletter över filerna.
    Varje händelse bär 'fil' och 'rad' (radnummer) så att varje tal i en tabell går att spåra."""
    sett, ut = set(), []
    for p in paths:
        with open(p, errors="replace") as f:
            for n, rad in enumerate(f, 1):
                rad = rad.rstrip("\n")
                t = tid_ur_rad(rad)
                if t is None or rad in sett:
                    continue
                sett.add(rad)
                h = klassa_rad(t, rad)
                if h:
                    h["fil"], h["radnr"] = os.path.basename(p), n
                    ut.append(h)
    ut.sort(key=lambda h: h["t"])
    return ut


def klassa_rad(t: float, rad: str) -> Optional[dict]:
    """En journalrad till en händelse, eller None för rader som inga tabeller använder."""
    h = {"t": t, "rad": rad[24:].strip()[:240]}
    if "audio burst" in rad:
        m = BURST.search(rad)
        if m:
            return {**h, "typ": "burst", "rx_ms": int(m.group(1)), "play_ms": int(m.group(2)), "rx_till_play_ms": int(m.group(3))}
        return None
    if "device: underrun" in rad:
        return {**h, "typ": "underrun"}
    m = SESS.search(rad)
    if m:
        return {**h, "typ": "sess", "enhet": m.group(1), "fran": m.group(2), "till": m.group(3), "orsak": m.group(5)}
    m = TURSTART.search(rad)
    if m:
        return {**h, "typ": "turstart", "niva_dbfs": float(m.group(1)), "sedan_idle_s": float(m.group(2)), "sedan_vackning_s": float(m.group(3))}
    m = TIDER.search(rad)
    if m:
        delar = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", m.group(2)))
        return {**h, "typ": "tider", "enhet": m.group(1), "steg": delar, "utan_ljud": "utan_ljud" in m.group(2)}
    if "rost-prov: the recorded voice starts" in rad:
        return {**h, "typ": "rostprov"}
    if "interrupted: his reply is cut" in rad:
        return {**h, "typ": "klipp"}
    if "interrupted by the engine" in rad:
        return {**h, "typ": "motorklipp"}
    if "over his reply: loud over a working canceller" in rad:
        return {**h, "typ": "nivaklipp"}
    if "over his reply:" in rad:
        m = HEARD.search(rad)
        return {**h, "typ": "overhans", "hort": (m.group(1).strip("'") if m else None),
                "dom": "person" if "a person, interrupts" in rad else "ignorerad"}
    m = USER.search(rad)
    if m:
        return {**h, "typ": "user", "text": m.group(1)}
    m = VERKTYG.search(rad)
    if m and "Calling function" in rad:
        return {**h, "typ": "verktyg", "namn": m.group(1)}
    m = BANA0_HORT.search(rad)
    if m and "app.bana0" in rad:
        return {**h, "typ": "bana0_hort", "text": m.group(1)}
    if "question may have been cut" in rad:
        return {**h, "typ": "kan_ha_klippts"}
    if "Gemini turn detection:" in rad:
        return {**h, "typ": "turlage", "lage": "lokala regler (Silero 800 ms)" if "LOCAL" in rad else "Geminis egen turtagning"}
    if "Follow-up window" in rad:
        return {**h, "typ": "foljdfonster", "text": rad.split("Follow-up window:")[1].split(",")[0].strip()}
    return None


def svarsfonster(handelser: List[dict], enhet: str = "kontoret") -> List[dict]:
    """Varje svar = tiden från övergången till SPEAKING till övergången ut ur SPEAKING (journalens tillståndsmaskin)."""
    ut, start = [], None
    for h in handelser:
        if h["typ"] != "sess" or h["enhet"] != enhet:
            continue
        if h["till"] == "SPEAKING" and start is None:
            start = h["t"]
        elif h["fran"] == "SPEAKING" and start is not None:
            ut.append({"t0": start, "t1": h["t"], "slut_till": h["till"]})
            start = None
    return ut


def lage_vid(handelser: List[dict], t: float) -> str:
    """Turtagningsläget som gällde vid tiden t (senaste 'Gemini turn detection'-raden före)."""
    lage = "okänt"
    for h in handelser:
        if h["t"] > t:
            break
        if h["typ"] == "turlage":
            lage = h["lage"]
    return lage


def bygg_pass(handelser: List[dict], inspelningar: List[dict], gap_s: float = 20 * 60) -> List[dict]:
    """Pass = en följd av kontorets händelser och inspelningar utan lucka större än gap_s. Numrerade i tid.
    Varje pass får turtagningsläge, om enheten rapporterar underrun och om spärren 'over his reply' körde (barge-in på)."""
    tider = [h["t"] for h in handelser if (h["typ"] in ("sess", "tider") and h.get("enhet") == "kontoret")
             or h["typ"] in ("rostprov", "klipp", "overhans", "user")]
    tider += [r["utc"] for r in inspelningar if r.get("utc")]
    tider.sort()
    pass_, cur = [], None
    for t in tider:
        if cur is None or t - cur["t1"] > gap_s:
            cur = {"t0": t, "t1": t}
            pass_.append(cur)
        cur["t1"] = t
    for i, p in enumerate(pass_, 1):
        inne = [h for h in handelser if p["t0"] - 5 <= h["t"] <= p["t1"] + 5]
        p["id"] = f"P{i:02d}"
        p["lage"] = lage_vid(handelser, p["t0"] + 1)
        p["underrun_rapport"] = any(h["typ"] == "underrun" for h in inne)
        p["bargein_pa"] = any(h["typ"] in ("overhans", "klipp", "nivaklipp") for h in inne)
        p["etikett"] = f"{tid_text(p['t0'])}-{tid_text(p['t1'])[6:11]} UTC"
    return pass_


def pass_for(pass_: List[dict], t: float, slack_s: float = 30.0) -> Optional[dict]:
    for p in pass_:
        if p["t0"] - slack_s <= t <= p["t1"] + slack_s:
            return p
    return None


# --------------------------------------------------------------------------------------------------------- recordings

def las_wav(path: str):
    with wave.open(path) as w:
        return w.getnchannels(), w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def inspelning_ur_fil(path: str) -> Optional[dict]:
    """En AEC_SPARA-fil (stereo 16 kHz, vänster mikrofon, höger referens) till en manifestpost. None = inte en sådan."""
    m = AEC_NAMN.match(os.path.basename(path))
    if not m:
        return None
    try:
        kan, hz, a = las_wav(path)
    except Exception:
        return None
    if kan != 2 or hz != RATE:
        return None
    a = a.reshape(-1, 2)
    nz = np.nonzero(a[:, 1])[0]
    utc = calendar.timegm((int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8]),
                           int(m.group(2)[:2]), int(m.group(2)[2:4]), int(m.group(2)[4:6])))
    return {"namn": os.path.basename(path), "utc": float(utc), "dur_s": round(len(a) / RATE, 2),
            "ref_s": round((nz[-1] - nz[0]) / RATE, 2) if len(nz) else 0.0,
            "ref_start_s": round(nz[0] / RATE, 2) if len(nz) else None}


def burst_kluster(handelser: List[dict], gap_s: float = 1.5) -> List[List[dict]]:
    k, cur = [], []
    for h in (x for x in handelser if x["typ"] == "burst"):
        if cur and h["t"] - cur[-1]["t"] > gap_s:
            k.append(cur)
            cur = []
        cur.append(h)
    if cur:
        k.append(cur)
    return k


def justera(rec: dict, kluster: List[List[dict]], underruns: List[float]) -> dict:
    """Lägg en inspelning på journalens tidslinje: avspelningsskuren ('audio burst ... rx->play N ms') ger filens riktiga
    start. Filnamnet är spolningstiden (strax efter sista skuren). Poäng = hur väl varaktighet och namnets tid stämmer
    (lägre är bättre; > 1,5 = ingen säker träff, inga journalhändelser knyts då till filen)."""
    bast = None
    for c in kluster:
        dur = c[-1]["t"] - c[0]["t"]
        e = abs((c[-1]["t"] - rec["utc"]) + 1.6)
        p = abs(dur - rec["ref_s"]) + 0.3 * e
        if bast is None or p < bast[0]:
            bast = (p, c, dur)
    if bast is None or rec["ref_start_s"] is None:
        return {"poang": None}
    p, c, dur = bast
    start = c[0]["t"] - rec["ref_start_s"]
    slut = start + rec["dur_s"]
    return {"poang": round(p, 2), "start": start, "slut": slut, "spelstart_ms": c[0]["rx_till_play_ms"], "burstar": len(c),
            "underruns": [round(u - start, 2) for u in underruns if start - 0.5 <= u <= slut + 1]}


# --------------------------------------------------------------------------------------------------------- statistics

def median(x: List[float]) -> Optional[float]:
    x = sorted(x)
    if not x:
        return None
    n = len(x)
    return x[n // 2] if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2


def percentil(x: List[float], p: float) -> Optional[float]:
    x = sorted(x)
    if not x:
        return None
    return x[min(len(x) - 1, int(round(p / 100 * (len(x) - 1))))]


def tal(x: Optional[float], d: int = 2) -> str:
    return "-" if x is None else f"{x:.{d}f}"


def andel(a: int, b: int) -> str:
    return f"{a}/{b}"


def oberoende(rader: List[dict], nyckel: str = "inspelning", utfall: str = "klippt") -> Dict[str, int]:
    """Räkna oberoende fall: en inspelning är EN, hur många röster/volymer/tider den än kördes med.
    Ger {körningar, inspelningar, klippt_alla (alla körningar i inspelningen), klippt_nagon, klippt_runs}.
    Samma inspelning med fyra volymer räknas alltså som ett fall; körningarna visas som eget tal."""
    per: Dict[str, List[bool]] = {}
    for r in rader:
        per.setdefault(r[nyckel], []).append(bool(r[utfall]))
    return {"korningar": len(rader), "inspelningar": len(per),
            "klippt_korningar": sum(bool(r[utfall]) for r in rader),
            "klippt_alla": sum(all(v) for v in per.values()),
            "klippt_nagon": sum(any(v) for v in per.values())}
