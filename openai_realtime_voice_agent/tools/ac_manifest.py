"""US-060: bygg ett manifest över allt material (inspelningar, journaler, röstklipp) för tools/ac_tabeller.py.

    python tools/ac_tabeller.py manifest --rot DIR [--rot DIR ...] --journal FIL [--journal FIL ...] --klipp KATALOG --ut manifest.json

Regler (alla skrivs ut i manifestet, inget tyst):
  - En AEC_SPARA-inspelning är en fil som heter aec_ÅÅÅÅMMDD_TTMMSS.wav, stereo 16 kHz. Allt annat som ligger i rotkatalogerna
    (aec_..._ren.wav, mic/aec-delningar, syntetiska prov) är arbetsfiler gjorda av tidigare mätningar: de räknas och listas
    som `ignorerade`, men blir inget fall.
  - Dubbletter avgörs med md5 av filens bytes. Första roten på kommandoraden vinner; alla kopior listas under `kopior`.
  - En inspelning läggs på journalens tidslinje med avspelningsskuren (tools/ac_journal.justera); poäng > 1,5 = ingen träff.
  - Pass = händelser och inspelningar utan lucka större än 20 min (tools/ac_journal.bygg_pass).
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ac_journal as J  # noqa: E402


def _filer(rot: str):
    for dp, dn, fn in os.walk(rot):
        dn.sort()
        for f in sorted(fn):
            yield os.path.join(dp, f)


def bygg(rotter, journaler, klipp_katalog=None) -> dict:
    handelser = J.las_journal(journaler)
    kluster = J.burst_kluster(handelser)
    underruns = [h["t"] for h in handelser if h["typ"] == "underrun"]
    sett = {}
    recs, turljud, allt3 = [], [], []
    ign = Counter()
    ign_ex = {}
    for rot in rotter:
        for p in _filer(rot):
            namn = os.path.basename(p)
            ext = os.path.splitext(namn)[1]
            if ext not in (".wav", ".pcm"):
                ign[f"annan filtyp {ext or '(ingen)'}"] += 1
                continue
            h = J.md5(p)
            rel = os.path.relpath(p, rot)
            if h in sett:
                sett[h]["kopior"].append(p)
                continue
            if J.AEC_NAMN.match(namn):
                r = J.inspelning_ur_fil(p)
                if r is None:
                    ign["aec-namn men inte stereo 16 kHz"] += 1
                    continue
                r.update({"md5": h, "path": p, "rot": os.path.dirname(rel) or os.path.basename(rot), "kopior": []})
                recs.append(r)
                sett[h] = r
            elif namn.startswith("tur_") and ext == ".wav":
                kan, hz, a = J.las_wav(p)
                x = {"namn": namn, "md5": h, "path": p, "dur_s": round(len(a) / kan / hz, 2), "kopior": [], "kanaler": kan}
                turljud.append(x)
                sett[h] = x
            elif namn.startswith("allt_") and ext == ".pcm":
                x = {"namn": namn, "md5": h, "path": p, "dur_s": round(os.path.getsize(p) / 6 / J.RATE, 1), "kopior": []}
                allt3.append(x)
                sett[h] = x
            else:
                k = "arbetsfil (härledd/syntetisk/delad kanal)" if ext == ".wav" else "arbetsfil pcm"
                ign[k] += 1
                sett[h] = {"kopior": []}
                ign_ex.setdefault(k, []).append(os.path.relpath(p, os.path.dirname(rot)))
    recs.sort(key=lambda r: r["utc"])
    for r in recs:
        r["id"] = f"{r['namn'][4:-4]}"
        r["just"] = J.justera(r, kluster, underruns)
    pass_ = J.bygg_pass(handelser, recs)
    for r in recs:
        p = J.pass_for(pass_, r["utc"])
        r["pass"] = p["id"] if p else None
    m = {"version": 1, "journaler": [{"fil": p, "md5": J.md5(p)} for p in journaler], "inspelningar": recs, "turljud": turljud, "allt3": allt3,
         "pass": pass_, "ignorerade": {"antal": dict(ign), "exempel": {k: v[:6] for k, v in ign_ex.items()}},
         "dubbletter": sum(len(r["kopior"]) for r in recs + turljud + allt3)}
    if klipp_katalog:
        m.update(klipp_ur_katalog(klipp_katalog))
    return m


def klipp_ur_katalog(kat: str) -> dict:
    """Röstklippen: rost_*.b64 (Henriks riktiga röst utklippt efter ekosläckaren, och syntetiska), syn_*.pcm (macOS-röst, gjorda
    av `ac_tabeller.py klipp`), C-*.pcm (meningar med en paus i, paus.json säger var). Identiska klipp (md5) räknas en gång."""
    import base64
    roster, paus = [], []
    sett = {}
    for f in sorted(os.listdir(kat)):
        p = os.path.join(kat, f)
        if f.startswith("rost_") and f.endswith(".b64"):
            x = np.frombuffer(base64.b64decode(open(p).read()), dtype=np.int16)
            namn = f[5:-4]
            kalla = "Henriks riktiga röst (utklippt ur live-fil)" if namn in ("vanlig1", "vanlig2", "vanlig3", "bestamd", "mening") else "syntetisk (macOS Alva)"
        elif f.startswith("syn_") and f.endswith(".pcm"):
            x = np.frombuffer(open(p, "rb").read(), dtype=np.int16)
            namn, kalla = f[4:-4], "syntetisk (macOS Alva)"
        else:
            continue
        roster.append({"namn": namn, "path": p, "kalla": kalla, "dur_s": round(len(x) / J.RATE, 2), "md5": J.md5(p)})
    pj = os.path.join(kat, "paus.json")
    if os.path.exists(pj):
        for e in json.load(open(pj)):
            p = os.path.join(kat, e["pcm"] + ".pcm")
            h = J.md5(p)
            e = {**e, "path": p, "md5": h, "dubblett_av": sett.get(h)}
            sett.setdefault(h, e["pcm"])
            paus.append(e)
    return {"roster": roster, "paus_klipp": paus}


def sammanfatta(m: dict) -> str:
    rec = m["inspelningar"]
    ju = [r for r in rec if r["just"].get("poang") is not None and r["just"]["poang"] <= 1.5]
    pa = Counter(r["pass"] for r in rec)
    rader = [f"{len(rec)} inspelningar (unika), {m['dubbletter']} dubbletter, {len(m['turljud'])} turljud, {len(m['allt3'])} allt-fil, "
             f"{len(ju)} justerade mot journalen (poäng <= 1,5), {len(m['pass'])} pass, ignorerade: {m['ignorerade']['antal']}"]
    for p in m["pass"]:
        rader.append(f"  {p['id']} {p['etikett']}  inspelningar={pa.get(p['id'], 0)}  {p['lage']}  underrun-rapport={p['underrun_rapport']}  barge-in={p['bargein_pa']}")
    return "\n".join(rader)
