#!/usr/bin/env python
"""US-060: en tabell per acceptanspunkt (AC-6 till AC-15), offline, mot alla inspelningar från 12-13/10. Inget moln, inget ljud ut.

    1. manifest över materialet (en gång, eller när något nytt kommit):
       python tools/ac_tabeller.py manifest --rot DIR [--rot DIR ...] --journal FIL [--journal FIL ...] --klipp KATALOG --ut manifest.json
    2. (en gång) syntetiska röstklipp med macOS-rösten Alva, skalade till Henriks vanliga nivå:
       python tools/ac_tabeller.py klipp --klipp KATALOG
    3. tabellerna (kan köras om; färdiga körningar ligger i <ut>/korningar.json och görs inte om utan --ny):
       python tools/ac_tabeller.py --manifest manifest.json --ut KATALOG [--sttd SOKVAG] [--stt-cache FIL] [--jobb 4] [--ny]
       -> KATALOG/tabeller.md, tabeller.json, korningar.json

Läs `tabeller.md` uppifrån: första avsnittet säger vad materialet är och vad verktyget INTE kan mäta.
Regler som gäller alla tabeller (kritiken av förra mätningen, 2026-10-13):
  - Delning av rader sker bara på sådant som är känt FÖRE utfallet (passets turtagningsläge, om firmware rapporterar underrun,
    mätt ERLE före rösten, första svaret i passet, underrun inom 3 s före rösten, journalens "inlagd röst"), med värdet utskrivet.
  - Oberoende fall = inspelningar. Körningar (inspelning x röst x tid x nivå) visas som eget tal bredvid.
  - Underrun-rapporter ur journalen matas in eller inte: båda utfallen står där det spelar roll.
  - Varje rad har en källa-kolumn och går att spåra: journalrader med fil:rad, replayrader med inspelningens id.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import ac_journal as J  # noqa: E402

HENRIK = ["vanlig1", "vanlig2", "vanlig3", "bestamd", "mening"]
ROST_AC6 = ["vanlig1", "vanlig3", "mening", "hjalp", "klockan"]
MEDHALL = ["mm", "ja", "okej", "precis", "aha"]
AC13_ROSTER = ["vanlig1", "bestamd", "mening"]
AC13_GAIN = [-6.0, -9.5, -14.0, -20.0]
T_ROST = 3.0                 # röstens början, s in i inspelningen (inspelningens ref börjar vid ~0)
BAS_MIN_REF_S = 5.5          # kortare svar går inte att lägga en röst i efter 3 s
VARMA_N = 1                  # så många tidigare inspelningar i samma pass värmer ekosläckaren före ett prov


# ------------------------------------------------------------------------------------------------ klipp (macOS say)

def gor_klipp(kat: str) -> None:
    import subprocess
    import wave
    import numpy as np
    import ac_kor as K
    import base64
    niva = float(np.median([K.topp(np.frombuffer(base64.b64decode(open(f"{kat}/rost_{n}.b64").read()), dtype=np.int16))
                            for n in ("vanlig1", "vanlig2", "vanlig3")]))
    for n, t in (("nej", "Nej."), ("mm", "Mm."), ("ja", "Ja."), ("okej", "Okej."), ("precis", "Precis."), ("aha", "Aha.")):
        w = f"{kat}/_say_{n}"
        subprocess.run(["say", "-v", "Alva", "-o", w + ".aiff", t], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", w + ".aiff", w + ".wav"], check=True)
        x = np.frombuffer(wave.open(w + ".wav").readframes(10 ** 8), dtype=np.int16)
        nz = np.nonzero(np.abs(x) > 60)[0]
        x = x[max(0, nz[0] - 320):nz[-1] + 320]
        y = np.clip(x.astype(np.float64) * 10 ** ((niva - K.topp(x)) / 20), -32768, 32767).astype(np.int16)
        open(f"{kat}/syn_{n}.pcm", "wb").write(y.tobytes())
        os.unlink(w + ".aiff")
        os.unlink(w + ".wav")
        print("syn", n, f"{len(y)/16000:.2f} s, topp {K.topp(y):.1f} dBFS (mål {niva:.1f})")


# ------------------------------------------------------------------------------------------- inspelningar + journal

def journal_per_inspelning(m: dict, h: list) -> dict:
    """Vad journalen säger om varje inspelning: händelser i filens fönster (bara om den är justerad mot journalen)."""
    ut = {}
    for r in m["inspelningar"]:
        j = r.get("just") or {}
        ok = j.get("poang") is not None and j["poang"] <= 1.5
        d = {"justerad": ok, "rostprov": [], "klipp": [], "person": [], "ignorerade": 0, "user": [], "tider": []}
        if ok:
            t0, t1 = j["start"] - 0.5, j["slut"] + 1.5
            for x in h:
                if x["t"] < t0 or x["t"] > t1:
                    continue
                if x["typ"] == "rostprov":
                    d["rostprov"].append(x)
                elif x["typ"] in ("klipp", "motorklipp"):
                    d["klipp"].append(x)
                elif x["typ"] == "nivaklipp":
                    d["person"].append(x)
                elif x["typ"] == "overhans":
                    (d["person"] if x["dom"] == "person" else d.__setitem__("ignorerade", d["ignorerade"] + 1))
                elif x["typ"] == "user":
                    d["user"].append(x)
                elif x["typ"] == "tider" and x.get("enhet") == "kontoret":
                    d["tider"].append(x)
        d["bara_eko_enligt_journal"] = ok and not d["rostprov"] and not d["klipp"] and not d["person"]
        ut[r["id"]] = d
    return ut


def valj_baser(m: dict, jpi: dict) -> list:
    """Baser för inlagd röst: bara det som är känt FÖRE utfallet. Journalen visar eko utan röst och utan klipp, svaret är minst
    BAS_MIN_REF_S långt, inspelningen går att lägga på journalens tidslinje."""
    return [r for r in m["inspelningar"] if jpi[r["id"]]["bara_eko_enligt_journal"] and r["ref_s"] >= BAS_MIN_REF_S]


def varma_for(m: dict, bas: dict, n: int = VARMA_N) -> list:
    """De n närmast föregående inspelningarna i samma pass (högst 15 min före), oavsett innehåll: ekosläckaren har hört dem live."""
    kand = [r for r in m["inspelningar"] if r["pass"] == bas["pass"] and r["utc"] < bas["utc"] and bas["utc"] - r["utc"] <= 900]
    return kand[-n:]


def bygg_jobb(m: dict, jpi: dict) -> list:
    """Jobben i den ordning de är viktigast om tiden tar slut: rena pass (varm), AC-6-prov, AC-8, AC-13, sist kall ekosläckare."""
    roster = {x["namn"]: x for x in m["roster"]}
    ren, kallt, inj = [], [], {"ac6": [], "ac8": [], "ac13": []}
    for p in m["pass"]:
        recs = [r for r in m["inspelningar"] if r["pass"] == p["id"]]
        if not recs:
            continue
        for med in ((True, False) if p["underrun_rapport"] else (False,)):
            ren.append({"typ": "pass_ren", "id": f"ren|{p['id']}|varm|{'ur' if med else 'utan-ur'}", "recs": recs, "kall": False, "med_underrun": med})
        kallt.append({"typ": "pass_ren", "id": f"ren|{p['id']}|kall|{'ur' if p['underrun_rapport'] else 'utan-ur'}", "recs": recs, "kall": True, "med_underrun": p["underrun_rapport"]})
    satser = {"ac6": [(n, T_ROST, 0.0) for n in ROST_AC6],
              "ac8": [(n, T_ROST, 0.0) for n in ("mm", "okej") if n in roster],
              "ac13": [("vanlig1", T_ROST, g) for g in (-9.5, -14.0, -20.0)]}
    for bas in valj_baser(m, jpi):
        varma = varma_for(m, bas)
        for med in ((True, False) if bas["pass_underrun"] else (False,)):
            for sat, lst in satser.items():
                prov = [{"set": sat, "rost": roster[n], "t": t, "gain": g} for n, t, g in lst]
                inj[sat].append({"typ": "injektion", "id": f"inj|{sat}|{bas['id']}|{'ur' if med else 'utan-ur'}", "bas": bas, "varma": varma, "prov": prov, "med_underrun": med})
    return ren + inj["ac6"] + inj["ac8"] + kallt + inj["ac13"]


def kor_jobb(jobb: list, ut_fil: str, args) -> dict:
    """Kör jobben i arbetsprocesser. Färdiga jobb (samma id) i ut_fil görs inte om utan --ny. Skriver ut_fil löpande."""
    import multiprocessing as mp
    import ac_kor as K
    res = json.load(open(ut_fil)) if os.path.exists(ut_fil) and not args.ny else {}
    kvar = [j for j in jobb if j["id"] not in res]
    print(f"{len(jobb)} jobb, {len(res)} färdiga i {ut_fil}, {len(kvar)} kvar", flush=True)
    if not kvar:
        return res
    tmp = os.path.join(args.ut, "tmp")
    os.makedirs(tmp, exist_ok=True)
    cache = args.stt_cache or os.path.join(args.ut, "stt_cache.json")
    import threading
    ctx = mp.get_context("spawn")
    req_q, resp_qs, index_q = ctx.Queue(), [ctx.Queue() for _ in range(args.jobb)], ctx.Queue()
    for i in range(args.jobb):
        index_q.put(i)
    stt = K.Stt(cache, args.sttd, tmp)
    server = threading.Thread(target=K.stt_server, args=(req_q, resp_qs, stt), daemon=True)
    server.start()
    with ctx.Pool(args.jobb, initializer=K.arbetare_start, initargs=(cache, tmp, req_q, resp_qs, index_q)) as pool:
        t_start = time.monotonic()
        for n, (jid, r, s_, saknas) in enumerate(pool.imap_unordered(K.arbetare_kor, kvar), 1):
            res[jid] = {"resultat": r, "s": s_, "stt_saknas": saknas}
            print(f"[{n}/{len(kvar)}] {jid} {s_} s, stt saknas {saknas}" + (f" FEL {r['fel']}" if isinstance(r, dict) and "fel" in r else ""), flush=True)
            if args.max_min and time.monotonic() - t_start > args.max_min * 60:
                print(f"--max-min {args.max_min} nådd: stoppar efter {n} jobb, resten körs inte", flush=True)
                pool.terminate()
                break
            if n % 3 == 0:
                json.dump(res, open(ut_fil, "w"), ensure_ascii=False)
                if stt.nya:
                    K.spara_cache(cache, dict(stt.nya))
                    stt.nya.clear()
    req_q.put(None)
    json.dump(res, open(ut_fil, "w"), ensure_ascii=False)
    if stt.nya:
        K.spara_cache(cache, dict(stt.nya))
    stt.stang()
    return res


# ------------------------------------------------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kommando")
    mp_ = sub.add_parser("manifest", help="bygg manifestet ur kataloger och journaler")
    mp_.add_argument("--rot", action="append", required=True)
    mp_.add_argument("--journal", action="append", required=True)
    mp_.add_argument("--klipp")
    mp_.add_argument("--ut", required=True)
    kp = sub.add_parser("klipp", help="gör de syntetiska röstklippen (macOS say)")
    kp.add_argument("--klipp", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--ut")
    ap.add_argument("--sttd", default=os.environ.get("AC_STTD"), help="sökväg till sttd (lokal WhisperKit); utan den används bara cachen")
    ap.add_argument("--stt-cache")
    ap.add_argument("--jobb", type=int, default=4)
    ap.add_argument("--ny", action="store_true", help="gör om alla körningar")
    ap.add_argument("--rendera-bara", action="store_true", help="kör inga nya jobb: rendera ur det som redan ligger i korningar.json")
    ap.add_argument("--max-min", type=float, default=0, help="stoppa jobben efter så här många minuter (0 = kör klart); resten redovisas som ej körda")
    ap.add_argument("--bara-journal", action="store_true", help="hoppa över ljudkörningarna (bara tabeller ur journalerna och koden)")
    a = ap.parse_args(argv)
    if a.kommando == "manifest":
        import ac_manifest as M
        m = M.bygg(a.rot, a.journal, a.klipp)
        json.dump(m, open(a.ut, "w"), ensure_ascii=False, indent=1, default=float)
        print(M.sammanfatta(m))
        return 0
    if a.kommando == "klipp":
        gor_klipp(a.klipp)
        return 0
    if not a.manifest or not a.ut:
        ap.error("--manifest och --ut krävs")
    import ac_tabeller_ut as T
    return T.kor(a, sys.modules[__name__])


if __name__ == "__main__":
    sys.exit(main())
