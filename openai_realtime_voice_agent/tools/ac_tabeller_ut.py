"""US-060: ur manifest + journaler + körningar till tabeller.md/tabeller.json. Anropas av ac_tabeller.main."""
import json
import os
import sys

import ac_journal as J


def ladda(args):
    m = json.load(open(args.manifest))
    h = J.las_journal([x["fil"] for x in m["journaler"]])
    P = {p["id"]: p for p in m["pass"]}
    for r in m["inspelningar"]:
        r["pass_underrun"] = bool(P[r["pass"]]["underrun_rapport"]) if r["pass"] else False
    return m, h


def kor(args, T):
    os.makedirs(args.ut, exist_ok=True)
    m, h = ladda(args)
    jpi = T.journal_per_inspelning(m, h)
    korn, jobb = {}, []
    kf = os.path.join(args.ut, "korningar.json")
    if args.rendera_bara and os.path.exists(kf):
        korn = json.load(open(kf))
        jobb = T.bygg_jobb(m, jpi)
    elif not args.bara_journal:
        jobb = T.bygg_jobb(m, jpi)
        korn = T.kor_jobb(jobb, kf, args)
    args.jobb_totalt = len(jobb)
    args.jobb_klara = sum(1 for j in jobb if j["id"] in korn)
    try:
        import ac_tabeller_md as R
    except ImportError:
        print("inga tabeller än (ac_tabeller_md saknas)")
        return 0
    md, js = R.rendera(m, h, jpi, korn, args)
    open(os.path.join(args.ut, "tabeller.md"), "w").write(md)
    json.dump(js, open(os.path.join(args.ut, "tabeller.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print("skrev", os.path.join(args.ut, "tabeller.md"))
    return 0
