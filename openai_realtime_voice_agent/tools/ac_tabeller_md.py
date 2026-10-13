"""US-060: tabellerna. En funktion per acceptanspunkt; varje returnerar (markdown, json). Ren formatering och räkning, ingen körning.

Räknereglerna (se ac_tabeller.py): delning bara på sådant som är känt före utfallet, oberoende fall = inspelningar,
varje tal med källa. Inget här anropar något moln.
"""
import collections
import os
import re

import ac_journal as J
from ac_journal import andel, median, percentil, tal, tid_text

KALLA_INLAGD_LIVE = "inlagd röst live"
KALLA_RIKTIG_LIVE = "riktig röst live (eller okänt)"
KALLA_INLAGD_OFFLINE = "inlagd röst offline"
KALLA_KOD = "bara kod"
KALLA_NEJ = "går inte offline"


def tabell(kol, rader):
    if not rader:
        return "_(inga rader)_\n"
    s = "| " + " | ".join(kol) + " |\n|" + "|".join("---" for _ in kol) + "|\n"
    for r in rader:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def lokal(t):
    return tid_text(t, True)[6:14]


def passetikett(p):
    return f"{p['id']} {tid_text(p['t0'], True)[:11]}-{tid_text(p['t1'], True)[6:11]} lokal"


def _lage(p):
    return f"{p['lage']}; underrun-rapport {'ja' if p['underrun_rapport'] else 'nej'}; barge-in {'på' if p['bargein_pa'] else 'ej synligt'}"


def _p_of(pass_, t):
    p = J.pass_for(pass_, t)
    return p["id"] if p else "?"


# ------------------------------------------------------------------------------------------------------------ material

def material(m, h, jpi):
    rec = m["inspelningar"]
    pa = collections.Counter(r["pass"] for r in rec)
    just = collections.Counter(r["pass"] for r in rec if jpi[r["id"]]["justerad"])
    rader = []
    kl = collections.Counter(x["t"] for x in h if x["typ"] == "klipp")
    for p in m["pass"]:
        sw = [w for w in J.svarsfonster(h) if p["t0"] - 5 <= w["t0"] <= p["t1"] + 5]
        rp = sum(1 for x in h if x["typ"] == "rostprov" and p["t0"] - 5 <= x["t"] <= p["t1"] + 5)
        rader.append([passetikett(p), p["lage"], "ja" if p["underrun_rapport"] else "nej", "på" if p["bargein_pa"] else "ej synligt",
                      pa.get(p["id"], 0), just.get(p["id"], 0), len(sw), rp])
    ig = m["ignorerade"]["antal"]
    s = "## Material\n\n"
    s += (f"{len(rec)} unika AEC_SPARA-inspelningar (stereo 16 kHz: mikrofon rå + referens), {m['dubbletter']} dubbletter borttagna med md5, "
          f"{len(m['turljud'])} turljud (mono, 12/10, används inte i tabellerna), {len(m['allt3'])} allt-fil (3 kanaler, Gemini-läget). "
          f"Arbetsfiler från tidigare mätningar (härledda/syntetiska/delade kanaler, räknas inte som fall): {sum(v for k, v in ig.items() if k.startswith('arbetsfil'))}. "
          f"Journaler: {', '.join(os.path.basename(x['fil']) for x in m['journaler'])} (UTC; lokal tid = UTC + 2 h).\n\n")
    s += tabell(["pass (lokal tid)", "turtagningsläge (journalen)", "firmware rapporterar underrun", "barge-in", "inspelningar", "varav justerade mot journalen", "svar i journalen (kontoret)", "inlagda röster (rost-prov)"], rader)
    s += ("\nPass = händelser/inspelningar utan lucka större än 20 min. Läge och underrun-rapport är ur journalen, inte ur kod eller firmware (firmwareversionen loggas inte).\n"
          "Röstklipp: " + ", ".join(f"{x['namn']} ({x['kalla'].split(' ')[0]}, {x['dur_s']} s)" for x in m["roster"]) + ".\n")
    return s


# ----------------------------------------------------------------------------------------------------- hjälp för körningar

def hamta_ren(korn, pid, kall, ur):
    return (korn.get(f"ren|{pid}|{'kall' if kall else 'varm'}|{'ur' if ur else 'utan-ur'}") or {}).get("resultat")


def injektioner(m, korn):
    """Alla injektionsprov som platta rader: bas, pass, set, röst, t, gain, utfall, med/utan underrun."""
    rader = []
    byid = {r["id"]: r for r in m["inspelningar"]}
    for jid, v in korn.items():
        if not jid.startswith("inj|") or not isinstance(v.get("resultat"), dict) or "prov" not in v["resultat"]:
            continue
        bas = byid[v["resultat"]["bas"]]
        ur = jid.endswith("|ur")
        for p in v["resultat"]["prov"]:
            rader.append({"inspelning": bas["id"], "pass": bas["pass"], "set": p["set"], "rost": p["rost"], "t": p["t"], "gain": p["gain"],
                          "ur": ur, "pass_underrun": bas["pass_underrun"], **p["ut"], "ref_s": bas["ref_s"], "bas": bas})
    return rader


def forsta_i_pass(m):
    f = {}
    for r in m["inspelningar"]:
        f.setdefault(r["pass"], r["id"])
    return set(f.values())


def underrun_nara(r, t_rost, fonster=3.0):
    u = (r["bas"].get("just") or {}).get("underruns", [])
    return any(t_rost - fonster <= x <= t_rost for x in u)


def erle_hink(e):
    return "ej mätt" if e is None else ("<10 dB" if e < 10 else ("10-20 dB" if e < 20 else ">=20 dB"))


def rad_stat(rs):
    """Standardkolumner för en grupp injektionskörningar."""
    o = J.oberoende(rs)
    lat = [r["latens_s"] for r in rs if r["klippt"] and r["via"] == "nivå" and r["latens_s"] is not None]
    lat_t = [r["latens_med_stt_s"] for r in rs if r["klippt"] and r["via"] == "text" and r["latens_med_stt_s"] is not None]
    return [o["inspelningar"], o["korningar"], o["klippt_korningar"], andel(o["klippt_nagon"], o["inspelningar"]), andel(o["klippt_alla"], o["inspelningar"]),
            f"{tal(median(lat))}/{tal(max(lat) if lat else None)} (n={len(lat)})", f"{tal(median(lat_t))}/{tal(max(lat_t) if lat_t else None)} (n={len(lat_t)})",
            sum(r["fore"] for r in rs)], o


KOL_INJ = ["oberoende inspelningar", "körningar", "körningar med klipp", "inspelningar klippta av NÅGON röst", "…av ALLA röster",
           "nivåväg median/max s (n)", "textväg median/max s (n) *", "klipp FÖRE rösten (summa)"]


# ---------------------------------------------------------------------------------------------------------------- AC-6

def ac6(m, h, jpi, korn, inj, pass_):
    s = "## AC-6 Avbrott med vad som helst (högst 0,7 s från att rösten började)\n\n"
    # 6a live
    sw = J.svarsfonster(h)
    rp = [x for x in h if x["typ"] == "rostprov"]
    kl = [x for x in h if x["typ"] in ("klipp", "nivaklipp", "motorklipp")]
    rader, detalj = [], []
    for p in m["pass"]:
        r_p = [x for x in rp if p["t0"] - 5 <= x["t"] <= p["t1"] + 5]
        k_p = [x for x in h if x["typ"] == "klipp" and p["t0"] - 5 <= x["t"] <= p["t1"] + 5]
        lat, klippta = [], 0
        for x in r_p:
            c = next((k for k in h if k["typ"] in ("klipp", "nivaklipp") and x["t"] - 0.2 <= k["t"] <= x["t"] + 3.0), None)
            if c:
                klippta += 1
                lat.append(c["t"] - x["t"])
            detalj.append((x, c))
        utan = [k for k in k_p if not any(x["t"] - 0.2 <= k["t"] <= x["t"] + 3.0 for x in rp)]
        if r_p or k_p:
            rader.append([passetikett(p), len(r_p), klippta, f"{tal(median(lat))}/{tal(max(lat) if lat else None)}", len(k_p), len(utan)])
    s += "### 6a Live, ur journalen (rost-prov = röstklipp inlagt i den levande mikrofonen)\n\n"
    s += tabell(["pass", "inlagda röster (rost-prov)", "…klippt inom 3 s", "röst->klipp median/max s", "klipp totalt", "klipp UTAN inlagd röst före (riktig röst, rum eller eko: orsak okänd)"], rader)
    s += "\nKälla: inlagd röst live (kolumn 2-4), riktig röst live eller okänt (kolumn 6). Klipp = journalraden `interrupted: his reply is cut`; röst->klipp är journalens tid (±~50 ms), inte ljudets. Ingen rad säger VILKEN röst som lades in.\n\n"
    s += "<details><summary>Spårning: varje inlagd röst (fil:rad) och dess klipp</summary>\n\n"
    s += tabell(["rost-prov (UTC)", "fil:rad", "klipp", "latens s"], [[tid_text(x["t"])[6:], f"{x['fil']}:{x['radnr']}", (f"{c['fil']}:{c['radnr']}" if c else "inget klipp inom 3 s"), (tal(c["t"] - x["t"]) if c else "-")] for x, c in detalj])
    s += "\n</details>\n\n"
    # 6b offline
    rs = [r for r in inj if r["set"] == "ac6"]
    s += "### 6b Offline, inlagd röst i den RÅA mikrofonen före ekosläckaren (lokala regler, kod 65d87aa)\n\n"
    if not rs:
        return s + "_Inga injektionskörningar än._\n", {}
    s += (f"Röst inlagd 3 s in i ett svar; baser = inspelningar där journalen visar eko utan inlagd röst och utan klipp (känt före utfallet), svaret minst 5,5 s. "
          "Ekosläckaren värms av de två föregående inspelningarna i passet. `*` textvägens tid är AUDIO som behövs plus lokal WhisperKit på Macen, inte core:s bana 0.\n\n")
    kol = ["pass", "underrun-rapporter"] + KOL_INJ
    rader = []
    for p in m["pass"]:
        for ur in ((True, False) if p["underrun_rapport"] else (False,)):
            g = [r for r in rs if r["pass"] == p["id"] and r["ur"] == ur]
            if g:
                st, _ = rad_stat(g)
                rader.append([passetikett(p), ("matas in (som live)" if ur else ("matas inte in" if p["underrun_rapport"] else "firmware rapporterar ej")), *st])
    g = [r for r in rs if r["ur"] or not r["pass_underrun"]]
    st, o = rad_stat(g)
    rader.append(["**alla pass**, med underrun där firmware har dem", "", *st])
    g2 = [r for r in rs if not r["ur"]]
    st2, o2 = rad_stat(g2)
    rader.append(["**alla pass**, utan underrun-matning överallt", "", *st2])
    s += tabell(kol, rader)
    # delning på sådant som är känt före
    s += "\n**Delning på sådant som är känt FÖRE utfallet** (värdet utskrivet, alla pass, underrun matas in där firmware har dem):\n\n"
    fp = forsta_i_pass(m)
    def del_(namn, nyckel, vals):
        out = []
        for v in vals:
            gg = [r for r in g if nyckel(r) == v]
            if gg:
                st_, _ = rad_stat(gg)
                out.append([namn, v, *st_])
        return out
    r2 = []
    r2 += del_("mätt ERLE före rösten", lambda r: erle_hink(r["erle_fore_db"]), ["ej mätt", "<10 dB", "10-20 dB", ">=20 dB"])
    r2 += del_("underrun inom 3 s före rösten", lambda r: "ja" if underrun_nara(r, r["t"]) else "nej", ["ja", "nej"])
    r2 += del_("första inspelningen i passet", lambda r: "ja" if r["inspelning"] in fp else "nej", ["ja", "nej"])
    r2 += del_("röst", lambda r: "Henriks riktiga röst" if r["rost"] in ("vanlig1", "vanlig2", "vanlig3", "bestamd", "mening") else "syntetisk", ["Henriks riktiga röst", "syntetisk"])
    s += tabell(["delning", "värde"] + KOL_INJ, r2)
    erl = [r["erle_fore_db"] for r in g if r["erle_fore_db"] is not None]
    s += f"\nERLE före rösten över alla körningar: median {tal(median(erl), 1)} dB, min {tal(min(erl) if erl else None, 1)}, max {tal(max(erl) if erl else None, 1)} (n={len(erl)}; ej mätt = färre än 80 ramar).\n"
    # sen röst
    late = [r for r in inj if r["set"] == "ac6sen" and (r["ur"] or not r["pass_underrun"])]
    if late:
        st_, _ = rad_stat(late)
        s += "\nRöst 6 s in (bara svar >= 9 s; samma inspelningar, räknas inte som nya fall):\n\n" + tabell(KOL_INJ, [st_])
    # svaret gäller det som sades
    ok = [r for r in g if r["klippt"] and r.get("tur_text")]
    s += (f"\n**Svaret gäller det som sades.** {len(ok)} av {sum(1 for r in g if r['klippt'])} klippta körningar gav en tur till motorn; turtexten är lokal WhisperKit av turljudet "
          "(inklusive upp till 1 s förljud), inte Geminis. Exempel (röst -> turtext):\n\n")
    ex = collections.OrderedDict()
    for r in ok:
        ex.setdefault(r["rost"], r["tur_text"])
    s += tabell(["röst", "turtext (första träffen)"], list(ex.items()))
    s += "\n### 6c Gemini sköter turerna (GEMINI_EGEN_TUR)\n\nGår inte offline: Geminis egen aktivitetsavkänning avgör avbrottet i molnet; verktyget kan bara mäta vad som NÄR motorn (se avsnittet 'Gemini sköter turerna' sist). Inget i tabellen ovan säger något om vad Gemini skulle göra.\n"
    return s, {"ac6_alla": o, "ac6_utan_underrun": o2}


# ---------------------------------------------------------------------------------------------------------------- AC-7

def ac7(m, args):
    import numpy as np
    import ac_kor as K
    from app.providers.local_turns import LocalTurns, PRE_END_MS
    s = "## AC-7 Han väntar ut en tankepaus (paus upp till ~1,2 s mitt i en mening)\n\n"
    unika = [e for e in m.get("paus_klipp", []) if not e.get("dubblett_av")]
    if not unika:
        return s + "_Inga pausklipp i manifestet._\n", {}
    rng = np.random.default_rng(1)

    def kor(sig):
        tur = LocalTurns.create(800, pre_ms=PRE_END_MS)
        ev = []
        for i in range(0, len(sig) - 319, 320):
            e = tur.feed(sig[i:i + 320].tobytes())
            if e:
                ev.append((round(i / J.RATE + 0.02, 2), e))
        return ev
    rader = collections.defaultdict(list)
    for e in unika:
        x = np.frombuffer(open(e["path"], "rb").read(), dtype=np.int16)
        a, b = e["paus"]
        fore, efter = x[:int(a * J.RATE)], x[int(b * J.RATE):]
        for brus in (None, -55.0):
            for plen in (0.0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0):
                gap = (rng.normal(0, 10 ** (brus / 20) * 32768, int(plen * J.RATE)).astype(np.int16) if brus else np.zeros(int(plen * J.RATE), np.int16))
                pad = lambda sec: (rng.normal(0, 10 ** (brus / 20) * 32768, int(sec * J.RATE)).astype(np.int16) if brus else np.zeros(int(sec * J.RATE), np.int16))
                sig = np.concatenate([pad(1.0), fore, gap, efter, pad(2.5)])
                ev = kor(sig)
                t_a = 1.0 + len(fore) / J.RATE
                t_b = t_a + plen
                slut_i_pausen = any(en == "end" and t_a < t <= t_b + 0.05 for t, en in ev)
                rader[(brus, plen)].append({"inspelning": e["pcm"], "klippt": slut_i_pausen, "kalla": e["kalla"], "andra": ev})
    ut = []
    js = {}
    for (brus, plen), rs in sorted(rader.items(), key=lambda kv: (kv[0][0] is not None, kv[0][1])):
        o = J.oberoende(rs)
        ut.append([("inget" if brus is None else f"{brus:.0f} dBFS"), ("ingen paus (kontroll)" if plen == 0 else f"{plen:.1f} s"),
                   o["inspelningar"], f"{o['klippt_nagon']}/{o['inspelningar']}", sum(1 for r in rs if r["kalla"].startswith("henrik")), sum(r["klippt"] for r in rs if r["kalla"].startswith("henrik"))])
        js[f"{brus}|{plen}"] = o
    s += (f"{len(unika)} olika meningar (C-henrik1 och C-henrik2 är samma ljud, md5, och räknas en gång), 2 av dem Henriks riktiga röst, resten macOS-röst. Pausen tas ur klippet och sätts in igen med olika längd, "
          "med och utan brus. Kedjan är LocalTurns (Silero via sherpa-onnx, 800 ms tystnad), samma klass och parametrar som agenten. 'Turslut i pausen' = LocalTurns säger `end` medan pausen pågår, alltså innan meningen är slut. "
          "Pauslängd 0 är kontrollen: samma klipp utan paus ska aldrig ge turslut före meningens slut.\n\n")
    s += tabell(["bakgrundsbrus", "paus", "oberoende meningar", "turslut i pausen", "…varav Henriks riktiga: antal", "…turslut i pausen"], ut)
    s += "\nKälla: inlagd röst offline (lokala regler). Tröskeln går vid paus >= 0,9 s: på 1,2 s tar turen slut mitt i meningen. Ingen kod i repot håller en tur öppen för en halv mening (ingen sådan flagga finns att slå på).\n"
    # live: varningen 'en fråga kan ha klippts'
    kh = [x for x in J.las_journal([x["fil"] for x in m["journaler"]]) if x["typ"] == "kan_ha_klippts"]
    pc = collections.Counter(_p_of(m["pass"], x["t"]) for x in kh)
    s += "\n**Live (journalen):** raden `bana0: turn audio X s, the engine holds Y s - the engine has more, a question may have been cut` = turen som bana 0 fick var kortare än den motorn hade. Antal per pass: " + (", ".join(f"{k}: {v}" for k, v in sorted(pc.items())) or "inga") + f" (totalt {len(kh)}). Det är en VARNING om att en tur kan ha klippts, inte ett mått på tankepauser; första raden: {kh[0]['fil']}:{kh[0]['radnr']}.\n" if kh else ""
    s += "\n**Gemini sköter turerna:** går inte offline (Geminis egen tystnadsklocka avgör i molnet). Ett tyst Gemini-prov med riktig motor finns i Docs/research/us060-matning-2026-10-13/gemini men saknar kontroll utan paus.\n"
    return s, js


# ---------------------------------------------------------------------------------------------------------------- AC-8

def grep_aterupptagning():
    rot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ut = []
    pat = re.compile(r"ateruppta|återuppta|resume_reply|fortsatt_svar|ta upp tråden|ta_upp", re.I)
    for dp, _dn, fn in os.walk(os.path.join(rot, "app")):
        for f in fn:
            if f.endswith(".py"):
                for n, rad in enumerate(open(os.path.join(dp, f), errors="replace"), 1):
                    if pat.search(rad):
                        ut.append(f"{os.path.relpath(os.path.join(dp, f), rot)}:{n}")
    return ut


def ac8(m, h, jpi, korn, inj):
    s = "## AC-8 Medhåll stoppar honom inte (\"mm\", \"ja\", \"okej\" över hans tal)\n\n"
    rs = [r for r in inj if r["set"] == "ac8" and (r["ur"] or not r["pass_underrun"])]
    if not rs:
        return s + "_Inga injektionskörningar än._\n", {}
    rader = []
    for p in m["pass"]:
        g = [r for r in rs if r["pass"] == p["id"]]
        if not g:
            continue
        st, o = rad_stat(g)
        from app import eko_vakt
        klipp = [r for r in g if r["klippt"]]
        bara = sum(1 for r in klipp if r.get("tur_text") is None or eko_vakt.bara_avbrott(r.get("tur_text")))
        rader.append([passetikett(p), *st[:5], bara, len(klipp) - bara])
    st, o = rad_stat(rs)
    from app import eko_vakt
    klipp = [r for r in rs if r["klippt"]]
    bara = sum(1 for r in klipp if r.get("tur_text") is None or eko_vakt.bara_avbrott(r.get("tur_text")))
    rader.append(["**alla pass**", *st[:5], bara, len(klipp) - bara])
    s += ("Syntetiska medhåll (macOS Alva: mm, ja, okej, precis, aha; ingen riktig röst) inlagda 3 s in i ett svar, samma baser och regler som AC-6. 'Bara avbrott' = turen som klippet startade innehöll inga ord att svara på "
          "(eko_vakt.bara_avbrott): han tystnar och lyssnar. 'Tur till motorn' = det gick vidare som vanlig tur.\n\n")
    s += tabell(["pass", "oberoende inspelningar", "körningar", "körningar med klipp", "inspelningar klippta av NÅGOT medhåll", "…av ALLA medhåll", "klipp, bara avbrott", "klipp, tur till motorn"], rader)
    per = collections.defaultdict(list)
    for r in rs:
        per[r["rost"]].append(r)
    s += "\nPer medhåll:\n\n" + tabell(["medhåll", "körningar", "klippta"], [[k, len(v), sum(r["klippt"] for r in v)] for k, v in per.items()])
    g = grep_aterupptagning()
    s += (f"\n**Tar han upp tråden efter medhållet?** Bara kod: sökning i app/ efter återupptagning av ett avbrutet svar ({'inga träffar' if not g else ', '.join(g[:5])}). "
          "Det avbrutna svaret fortsätter inte; `VANTA_TYST_S` tystar resten av det avbrutna svaret. Hängning (att han blir tyst utan att svara) kräver motorn: går inte offline.\n"
          "Källa: inlagd röst offline (syntetisk, inte Henrik). Gemini sköter turerna: går inte offline.\n")
    return s, {"ac8": o}


# ---------------------------------------------------------------------------------------------------------------- AC-9

def ac9(m, h):
    from app import bana0
    s = "## AC-9 Följdfrågor utan väckord (upp till 10 s) och ett naturligt slut\n\n"
    fraser = ["Tack, det var allt.", "Tack det var allt", "tack det var allt!", "Tack.", "Nej tack", "Tack för det", "Det räcker", "Det var allt, tack", "Nej.", "Tack, det är bra", "Tack Björn"]
    s += "**Bara kod: `bana0.ar_nej` på avslutningsfraser** (lagat i 65d87aa: ordet 'tack, det var allt' med komma):\n\n"
    s += tabell(["fras", "ar_nej"], [[f"`{f}`", bana0.ar_nej(f)] for f in fraser])
    s += "\nFönstrets längd (10 s) är enhetens/addonets (`Follow-up window: 6s` i journalen), inte kodens; att samtalet 'stängs på tystnad' syns som `close: quiet for 30s`.\n\n"
    ev = [x for x in h if x["typ"] == "sess" and x["enhet"] == "kontoret"]
    tider = [x for x in h if x["typ"] == "tider" and x.get("enhet") == "kontoret" and not x["utan_ljud"]]
    rader = []
    for p in m["pass"]:
        ev_p = [x for x in ev if p["t0"] - 5 <= x["t"] <= p["t1"] + 5]
        if not ev_p:
            continue
        nej = sum(1 for x in ev_p if x["till"] == "CLOSING" and x["orsak"] == "nej")
        tyst = sum(1 for x in ev_p if x["till"] == "CLOSING" and "quiet" in x["orsak"])
        tur = [x for x in h if x["typ"] == "turstart" and p["t0"] - 5 <= x["t"] <= p["t1"] + 5]
        # följdfråga: tur som startar högst 10 s efter att passet gick idle OCH utan ny väckning (sedan_vackning > sedan_idle), inte en provrunda
        folj = []
        for x in tur:
            reason = next((e["orsak"] for e in reversed(ev) if e["t"] <= x["t"] and e["till"] == "LISTENING"), "")
            if 0 <= x["sedan_idle_s"] <= 10 and x["sedan_vackning_s"] > x["sedan_idle_s"] + 0.5 and not reason.endswith("-prov"):
                svar = next((t for t in tider if x["t"] <= t["t"] <= x["t"] + 40), None)
                folj.append((x, svar))
        rader.append([passetikett(p), nej, tyst, len(folj), sum(1 for _, sv in folj if sv)])
    s += tabell(["pass", "samtal stängda av 'nej/tack det var allt' (close: nej)", "stängda av tystnad (30 s)", "följdfrågor <= 10 s efter svaret utan ny väckning (turstart)", "…med hörbart svar inom 40 s (tider-rad)"], rader)
    s += ("\nKälla: journalen (`close: nej` = bana 0:s ar_nej slog till; följdfråga = `turn start level ... s since idle` <= 10 och 'since wake' större än 'since idle'). Det går inte att se i journalen att "
          "'inget mer sägs' efter avslutningen och ingen rad säger att följdfrågan i själva verket var en provrunda utöver `*-prov`-orsaken, som är bortfiltrerad. Tio planerade försök per punkt finns inte; siffrorna är vad passen råkade ge.\n")
    return s, {}


# --------------------------------------------------------------------------------------------------------------- AC-10

def ac10(m, h, jpi, korn, pass_):
    s = "## AC-10 Han svarar aldrig på sitt eget eko\n\n"
    sw = J.svarsfonster(h)
    ep = []   # eko-prov-fönster (listening: eko-prov ... idle: eko-prov)
    st = None
    for x in h:
        if x["typ"] == "sess" and x["enhet"] == "kontoret" and x["orsak"] == "eko-prov":
            if x["till"] == "LISTENING":
                st = x["t"]
            elif st is not None:
                ep.append({"t0": st, "t1": x["t"]})
                st = None
    rp = [x for x in h if x["typ"] == "rostprov"]
    kl = [x for x in h if x["typ"] == "klipp"]       # journalens definitiva rad; nivaklipp/motorklipp är samma händelse en rad tidigare
    rader, utan_lista = [], []
    for p in m["pass"]:
        fon = [w for w in sw + ep if p["t0"] - 5 <= w["t0"] <= p["t1"] + 5]
        if not fon:
            continue
        mad_roster, utan = 0, 0
        for w in fon:
            if any(w["t0"] - 1 <= x["t"] <= w["t1"] + 3 for x in rp):
                mad_roster += 1
                continue
            ks = [k for k in kl if w["t0"] <= k["t"] <= w["t1"] + 1]
            ign = sum(1 for x in h if x["typ"] == "overhans" and x["dom"] == "ignorerad" and w["t0"] <= x["t"] <= w["t1"])
            if ks:
                utan += 1
                for k in ks:
                    utan_lista.append([passetikett(p), lokal(k["t"]), f"{k['fil']}:{k['radnr']}"])
        ren = len(fon) - mad_roster
        ign = sum(1 for x in h if x["typ"] == "overhans" and x["dom"] == "ignorerad" and p["t0"] - 5 <= x["t"] <= p["t1"] + 5)
        user = sum(1 for x in h if x["typ"] == "user" and any(w["t0"] <= x["t"] <= w["t1"] + 3 for w in fon) and not any(r["t0"] <= x["t"] <= r["t1"] + 3 for r in ep))
        rader.append([passetikett(p), len(fon), mad_roster, ren, utan, ign, user])
    s += "### 10a Live, ur journalen: alla svar (SPEAKING-fönster och eko-prov-fönster)\n\n"
    s += tabell(["pass", "svar/eko-fönster", "…med inlagd röst (rost-prov) i fönstret: räknas inte", "…utan inlagd röst", "…av dem med klipp (självavbrott, eller riktig person/rum: orsak okänd)", "'over his reply ... ignored' (nätet höll)", "användartexter i svar-fönster (🗣️ user:)"], rader)
    s += "\nKälla: journalen. Ett klipp i ett svar utan inlagd röst kan vara eko, en person i rummet eller Henrik som provade; journalen skiljer inte. Klippen utan inlagd röst, spårbara:\n\n"
    s += tabell(["pass", "klipp (lokal tid)", "fil:rad"], utan_lista)
    # 10b offline
    s += "\n### 10b Offline: ALLA inspelningar genom kedjan utan inlagd röst\n\n"
    rader, detalj = [], []
    jid = collections.defaultdict(dict)
    for p in m["pass"]:
        recs = [r for r in m["inspelningar"] if r["pass"] == p["id"]]
        if not recs:
            continue
        for kall in (False, True):
            for ur in ((True, False) if p["underrun_rapport"] else (False,)):
                res = hamta_ren(korn, p["id"], kall, ur)
                if not res or isinstance(res, dict):
                    continue
                for cls in ("bara eko enligt journalen", "journalen har inlagd röst eller klipp", "ej justerad mot journalen"):
                    g = [x for x in res if (("ej justerad" if not jpi[x["id"]]["justerad"] else ("bara eko" if jpi[x["id"]]["bara_eko_enligt_journal"] else "inlagd")) == {"bara eko enligt journalen": "bara eko", "journalen har inlagd röst eller klipp": "inlagd", "ej justerad mot journalen": "ej justerad"}[cls])]
                    if not g:
                        continue
                    klipp = [x for x in g if x["klipp"]]
                    hans = sum(1 for x in g for t in x["turer"] if t["hans_ord"])
                    tur = sum(len(x["turer"]) for x in g)
                    rader.append([passetikett(p), "kall" if kall else "varm", ("matas in" if ur else ("matas inte in" if p["underrun_rapport"] else "ej rapporterad")), cls, len(g),
                                  f"{len(klipp)} ({sum(len(x['klipp']) for x in klipp)} klipp)", tur, hans, sum(x["kandidater"] for x in g), sum(x["ignorerade"] for x in g)])
                    if cls.startswith("bara eko") and not kall and (ur or not p["underrun_rapport"]):
                        for x in klipp:
                            for k in x["klipp"]:
                                detalj.append([x["id"], passetikett(p), f"{k['t']:.1f}", k["via"], repr(k["hort"]), f"{k['niva']:.1f}"])
    s += ("Varje inspelning är ett fall; ingen volym- eller mikrofonskalning (verklig högtalarvolym 0,2/0,4/0,6 går inte att mappa: den loggas inte, och mikrofonskalning är inte volym). "
          "'varm' = en delad ekosläckare per pass i tidsordning (som live), 'kall' = ny per inspelning (värsta fallet). 'hans ord' = turtext där de flesta orden finns i HANS eget svar enligt lokal avskrift av referenskanalen "
          "(det agenten skickade) — det är facit, och det är hans ord, inte Geminis avskrift. 'ej justerad' = inspelningen gick inte att lägga på journalen (okänt om röst var inlagd).\n\n")
    s += tabell(["pass", "ekosläckare", "underrun-rapporter", "journalens klassning (känd före utfallet)", "inspelningar", "inspelningar med klipp i replay", "turer till motorn", "…med hans ord", "kandidater", "…ignorerade"], rader)
    s += "\nKälla: inlagd röst offline betyder här att ingen röst lagts in; det är eko genom kedjan (bara kod+inspelning). Replay-klipp i inspelningar som journalen kallar 'bara eko' (varm, underrun som live), spårbara:\n\n"
    s += tabell(["inspelning (aec_ÅÅÅÅMMDD_TTMMSS)", "pass", "klipp vid s", "väg", "hörde", "nivå dBFS"], detalj)
    s += "\nVolym 0,2/0,4/0,6 med tio frågor per volym är inte mätt: det kräver högtalaren. Gemini sköter turerna: se sista avsnittet (nivå och längd på det som når motorn).\n"
    return s, {}


# --------------------------------------------------------------------------------------------------------------- AC-11

def ac11(m, h):
    s = "## AC-11 Inget svar på prat i rummet utan väckord\n\n"
    ev = [x for x in h if x["typ"] == "sess" and x["enhet"] == "kontoret"]
    rader = []
    for p in m["pass"]:
        tur = [x for x in h if x["typ"] == "turstart" and p["t0"] - 5 <= x["t"] <= p["t1"] + 5]
        if not tur:
            continue
        b = collections.Counter()
        for x in tur:
            reason = next((e["orsak"] for e in reversed(ev) if e["t"] <= x["t"] and e["till"] == "LISTENING"), "")
            if reason.endswith("-prov"):
                b["provrunda (inlagd fråga)"] += 1
            elif x["sedan_vackning_s"] <= 10:
                b["<= 10 s efter väckning"] += 1
            elif x["sedan_vackning_s"] <= 60:
                b["10-60 s efter väckning"] += 1
            else:
                b["> 60 s efter väckning"] += 1
        rader.append([passetikett(p), len(tur), b["provrunda (inlagd fråga)"], b["<= 10 s efter väckning"], b["10-60 s efter väckning"], b["> 60 s efter väckning"]])
    s += tabell(["pass", "turstarter (`turn start level`)", "provrunda (inlagd fråga)", "<= 10 s efter väckning", "10-60 s efter väckning (öppet)", "> 60 s efter väckning (spärren kastar)"], rader)
    try:
        from app.websocket_handler import tur_utan_vackning, TUR_UTAN_VACKNING_S
        kastas = sum(1 for x in h if x["typ"] == "turstart" and tur_utan_vackning(x["sedan_idle_s"], x["sedan_vackning_s"], None, 1e9))
        s += f"\n**Bara kod:** `tur_utan_vackning` (grans {TUR_UTAN_VACKNING_S:.0f} s) skulle ha kastat {kastas} av {sum(1 for x in h if x['typ']=='turstart')} turstarter i journalen (alla pass, sedan_svar utelämnat).\n"
    except Exception as e:  # modulen går inte att ladda utan agentens miljö: redovisas, ingen gissning
        s += f"\n**Bara kod:** `tur_utan_vackning` kunde inte laddas här ({e!r}); gränsen står i app/websocket_handler.py (TUR_UTAN_VACKNING_S).\n"
    s += ("\nGår inte offline på riktigt: ingen av inspelningarna har rumsljud utan väckord (TV, samtal i rummet) att spela upp, och tolkningen av 10-60 s kräver att man vet vem som pratade. "
          "Tabellen visar bara hur gamla turstarterna var; den visar inte att något i rummet INTE ledde till en tur.\n")
    return s, {}


# --------------------------------------------------------------------------------------------------------------- AC-12

def tursort(h, tider_h, ev, klipp):
    """Sortera en tur: provrunda > avbrott > verktyg > bana 0 > modellfråga. Ordningen är vald före utfallet."""
    t = tider_h["t"]
    reason = next((e["orsak"] for e in reversed(ev) if e["t"] <= t and e["till"] == "LISTENING"), "")
    steg = tider_h["steg"]
    if reason.endswith("-prov"):
        return "provrunda (inlagd fråga)"
    if any(t - 25 <= k["t"] <= t for k in klipp):
        return "avbrott (turen kom efter ett klipp)"
    if "verktyg" in steg:
        return "verktyg"
    if "modell" not in steg:
        return "bana 0 (lokalt svar)"
    return "modellfråga"


def ac12(m, h):
    s = "## AC-12 Första ordet inom 1,5 s (median, fråga till modellen)\n\n"
    ev = [x for x in h if x["typ"] == "sess" and x["enhet"] == "kontoret"]
    klipp = [x for x in h if x["typ"] in ("klipp", "nivaklipp")]
    tider = [x for x in h if x["typ"] == "tider" and x.get("enhet") == "kontoret"]
    ljud = [x for x in tider if "enhet" in x["steg"] and not x["utan_ljud"]]
    utan = [x for x in tider if x["utan_ljud"]]
    rader = []
    allt = collections.defaultdict(list)
    for x in ljud:
        p = _p_of(m["pass"], x["t"])
        allt[(p, tursort(h, x, ev, klipp))].append(x)
        allt[("alla", tursort(h, x, ev, klipp))].append(x)
        allt[(p, "alla sorter")].append(x)
        allt[("alla", "alla sorter")].append(x)
    pmap = {p["id"]: p for p in m["pass"]}
    ordning = ["modellfråga", "verktyg", "bana 0 (lokalt svar)", "avbrott (turen kom efter ett klipp)", "provrunda (inlagd fråga)", "alla sorter"]
    for (p, sort), xs in sorted(allt.items(), key=lambda kv: (kv[0][0] == "alla", kv[0][0], ordning.index(kv[0][1]))):
        if p != "alla" and sort not in ("modellfråga", "alla sorter"):
            continue                    # per pass visas bara modellfrågorna (AC-12:s rad) och summan; sorterna visas över alla pass
        e = [x["steg"]["enhet"] / 1000 for x in xs]
        fran_tur = [(x["steg"]["enhet"] - x["steg"]["turslut"]) / 1000 for x in xs]
        lage = pmap[p]["lage"] if p in pmap else "-"
        rader.append([("alla pass" if p == "alla" else passetikett(pmap[p])), lage if p != "alla" else "-", sort, len(xs), tal(median(e)), tal(percentil(e, 90)), tal(max(e)),
                      andel(sum(1 for v in e if v <= 1.5), len(e)), tal(median(fran_tur)), "; ".join(f"{x['fil']}:{x['radnr']}" for x in xs[:2]) + (" ..." if len(xs) > 2 else "")])
    s += ("**Definition.** Journalens `⏱ tider`-rad: `enhet` = ms från TALETS SLUT (turslut minus den 800 ms tystnad detektorn väntar, en uppskattning på några tiotal ms) till första ljudet vid enheten. "
          "Kolumnen 'från turslut' är samma tid räknad från det att turen avslutades (alltså 0,8 s kortare). Sortering, i denna ordning och vald före utfallet: provrunda (turen följde på en inlagd fråga: `listening: *-prov`), "
          "avbrott (ett klipp inom 25 s före), verktyg (`verktyg=` finns), bana 0 (inget `modell=`: lokalt svar), annars modellfråga. AC-12:s rad är 'modellfråga'. "
          f"{len(utan)} turer fick aldrig ljud (`utan_ljud`) och är inte med.\n\n")
    s += tabell(["pass", "läge", "sort", "turer", "median s", "p90 s", "max s", "<= 1,5 s", "median från turslut s", "källa (fil:rad, första två)"], rader)
    s += "\nKälla: riktig röst live eller inlagd fråga live beroende på sort (provrunda = inlagd fråga, digital); inga tider är uppmätta i ljud. Gemini sköter turerna: se sista avsnittet — journalen har inga `tider`-rader för det läget.\n"
    return s, {}


# --------------------------------------------------------------------------------------------------------------- AC-13

def ac13(m, korn, inj):
    s = "## AC-13 Vanlig samtalsröst räcker på 1 m och 3 m\n\n"
    rs = [r for r in inj if r["set"] == "ac13" and (r["ur"] or not r["pass_underrun"])]
    if not rs:
        return s + "_Inga injektionskörningar än._\n", {}
    rader = []
    base0 = [r for r in inj if r["set"] == "ac6" and r["rost"] in ("vanlig1", "bestamd", "mening") and (r["ur"] or not r["pass_underrun"])]
    st, _ = rad_stat(base0)
    rader.append(["0 dB (kontroll, samma röster, AC-6-körningarna)", *st[:5]])
    for g in sorted({r["gain"] for r in rs}, reverse=True):
        gg = [r for r in rs if r["gain"] == g]
        st, _ = rad_stat(gg)
        rader.append([f"{g:+.1f} dB", *st[:5]])
    s += ("Henriks röst (vanlig1, bestamd, mening) dämpad och inlagd 3 s in i ett svar, samma baser och regler som AC-6. **Kopplingen dB-avstånd är ett antagande**: fritt fält ger 6 dB per avståndsfördubbling, "
          "alltså ungefär -9,5 dB vid 3 m jämfört med 1 m; rummet, klangen och talarens riktning är inte mätta. -6 dB motsvarar ~2 m, -14/-20 dB ~5/10 m (om antagandet gäller).\n\n")
    s += tabell(["dämpning", "oberoende inspelningar", "körningar", "körningar med klipp", "inspelningar klippta av NÅGON röst", "…av ALLA röster"], rader)
    s += ("\nKälla: inlagd röst offline. Det som INTE är mätt: att väckordet hörs på 1 m och 3 m, mikrofonförstärkningen, rumsklang, att frågan uppfattas av Gemini. Dessa kräver en människa i rummet: går inte offline.\n")
    return s, {}


# --------------------------------------------------------------------------------------------------------------- AC-14

def ac14(m, h, korn, inj, args):
    from app import eko_vakt
    s = "## AC-14 Husstyrning mitt i samtalet, och aldrig av eko eller brus\n\n"
    # (b) läckaget ur replay: turer med hans ord
    turer = []
    for p in m["pass"]:
        res = hamta_ren(korn, p["id"], False, p["underrun_rapport"])
        if res and isinstance(res, list):
            for x in res:
                for t in x["turer"]:
                    turer.append((x["id"], t, x.get("hans") or ""))
    n_hans = sum(1 for _i, t, _h in turer if t["hans_ord"])
    vetoade = sum(1 for _i, t, hn in turer if t["hans_ord"] and eko_vakt.veto_tur("replying", 0.0, 1.0, t["text"], hn))
    s += ("**(b) Eko till verktyg, bara kod + replay.** Turer som replay skickar till motorn (alla inspelningar, varm ekosläckare, underrun som live): "
          f"{len(turer)} st ({sum(1 for _i, t, _h in turer if t['efter_avbrott'])} efter ett replay-klipp), {n_hans} av dem hans egna ord (liknar_svar mot lokal avskrift av referenskanalen). `veto_tur` (spärren före modellen) stoppar {vetoade} av {n_hans}. "
          "Replayen kör inte verktygen och inte motorn: 'noll verktygsanrop' är bara belagt som att spärren skulle stoppa turen, inte som att inget anrop skedde.\n\n")
    byid = {r["id"]: r for r in m["inspelningar"]}
    # (a) order efter klipp: veto_verktyg
    bases = [r for r in m["inspelningar"] if r["ref_s"] >= BAS_REF]
    roster = {x["namn"]: x for x in m["roster"]}
    rader = []
    import ac_kor as K
    stt = K.Stt(args.stt_cache or os.path.join(args.ut, "stt_cache.json"), args.sttd, os.path.join(args.ut, "tmp"))
    import asyncio
    hans_texter = {}
    for r in bases[:12]:
        txt, ok = asyncio.run(stt(open(r["path"], "rb").read() and K.las_stereo(r["path"])[1].tobytes()))
        if ok:
            hans_texter[r["id"]] = txt
    for n in ("vanlig1", "vanlig2", "vanlig3", "bestamd", "mening", "hjalp", "tand", "klockan"):
        if n not in roster:
            continue
        x = K.las_klipp(roster[n])
        txt, ok = asyncio.run(stt(x.tobytes()))
        pass_n, kast = 0, 0
        for bid, hans in hans_texter.items():
            v = eko_vakt.veto_verktyg("replying", 0.0, 1.0, txt if ok else "", hans)
            pass_n += v is None
            kast += v is not None
        rader.append([n, roster[n]["kalla"].split(" ")[0], repr(txt) if ok else "(ingen avskrift: sttd/cache saknas)", len(hans_texter), pass_n, kast])
    stt.stang()
    s += ("**(a) Order mitt i samtalet, bara kod.** Klippets lokala avskrift körs genom `veto_verktyg` 1 s efter ett klipp (spärren är öppen) mot hans svar i 12 olika inspelningar. 'passerar' = ingen spärr, alltså ordern får utföras. "
          "En order som lokal STT missar helt ('') stoppas av spärren (okänt tal direkt efter ett klipp). Källa: bara kod + lokal avskrift; Home Assistant körs aldrig.\n\n")
    s += tabell(["klipp", "röst", "lokal avskrift", "olika svar att jämföra med", "passerar", "stoppas"], rader)
    # live
    vk = [x for x in h if x["typ"] == "verktyg"]
    users = [x for x in h if x["typ"] == "user"]
    rr = []
    for p in m["pass"]:
        v_p = [x for x in vk if p["t0"] - 5 <= x["t"] <= p["t1"] + 5]
        if not v_p:
            continue
        utan_user = [x for x in v_p if not any(0 <= x["t"] - u["t"] <= 30 for u in users)]
        rr.append([passetikett(p), len(v_p), collections.Counter(x["namn"] for x in v_p).most_common(2), len(utan_user),
                   ", ".join(f"{x['namn']} ({x['fil']}:{x['radnr']})" for x in utan_user[:3])])
    s += "\n**Live (journalen): verktygsanrop (`Calling function`) och om en användartext (🗣️ user:) föregick dem inom 30 s.**\n\n"
    s += tabell(["pass", "verktygsanrop", "vanligaste", "utan användartext inom 30 s före (ska vara 0)"], rr)
    s += "\nHusstyrning som faktiskt utförs i huset kräver Home Assistant och är inte mätt. Gemini sköter turerna: verktygsspärren är kvar i koden men inte mätt här.\n"
    return s, {}


BAS_REF = 5.5


# --------------------------------------------------------------------------------------------------------------- AC-15

def ac15(m, h, jpi, pass_):
    s = "## AC-15 Inga dubbla röster efter ett avbrott\n\n"
    sw = J.svarsfonster(h)
    byrec = [r for r in m["inspelningar"] if jpi[r["id"]]["justerad"]]
    rader, utan_insp = [], 0
    for k in (x for x in h if x["typ"] == "klipp"):
        r = next((r for r in byrec if r["just"]["start"] <= k["t"] <= r["just"]["slut"]), None)
        if r is None:
            utan_insp += 1
            continue
        nasta = next((w for w in sw if w["t0"] > k["t"]), None)
        x = k["t"] - r["just"]["start"]
        tail = (r["ref_start_s"] or 0) + r["ref_s"] - x
        d = (nasta["t0"] - k["t"]) if nasta else None
        rader.append({"pass": _p_of(m["pass"], k["t"]), "t": k["t"], "rad": f"{k['fil']}:{k['radnr']}", "insp": r["id"], "tail": tail,
                      "nasta": d if d is not None and d <= 30 else None})
    tails = [r["tail"] for r in rader]
    nxt = [r["nasta"] for r in rader if r["nasta"] is not None]
    unika = len({r["insp"] for r in rader})
    s += ("Per live-klipp (journalens `interrupted: his reply is cut`) som går att lägga på en inspelning. 'Agenten skickade ljud efter klippet' = referenskanalens sista ljud minus klippets tid i filen (negativt = ljudet hade redan slutat). "
          "'Nästa svar börjar efter' = nästa SPEAKING i journalen, bara om det är inom 30 s (annars fanns inget direkt nytt svar). Referenskanalen är det AGENTEN skickade, inte det enheten spelade: att enhetens buffert töms (firmware på listening) syns inte i något material.\n\n")
    s += (f"**Sammanfattning:** {len(rader)} klipp i {unika} olika inspelningar (oberoende fall = inspelningar); {utan_insp} live-klipp har ingen justerad inspelning (främst 12/10) och är inte med. "
          f"Agenten skickade ljud i median {tal(median(tails))} s (max {tal(max(tails) if tails else None)}) efter klippet; ett direkt nytt svar (inom 30 s) började i {len(nxt)} fall, median {tal(median(nxt))} s efter klippet (min {tal(min(nxt) if nxt else None)}).\n\n")
    s += tabell(["pass", "klipp (lokal)", "fil:rad", "inspelning", "agent skickade ljud efter klippet s", "nästa svar börjar efter s"],
                [[r["pass"], lokal(r["t"]), r["rad"], r["insp"], tal(r["tail"]), tal(r["nasta"])] for r in rader])
    s += "\nKälla: riktig eller inlagd röst live (journalen skiljer inte; se AC-6a), inget offline-bevis. 'Inget av det gamla svaret spelas' kräver enhetens ljud: går inte offline.\n"
    return s, {"klipp": len(rader), "inspelningar": unika}


# ---------------------------------------------------------------------------------------------------- Gemini sköter turerna

def gemini_lage(m, h, args):
    """Det som NÄR motorn när Gemini sköter turerna: nivå och längd på det som släpps igenom under svar, med SOFT_AEC_NS/REST av och på."""
    import numpy as np
    import ac_kor as K
    s = "## Gemini sköter turerna (GEMINI_EGEN_TUR): vad som når motorn\n\n"
    allt = m.get("allt3") or []
    if not allt:
        return s + "_Ingen allt-fil (3 kanaler) i materialet._\n", {}
    a = np.frombuffer(open(allt[0]["path"], "rb").read(), dtype=np.int16).reshape(-1, 3)
    mic, ref, ut_live = a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy()
    FR = 160
    n = len(mic) // FR
    refakt = np.array([np.any(ref[i * FR:(i + 1) * FR]) for i in range(n)])
    # svar-segment: referens aktiv, sammanslagna över luckor < 1 s; fönstret utvidgas 1,5 s efter (rummet bär rösten)
    seg, i = [], 0
    while i < n:
        if refakt[i]:
            j = i
            while j < n and (refakt[j] or (j < n and refakt[j:j + 100].any())):
                j += 1
            seg.append((i, j))
            i = j
        else:
            i += 1
    seg = [(a_, b_) for a_, b_ in seg if (b_ - a_) >= 100]

    def db10(x):
        return float(20 * np.log10(max(float(np.sqrt(np.mean(x.astype(np.float32) ** 2))), 1.0) / 32768))

    def mat(kanal, namn):
        rader = []
        for (a_, b_) in seg:
            b2 = min(n, b_ + 150)
            niv = np.array([db10(kanal[i * FR:(i + 1) * FR]) for i in range(a_, b2)])
            skip = 200                       # de första 2 s separat: eko läcker i början av ett svar
            f2, e2 = niv[:skip], niv[skip:]
            run, mx = 0, 0
            for v in niv:
                run = run + 1 if v > -50 else 0
                mx = max(mx, run)
            rader.append([namn, f"{a_ * FR / J.RATE:.1f}-{b2 * FR / J.RATE:.1f} s", f"{float(niv.max()):.1f}", f"{float(f2.max()) if len(f2) else float('nan'):.1f}",
                          f"{float(e2.max()) if len(e2) else float('nan'):.1f}", f"{(niv > -50).sum() / 100:.2f}", f"{(niv > -40).sum() / 100:.2f}", f"{mx / 100:.2f}"])
        return rader

    rader = mat(mic, "mikrofon rå") + mat(ut_live, "till motorn, som det gick live")
    # omkörning med flaggor
    import asyncio
    from app.mjuk_aec import MjukAec

    def kor_aec(ns, rest):
        os.environ["SOFT_AEC"], os.environ["STADIG_TAKT"] = "1", "1"
        if ns is None:
            os.environ.pop("SOFT_AEC_NS", None)
        else:
            os.environ["SOFT_AEC_NS"] = str(ns)
        os.environ["SOFT_AEC_REST"] = str(rest)
        aec = MjukAec.skapa()
        aec.agc_max_db = 0.0     # i det här läget förstärks mikrofonen inte mellan svaren (MjukAec.for_enhet)
        ut = np.zeros_like(mic)
        STEG = 320
        for k in range(0, len(mic) - STEG + 1, STEG):
            t = 1000.0 + k / J.RATE
            r = ref[k:k + STEG]
            if np.any(r):
                aec.ref(r.tobytes(), 16000, t)
            o = aec.process(mic[k:k + STEG].tobytes(), t + STEG / J.RATE)
            if o:
                y = np.frombuffer(o, dtype=np.int16)
                ut[k + STEG - len(y):k + STEG] = y if False else y[:STEG]
        return ut
    for ns, rest in ((None, 0), (3, 0), (None, 2.0), (3, 2.0)):
        y = kor_aec(ns, rest)
        rader += mat(y, f"omkörd: NS={'av' if ns is None else ns}, REST={rest}")
    js = {"segment": len(seg)}
    s += (f"Underlaget är en enda inspelning ({os.path.basename(allt[0]['path'])}, {allt[0]['dur_s']} s, mikrofon/referens/till motorn). Den innehåller {len(seg)} svar-segment (referenskanalen aktiv, +1,5 s). Verktyget vet inte om en person pratade i rummet under dem; "
          "journalen (pass som ovan) har inga rost-prov i det läget. 'Omkörd' = samma mikrofon+referens genom MjukAec igen med SOFT_AEC_NS och SOFT_AEC_REST som anges "
          "(kanal 3 live = vad som faktiskt gick till motorn med de flaggor som stod på då, okända i filen). Per svar-segment: högsta nivå (100 ms-RMS, dBFS), högsta nivå de första 2 s respektive efter, sekunder över -50 och -40 dBFS, längsta sammanhängande tid över -50 dBFS. "
          "Det säger hur mycket av hans eget ljud som släpps igenom; det säger ingenting om vad Gemini gör med det.\n\n")
    s += tabell(["kanal", "segment (s)", "högsta dBFS", "…första 2 s", "…efter 2 s", "s över -50", "s över -40", "längsta följd över -50 s"], rader)
    return s, js


# ------------------------------------------------------------------------------------------------------------------ rendera

def rendera(m, h, jpi, korn, args):
    pass_ = m["pass"]
    inj = injektioner(m, korn)
    js = {}
    s = "# US-060 AC-6 till AC-15: tabeller, offline, 12-13/10 2026\n\n"
    s += ("Genererad av `tools/ac_tabeller.py`. Inget moln, inget ljud ut, ingen tjänst rörd. Ljudkörningarna är agentens egen kedja (tools/eko_replay.kor_fil: MjukAec -> LocalTurns -> eko_vakt -> nivåklippet) mot inspelningarna; "
          "tal-till-text är lokal WhisperKit (inte bana 0 på core), det påverkar textvägen men inte nivåvägen. Källa-kolumnen: *inlagd röst live* (rost-prov i den levande mikrofonen), *riktig röst live (eller okänt)*, *inlagd röst offline* (digitalt i den råa mikrofonkanalen före ekosläckaren), *bara kod*, *går inte offline*.\n\n"
          "**Vad verktyget inte kan mäta:** en riktig röst i rummet (väckord, avstånd, rumsklang, förstärkning), enhetens ljud (att bufferten töms), Geminis egna beslut, verkliga volymsteg, Home Assistant, vad motorn gör med ljudet, rumsprat utan väckord.\n\n")
    tot, klara = getattr(args, "jobb_totalt", 0), getattr(args, "jobb_klara", 0)
    if tot:
        s += (f"**Körningar: {klara} av {tot} ljudjobb är färdiga** ({'alla' if klara == tot else 'resten kördes inte i denna omgång: kör samma kommando igen, färdiga jobb hoppas över'}). "
              "Tabellerna redovisar bara det som kördes; antalet baser står i varje rad.\n\n")
    s += material(m, h, jpi) + "\n"
    for namn, f in (("ac6", lambda: ac6(m, h, jpi, korn, inj, pass_)), ("ac7", lambda: ac7(m, args)), ("ac8", lambda: ac8(m, h, jpi, korn, inj)), ("ac9", lambda: ac9(m, h)),
                    ("ac10", lambda: ac10(m, h, jpi, korn, pass_)), ("ac11", lambda: ac11(m, h)), ("ac12", lambda: ac12(m, h)), ("ac13", lambda: ac13(m, korn, inj)),
                    ("ac14", lambda: ac14(m, h, korn, inj, args)), ("ac15", lambda: ac15(m, h, jpi, pass_)), ("gemini", lambda: gemini_lage(m, h, args))):
        try:
            t, j = f()
        except Exception as e:      # en tabell som kraschar får inte ta de andra: felet står i tabellen
            import traceback
            t, j = f"## {namn}\n\n**Tabellen kunde inte byggas:** {e!r}\n\n```\n{traceback.format_exc()[-800:]}\n```\n", {"fel": repr(e)}
        s += "\n" + t
        js[namn] = j
    return s, js
