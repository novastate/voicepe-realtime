"""raawr US-098: the same questions through today's tools (a), the two meta tools (b), and (c), measured alike.

Runs the satellite stand-in once per question and rep against one agent instance and reads that instance's
journal: time to first sound (tools/forstaljud.py's split), the tools the model called (for `anvand_verktyg`,
the comms tool it named), what Björn said, and whether that was right by the question's key
(`fragor.jsonl`: {"id", "wav", "vantat": [tool names, "-" = no tool is right too], "ord": [words, one must be said],
"max_ord": longest answer in words}). US-907: per turn the words and sentences Björn said (`💬 Björn sa:` lines only), and
a turn is wrong when the answer is over `max_ord`, a list of times, has a sign that is not speech (@ * # |), is cut at
400 characters, or was not measured (sound but no `💬 Björn sa:` line).

    python tools/skills_matning.py fragor.jsonl --vag b --url ws://127.0.0.1:18080/ --unit prov-skills-b \
        --reps 2 --ut b.jsonl

Run on core with /root/PROVPLATS set, next to tools/forstaljud.py and tools/satellit_attrapp.py. The stand-in's
cloud minutes count in the test ledger (MOLN_MAX_MINUTER_PROV_PER_DAG). Never a question that touches a lock, the
alarm, the garage, the TV or the living room; a light that is turned on is turned off by the next question.
"""
import argparse
import ast
import datetime as dt
import json
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forstaljud import PY, dela  # noqa: E402

ANROP = re.compile(r"Calling function \[([^:\]]+)[^\]]*\] with arguments (\{.*\})")
TID = re.compile(r"⏱ tool (\S+) (\d+) (\S+)")
SA = re.compile(r"(?:🤖 assistant|💬 Björn sa): (.*)")
# US-907: length and form are measured from this line only. `🤖 assistant:` repeats the same answer without a cap, and an
# interrupted answer (`💬 Björn sa (avbruten, ...):`) does not match.
BJORN_SA = re.compile(r"💬 Björn sa: (.*)")
KLOCKSLAG = re.compile(r"(?<![\d.:])(?:[01]?\d|2[0-3])[:.][0-5]\d(?![\d])")
OTALAT = "@*#|"
TAK = 400  # app/providers/gemini_live.py cuts the journal line at 400 characters


def anropade(rader: list[str]) -> list[str]:
    """Tool names in call order; `anvand_verktyg` is shown as the comms tool it ran (`anvand_verktyg:styr`)."""
    ut = []
    for r in rader:
        m = ANROP.search(r)
        if not m:
            continue
        namn = m.group(1).rsplit("__", 1)[-1]
        if namn == "anvand_verktyg":
            try:
                namn += ":" + str(ast.literal_eval(m.group(2)).get("namn", "?")).rsplit("__", 1)[-1]
            except (ValueError, SyntaxError):
                namn += ":?"
        ut.append(namn)
    return ut


def verktygstider(rader: list[str], anrop: list[str]) -> list[list]:
    """[name, ms, outcome] per finished tool, `anvand_verktyg` named by the comms tool it ran (in call order)."""
    inre = iter(a for a in anrop if a.startswith("anvand_verktyg"))
    ut = []
    for r in rader:
        if m := TID.search(r):
            namn = m.group(1).rsplit("__", 1)[-1]
            ut.append([next(inre, namn) if namn == "anvand_verktyg" else namn, int(m.group(2)), m.group(3)])
    return ut


# Same list as app/bana0.py SKRIV_VERKTYG (tools/ does not import app/).
SKRIVANDE = frozenset({"HassTurnOn", "HassTurnOff", "HassLightSet", "styr"})


def ratt(fraga: dict, verktyg: list[str], sagt: str) -> bool:
    """Right = a wanted tool was called (or none, when "-" is allowed), and one key word was said if any are set."""
    kallade = {v.split(":", 1)[-1] for v in verktyg if not v.startswith("las_skill")}
    vantat = set(fraga.get("vantat", []))
    verktyg_ok = bool(kallade & vantat) or ("-" in vantat and not kallade)
    if kallade & SKRIVANDE and not vantat & SKRIVANDE:  # a read question that also switched something is wrong (AC-6)
        return False
    ord = [o.lower() for o in fraga.get("ord", [])]
    if not kallade and not ord and not sagt.strip():  # silence is not an answer (G on US-098)
        return False
    return verktyg_ok and (not ord or any(o in sagt.lower() for o in ord))


def mat_tur(rader: list[str]) -> dict:
    """Length and form of what Björn said in one turn, from the `💬 Björn sa:` lines only (US-907)."""
    texter = [m.group(1).strip() for r in rader if (m := BJORN_SA.search(r))]
    text = " ".join(texter)
    meningar = [d for d in re.split(r"(?<=[.?!])(?:\s+|$)", text) if d.strip()]
    return {"matt": bool(texter), "ord": len(text.split()), "meningar": len(meningar),
            "upprakning": len(KLOCKSLAG.findall(text)) > 3, "otalat": [c for c in OTALAT if c in text],
            "avhugget": any(len(t) >= TAK for t in texter)}


def bedom(fraga: dict, verktyg: list[str], rader: list[str], svarsljud: bool = True) -> tuple[bool, list[str]]:
    """(right, reasons). Tool and key-word judgement is `ratt`, unchanged; length and form are added as reasons."""
    sagt = " ".join(m.group(1).strip() for r in rader if (m := SA.search(r)))
    m = mat_tur(rader)
    skal = []
    if not ratt(fraga, verktyg, sagt):
        skal.append("fel verktyg eller nyckelord")
    if svarsljud and not m["matt"]:
        skal.append("ej mätt")
    if m["avhugget"]:
        skal.append("avhugget")
    if m["avhugget"] or (fraga.get("max_ord") is not None and m["ord"] > fraga["max_ord"]):
        skal.append(f"för långt ({m['ord']} ord" + (f", gräns {fraga['max_ord']}" if fraga.get("max_ord") is not None else "") + ")")
    if m["upprakning"]:
        skal.append("uppräkning")
    if m["otalat"]:
        skal.append("otalat tecken (" + " ".join(m["otalat"]) + ")")
    return not skal, skal


def kor(fraga: dict, a) -> dict:
    t0 = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    attrapp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "satellit_attrapp.py")
    ut = subprocess.run([PY, attrapp, "--wav", fraga["wav"], "--url", a.url, "--device", "attrapp",
                         "--sekunder", str(a.sekunder)], capture_output=True, text=True).stdout
    h = {}
    for rad in ut.splitlines():
        if rad.startswith("{"):
            j = json.loads(rad)
            h.setdefault(j["vad"], j["t"])
    rader = subprocess.run(["journalctl", "-u", a.unit, "--no-pager", "-o", "short-iso-precise",
                            "--since", t0.strftime("%Y-%m-%d %H:%M:%S")], capture_output=True, text=True).stdout.splitlines()
    verktyg = anropade(rader)
    sagt = " ".join(m.group(1).strip() for r in rader if (m := SA.search(r)))
    # "anrop", not "verktyg": dela() has a "verktyg" of its own (tool seconds before the first sound)
    rad = {"vag": a.vag, "id": fraga["id"], "anrop": verktyg, "tider": verktygstider(rader, verktyg),
           "sagt": sagt[:400]}
    if "forsta_svarsljud" not in h:
        return {**rad, "fel": "inget svarsljud", "ratt": False}
    mat = mat_tur(rader)
    ok, skal = bedom(fraga, verktyg, rader, True)
    return {**rad, **dela(h, rader), "ord": mat["ord"], "meningar": mat["meningar"], "skal": skal, "ratt": ok}


def p95(v: list[float]) -> float:
    return sorted(v)[max(0, round(0.95 * len(v)) - 1)]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("fragor")
    p.add_argument("--vag", required=True)
    p.add_argument("--url", default="ws://127.0.0.1:8080/")
    p.add_argument("--unit", default="raawr-rostagent")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--sekunder", type=int, default=18)
    p.add_argument("--ut", required=True)
    a = p.parse_args()
    fragor = [json.loads(r) for r in open(a.fragor) if r.strip()]
    rader = []
    with open(a.ut, "a") as ut:
        for _ in range(a.reps):
            for f in fragor:
                r = kor(f, a)
                rader.append(r)
                ut.write(json.dumps(r, ensure_ascii=False) + "\n")
                ut.flush()
                print(json.dumps(r, ensure_ascii=False)[:300], flush=True)
    tider = [r["totalt"] for r in rader if r.get("totalt") is not None]
    per_verktyg: dict = {}
    for r in rader:
        for namn, ms, _ in r.get("tider", []):
            per_verktyg.setdefault(namn, []).append(ms)
    print(json.dumps({"vag": a.vag, "turer": len(rader), "ratt": sum(r["ratt"] for r in rader),
                      "forsta_ljud_median": statistics.median(tider) if tider else None,
                      "forsta_ljud_p95": p95(tider) if tider else None,
                      "verktyg_p95_ms": {n: p95(v) for n, v in sorted(per_verktyg.items())}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
