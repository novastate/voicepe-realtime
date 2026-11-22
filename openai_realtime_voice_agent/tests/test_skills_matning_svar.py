"""raawr US-907: tools/skills_matning.py measures how long Björn's answer was, from `💬 Björn sa:` lines only."""
import importlib.util
import json
import pathlib

_TOOLS = pathlib.Path(__file__).parent.parent / "tools"
_spec = importlib.util.spec_from_file_location("skills_matning", _TOOLS / "skills_matning.py")
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)

PRE = "2026-11-20T19:00:00.000+0000 core env[1]: "


def _sa(text):
    return f"{PRE}💬 Björn sa: {text}"


def _asst(text):
    return f"{PRE}🤖 assistant: {text}"


def _ord(n, start=0):
    return " ".join(f"ord{i}" for i in range(start, start + n))


def _tur(rader, fraga, verktyg=("fraga_core",), ljud=True):
    return sm.bedom(fraga, list(verktyg), rader, ljud)


# Frozen copy of `ratt` as it was before US-907 (AC-5 compares against it).
def _ratt_fore(fraga, verktyg, sagt):
    kallade = {v.split(":", 1)[-1] for v in verktyg if not v.startswith("las_skill")}
    vantat = set(fraga.get("vantat", []))
    verktyg_ok = bool(kallade & vantat) or ("-" in vantat and not kallade)
    if kallade & {"HassTurnOn", "HassTurnOff", "HassLightSet", "styr"} and not vantat & {"HassTurnOn", "HassTurnOff", "HassLightSet", "styr"}:
        return False
    o = [x.lower() for x in fraga.get("ord", [])]
    if not kallade and not o and not sagt.strip():
        return False
    return verktyg_ok and (not o or any(x in sagt.lower() for x in o))


def test_ac1_ord_och_meningar_en_gang_trots_assistantrad():
    text = f"{_ord(9)}. {_ord(9, 9)}."
    rader = [_sa(text), _asst(text)]
    m = sm.mat_tur(rader)
    assert m["ord"] == 18 and m["meningar"] == 2


def test_ac1_klockslag_raknas_inte_som_meningsslut():
    m = sm.mat_tur([_sa("Matchen börjar 19.00 i kväll.")])
    assert m["meningar"] == 1 and m["ord"] == 5


def test_ac2_for_langt_svar_fells_och_kort_gar():
    fraga = {"id": "x", "vantat": ["fraga_core"], "max_ord": 25}
    ratt, skal = _tur([_sa(_ord(40) + ".")], fraga)
    assert not ratt and any("för långt" in s for s in skal)
    ratt, skal = _tur([_sa(_ord(20) + ".")], fraga)
    assert ratt and skal == []


def test_ac3_upprakning_otalat_tecken_och_loptext():
    fraga = {"id": "x", "vantat": ["fraga_core"], "max_ord": 60}
    a = "Brynäs 18:00, Luleå 18:30, Färjestad 19:00, Leksand 19.00, Timrå 19:30 och Rögle 20:00."
    ratt, skal = _tur([_sa(a)], fraga)
    assert not ratt and any("uppräkning" in s for s in skal)
    ratt, skal = _tur([_sa("Frölunda @ Brynäs")], fraga)
    assert not ratt and any("otalat tecken" in s for s in skal)
    c = "Brynäs möter Luleå klockan sju, Färjestad möter Leksand klockan åtta och Timrå möter Rögle senare."
    ratt, skal = _tur([_sa(c)], fraga)
    assert ratt and skal == []


def test_ac4a_avhugget_svar_ar_for_langt_utan_max_ord():
    text = ("a" * 399) + "b"
    assert len(text) == 400
    ratt, skal = _tur([_sa(text)], {"id": "x", "vantat": ["fraga_core"]})
    assert not ratt and any("avhugget" in s for s in skal) and any("för långt" in s for s in skal)
    assert sm.mat_tur([_sa("a" * 399)])["avhugget"] is False


def test_ac4b_svarsljud_utan_bjorn_sa_ar_ej_matt():
    ratt, skal = _tur([_asst("Brynäs spelar i kväll.")], {"id": "x", "vantat": ["fraga_core"], "max_ord": 60})
    assert not ratt and "ej mätt" in skal
    # an empty `💬 Björn sa:` line is not a measurement either (zero words must never pass as an answer)
    ratt, skal = _tur([_sa("")], {"id": "x", "vantat": ["fraga_core"], "max_ord": 60})
    assert not ratt and "ej mätt" in skal
    # an interrupted answer is not measured either
    ratt, skal = _tur([f"{PRE}💬 Björn sa (avbruten, 3 s): Brynäs"], {"id": "x", "vantat": ["fraga_core"]})
    assert not ratt and "ej mätt" in skal


def test_ac5_gamla_fragor_doms_som_forut():
    fragor = [json.loads(r) for r in (_TOOLS / "skills_fragor.jsonl").read_text().splitlines() if r.strip()]
    assert len(fragor) == 8
    for f in fragor:
        for verktyg in (list(f["vantat"][:1]), [], ["HassTurnOn"], ["las_skill:x"]):
            for sagt in ("", "Det är 12 grader ute.", "Ja, det är tänt.", "Nej."):
                verktyg = [v for v in verktyg if v != "-"]
                if not sagt:
                    continue
                rader = [_sa(sagt), _asst(sagt)]
                ratt, skal = sm.bedom(f, verktyg, rader, True)
                assert ratt == _ratt_fore(f, verktyg, sagt), (f["id"], verktyg, sagt)


def test_ac6_sportfragorna_finns_och_ar_lasbara():
    rader = [json.loads(r) for r in (_TOOLS / "sport_fragor.jsonl").read_text().splitlines() if r.strip()]
    assert len(rader) >= 4
    forbjudet = ("lås", "larm", "garage", "tv", "vardagsrum")
    ids = set()
    for r in rader:
        assert {"id", "wav", "text", "vantat", "max_ord"} <= set(r)
        assert "fraga_core" in r["vantat"] and isinstance(r["max_ord"], int) and r["max_ord"] > 0
        assert r["id"] not in ids
        ids.add(r["id"])
        ord_i_fragan = r["text"].lower().replace("?", "").split()
        assert not any(f in ord_i_fragan for f in forbjudet), r["text"]
    assert any("Brynäs" in r["text"] for r in rader)
