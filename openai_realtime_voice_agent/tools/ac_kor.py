"""US-060: körningarna bakom tools/ac_tabeller.py. Agentens egen kedja mot inspelningar, offline: ingen motor, ingen högtalare, inget moln.

Kedjan är tools/eko_replay.kor_fil (oförändrad förutom två valfria nycklar: underruns och erle_vid):
    mikrofon -> MjukAec (AEC3) -> LocalTurns (Silero, 800 ms) -> kandidat -> eko_vakt.dom -> nivåklippet MjukAec.person
Tal-till-text är en lokal WhisperKit (sttd) eller svaren som redan ligger i cachen (sha1 av ljudet). Aldrig bana 0 på core
och aldrig ett molnanrop: en saknad avskrift blir (None, False), räknas och redovisas.
"""
import asyncio
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

os.environ.setdefault("SOFT_AEC", "1")
os.environ.setdefault("STADIG_TAKT", "1")
RATE = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
STT_MODELL = os.environ.get("AC_STT_MODELL", "/Users/collin/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3-v20240930_turbo")


class Stt:
    """pcm16k -> (text, ok). Cache på sha1 av ljudet; en lokal sttd-process startas först när något saknas i cachen."""

    def __init__(self, cache_fil=None, sttd=None, tmp=None):
        self.cache_fil = cache_fil
        self.cache = json.load(open(cache_fil)) if cache_fil and os.path.exists(cache_fil) else {}
        self.nya = {}
        self.sttd = sttd if sttd and os.path.exists(sttd) else None
        self.tmp = tmp or tempfile.gettempdir()
        os.makedirs(self.tmp, exist_ok=True)
        self.p = None
        self.lock = None
        self.saknas = 0       # frågor som varken cachen eller sttd kunde svara på
        self.kallt = 0
        self.frågor = 0

    async def __call__(self, pcm: bytes):
        self.frågor += 1
        h = hashlib.sha1(pcm).hexdigest()
        if h in self.cache:
            return self.cache[h], True
        if self.sttd is None:
            self.saknas += 1
            return None, False
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:
            if self.p is None:
                self.p = subprocess.Popen([self.sttd, STT_MODELL, "gpu"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=open(os.path.join(self.tmp, "sttd.err"), "a"), text=True, bufsize=1)
            fd, path = tempfile.mkstemp(suffix=".wav", dir=self.tmp)
            os.close(fd)
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                pad = b"\0" * (2 * RATE)    # 1 s tystnad på varje sida: WhisperKit skriver inget för ett nakent 0,7 s-ord
                w.writeframes(pad + pcm + pad)
            self.p.stdin.write(path + "\n")
            self.p.stdin.flush()
            line = await asyncio.get_running_loop().run_in_executor(None, self.p.stdout.readline)
            os.unlink(path)
            _, _, text = line.rstrip("\n").partition("\t")
            if text.startswith("<<ERR") or not line:
                self.saknas += 1
                return None, False
            self.cache[h] = text
            self.nya[h] = text
            self.kallt += 1
            return text, True

    def stang(self):
        if self.p:
            try:
                self.p.stdin.close()
                self.p.wait(timeout=10)
            except Exception:
                self.p.kill()
            self.p = None


class SttProxy:
    """Samma gränssnitt som Stt, men frågan går till föräldraprocessens EN delade sttd (arbetsprocesserna delar en modell i minnet)."""

    def __init__(self, req_q, resp_q, cache):
        self.req_q, self.resp_q, self.cache = req_q, resp_q, cache
        self.nya, self.saknas, self.kallt, self.frågor = {}, 0, 0, 0

    async def __call__(self, pcm: bytes):
        self.frågor += 1
        h = hashlib.sha1(pcm).hexdigest()
        if h in self.cache:
            return self.cache[h], True
        self.req_q.put((h, pcm))
        h2, text, ok = await asyncio.get_running_loop().run_in_executor(None, self.resp_q.get)
        if not ok:
            self.saknas += 1
            return None, False
        self.cache[h] = text
        self.kallt += 1
        return text, True

    def stang(self):
        pass


def stt_server(req_q, resp_qs, stt):
    """Körs som tråd i föräldern: svarar på arbetsprocessernas frågor med en enda Stt (cache + sttd). Fråga = (arbetare, hash, pcm)."""
    while True:
        x = req_q.get()
        if x is None:
            return
        who, h, pcm = x
        text, ok = asyncio.run(stt(pcm))
        resp_qs[who].put((h, text, ok))


def spara_cache(cache_fil, nya):
    """Slå ihop nya avskrifter in i cachefilen (atomiskt)."""
    c = json.load(open(cache_fil)) if os.path.exists(cache_fil) else {}
    c.update(nya)
    tmp = cache_fil + ".tmp"
    json.dump(c, open(tmp, "w"), ensure_ascii=False)
    os.replace(tmp, cache_fil)


# ------------------------------------------------------------------------------------------------------------- ljud

def db(x) -> float:
    x = np.asarray(x).astype(np.float32)
    return float(20 * np.log10(max(float(np.sqrt(np.mean(x * x))), 1.0) / 32768)) if len(x) else -99.0


def topp(x) -> float:
    """Högsta 0,25 s."""
    return max(db(x[i:i + 4000]) for i in range(0, max(1, len(x) - 4000), 800))


def las_stereo(path):
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, 2)
    return a[:, 0].copy(), a[:, 1].copy()


def skriv_stereo(path, mic, ref):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.stack([mic, ref], axis=1).astype(np.int16).tobytes())


def las_klipp(post) -> np.ndarray:
    """Ett röstklipp (manifestets roster/paus_klipp-post) som int16."""
    p = post["path"]
    if p.endswith(".b64"):
        return np.frombuffer(base64.b64decode(open(p).read()), dtype=np.int16)
    return np.frombuffer(open(p, "rb").read(), dtype=np.int16)


def blanda(mic, x, t, gain_db=0.0):
    """Lägg röst x (16 kHz int16) i den RÅA mikrofonkanalen från t sekunder: samma sak som /rost-prov gör live (före ekosläckaren)."""
    y = np.concatenate([mic, np.zeros(max(0, int(t * RATE) + len(x) - len(mic)), np.int16)]).astype(np.int32)
    a = int(t * RATE)
    y[a:a + len(x)] += np.round(x.astype(np.float64) * 10 ** (gain_db / 20)).astype(np.int32)
    return np.clip(y, -32768, 32767).astype(np.int16)


# --------------------------------------------------------------------------------------------------------- kedjan

def post_for(rec: dict, med_underrun: bool, **extra) -> dict:
    """Manifestets inspelning till en post för kor_fil. Underrun-rapporterna matas in bara om passet har dem OCH `med_underrun`."""
    j = rec.get("just") or {}
    ok = j.get("poang") is not None and j["poang"] <= 1.5
    p = {"fil": rec["path"], "person": None}
    if ok and j.get("spelstart_ms", 0) >= 60:
        p["spelstart_ms"] = j["spelstart_ms"]
    if ok and med_underrun and rec.get("pass_underrun"):
        p["underrun_stod"] = True
        p["underruns"] = j.get("underruns", [])
    p.update(extra)
    return p


async def kor_sekvens(poster, stt, ny_aec_per_fil=False, tystnad_ms=None):
    """kor_fil över poster i given ordning med EN delad ekosläckare (som live: en per enhet) eller en ny per fil."""
    import eko_replay
    from app import eko_vakt
    from app.mjuk_aec import MjukAec
    from app.providers.local_turns import LocalTurns, PRE_END_MS
    turns = LocalTurns.create(tystnad_ms or int(os.environ.get("GEMINI_TURN_SILENCE_MS", "800")), pre_ms=PRE_END_MS)
    assert turns is not None, "ingen lokal turdetektering (sherpa_onnx/pipecat saknas)"
    aec = MjukAec.skapa()
    assert aec is not None and aec.aec_pa, "ekosläckaren startade inte (aec-audio-processing saknas?)"
    svar = eko_vakt.Svar()
    ut, t_bas = [], 1000.0
    for post in poster:
        if ny_aec_per_fil:
            aec = MjukAec.skapa()
            svar = eko_vakt.Svar()
            turns.reset()
        r, s = await eko_replay.kor_fil(dict(post), aec, turns, svar, stt, t_bas)
        t_bas += s + 10.0
        ut.append(r)
    return ut


def avbrott_i(r: dict) -> list:
    return [k for k in r["kandidater"] if k["dom"].startswith("person")]


def sammanfatta_fil(r: dict) -> dict:
    """Det en tabell behöver ur ett körresultat, med radens egna siffror (ingen tolkning)."""
    av = avbrott_i(r)
    return {"fil": r["fil"], "s": r["s"], "hans": r.get("hans"), "hans_ord_n": len((r.get("hans") or "").split()),
            "klipp": [{"t": k["dom_vid"], "via": "nivå" if k["dom"] == "person (nivå)" else "text", "hort": k["hort"], "niva": k["niva"]} for k in av],
            "kandidater": len(r["kandidater"]), "ignorerade": sum(1 for k in r["kandidater"] if k["dom"].startswith("ignorerad")),
            "turer": [{"t": t["t"], "s": t["s"], "text": t["text"], "efter_avbrott": t["efter_avbrott"], "hans_ord": t["hans_ord"]} for t in r["turer"]]}


def utfall_rost(r: dict, t_rost: float) -> dict:
    """Klipptes han av rösten som lades in vid t_rost? Samma regel som eko_replay.summera, plus vad som hände före rösten."""
    av = avbrott_i(r)
    fore = [k for k in av if k["dom_vid"] < t_rost]
    traff = next((k for k in av if k["dom_vid"] >= t_rost and k["t"] <= t_rost + 1.5), None)
    ut = {"klippt": traff is not None, "fore": len(fore), "via": None, "latens_s": None, "latens_med_stt_s": None, "hort": None}
    if traff:
        ut.update({"via": "nivå" if traff["dom"] == "person (nivå)" else "text", "latens_s": round(traff["dom_vid"] - t_rost, 2),
                   "latens_med_stt_s": round(traff["dom_vid"] + traff["stt_s"] - t_rost, 2), "hort": traff["hort"]})
    efter = [t for t in r["turer"] if t["efter_avbrott"]]
    ut["tur_text"] = efter[0]["text"] if efter else None
    ut["hans_ord_till_motor"] = sum(1 for t in r["turer"] if t["hans_ord"] and t["t"] + t["s"] < t_rost)
    return ut


def ord_lika(klipp_text, tur_text) -> "float | None":
    """Andel av klippets ord som finns i turtexten (0-1). Referensen är en lokal avskrift av klippet, inte facit."""
    from app import eko_vakt
    a, b = eko_vakt.ord_(klipp_text), set(eko_vakt.ord_(tur_text))
    return None if not a else round(sum(w in b for w in a) / len(a), 2)


async def kor_injektion(bas, varma, rost, t_rost, gain_db, stt, tmp, med_underrun=True):
    """Ett prov: `varma` (tidigare inspelningar i samma pass) värmer ekosläckaren, sedan basen med `rost` inlagd vid t_rost s."""
    mic, ref = las_stereo(bas["path"])
    x = las_klipp(rost)
    fd, path = tempfile.mkstemp(suffix=".wav", dir=tmp)
    os.close(fd)
    try:
        skriv_stereo(path, blanda(mic, x, t_rost, gain_db), ref)
        poster = [post_for(v, med_underrun) for v in varma]
        e = post_for(bas, med_underrun, erle_vid=[t_rost - 0.05])
        e["fil"] = path
        e["person"] = {"t": t_rost, "ord": rost["namn"]}
        r = (await kor_sekvens(poster + [e], stt))[-1]
    finally:
        os.unlink(path)
    u = utfall_rost(r, t_rost)
    u["erle_fore_db"] = (r.get("erle_vid") or {}).get(str(round(t_rost - 0.05, 2))) if r.get("erle_vid") else None
    if u["erle_fore_db"] is None and r.get("erle_vid"):
        u["erle_fore_db"] = next(iter(r["erle_vid"].values()))
    return u, r


# ------------------------------------------------------------------------------------------------ arbetare (multiprocessing)

_ST = {}


def arbetare_start(cache_fil, tmp, req_q, resp_qs, index_q):
    import multiprocessing as mp
    os.makedirs(tmp, exist_ok=True)
    who = index_q.get()
    _ST["tmp"] = tmp
    cache = json.load(open(cache_fil)) if cache_fil and os.path.exists(cache_fil) else {}

    class _Q:                       # lägger fråga + vem som frågar
        def put(self, x):
            req_q.put((who, x[0], x[1]))
    _ST["stt"] = SttProxy(_Q(), resp_qs[who], cache)


def arbetare_kor(uppgift: dict):
    """Ett jobb i en arbetsprocess. uppgift = {"typ": ..., ...}; returnerar (uppgift_id, resultat, nya avskrifter)."""
    stt, tmp = _ST["stt"], _ST["tmp"]
    t0 = time.monotonic()
    try:
        res = asyncio.run(_jobb(uppgift, stt, tmp))
    except Exception as e:  # ett fel i ett jobb får inte fälla hela körningen: det redovisas som fel
        res = {"fel": repr(e)}
    return uppgift["id"], res, round(time.monotonic() - t0, 1), stt.saknas


async def _jobb(u, stt, tmp):
    if u["typ"] == "pass_ren":       # alla inspelningar i ett pass i tidsordning, utan inlagd röst
        poster = [post_for(r, u["med_underrun"]) for r in u["recs"]]
        rr = await kor_sekvens(poster, stt, ny_aec_per_fil=u["kall"])
        return [{"id": r["id"], **sammanfatta_fil(x)} for r, x in zip(u["recs"], rr)]
    if u["typ"] == "injektion":     # en bas, flera (röst, t, gain)-prov
        ut = []
        for v in u["prov"]:
            ur, _r = await kor_injektion(u["bas"], u["varma"], v["rost"], v["t"], v["gain"], stt, tmp, u["med_underrun"])
            ur["tur_text"] = ur.get("tur_text")
            ut.append({**{k: v[k] for k in ("rost", "t", "gain", "set")}, "rost": v["rost"]["namn"], "ut": ur})
        return {"bas": u["bas"]["id"], "prov": ut}
    raise ValueError(u["typ"])
