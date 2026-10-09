"""Compare how Gemini Live hears the same recorded turns, two ways (raawr US-032, Henrik 2026-10-09).

A standalone process: its own sessions to Google, no house, no Home Assistant, no rostagent.
Needs GEMINI_API_KEY in the environment (it is never printed or written).

    python tools/gemini_jamforelse.py DIR [--modell models/gemini-3.8-live] [--lage drift,google]

DIR holds `<namn>.wav` (16 kHz mono PCM16) and `<namn>.txt` (what was really said). An optional
`<namn>.json` with {"forrulle_bytes": N} says how many bytes at the start are pre-roll (the audio
from before the local VAD said speech; it may carry the tail of Björn's own reply).

Two ways, per turn:
  drift   as in the house today: Google's own detection OFF, our activityStart + the audio
          (pre-roll included) + activityEnd after 800 ms of silence, language sv-SE.
  google  as Google intends: Google's own detection ON (no activityStart/End), language sv-SE,
          the audio streamed in real time WITHOUT the pre-roll, then silence until the answer.

Per turn it prints one JSON line: the transcript Google made, the words that differ from the truth
(word error rate), whether the language looks wrong, and the time from the end of speech to the
first sound of the answer. A summary per way comes last.
"""
import argparse
import asyncio
import glob
import json
import os
import re
import time
import wave

from google import genai
from google.genai import types

RATE = 16000
CHUNK = RATE * 2 * 20 // 1000  # 20 ms
SILENCE_MS = 800
SV = {"och", "att", "det", "är", "jag", "du", "inte", "en", "på", "för", "med", "som", "vad", "kan", "hur", "har", "ska", "mig", "dig", "vi", "här"}
EJ_SV = {"the", "and", "you", "is", "what", "weiter", "und", "ich", "nicht", "der", "die", "das", "los", "que", "para", "una", "el"}
PROMPT = "Du är Björn, en röstassistent i ett hem i Sverige. Svara kort på svenska."


def skrubba(text):
    """An exception's text can carry the API key (a `key=` in the URL, or the key itself)."""
    text = re.sub(r"(key=)[^&\s'\")]+", r"\1<dold>", str(text), flags=re.I)
    nyckel = os.environ.get("GEMINI_API_KEY", "")
    return text.replace(nyckel, "<dold>") if nyckel else text


def ord_(t):
    return re.sub(r"[^\wåäö ]+", " ", (t or "").lower()).split()


def wer(sant, hort):
    a, b = ord_(sant), ord_(hort)
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return round(d[len(b)] / max(1, len(a)), 3)


def fel_sprak(t):
    w = set(ord_(t))
    return len(w & EJ_SV) > len(w & SV)


def las(namn):
    with wave.open(namn + ".wav") as f:
        pcm = f.readframes(f.getnframes())
    sant = open(namn + ".txt", encoding="utf-8").read().strip()
    meta = json.load(open(namn + ".json")) if os.path.exists(namn + ".json") else {}
    return pcm, sant, int(meta.get("forrulle_bytes", 0))


def config(lage):
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=PROMPT,
        speech_config=types.SpeechConfig(
            language_code="sv-SE",
            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon"))),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=(lage == "drift"))),
    )


async def tur(client, modell, lage, pcm, forrulle):
    """One turn in a fresh session. Returns (transcript, seconds from speech end to first sound)."""
    delar, forsta, slut = [], None, [None]
    async with client.aio.live.connect(model=modell, config=config(lage)) as s:
        async def lyssna():
            nonlocal forsta
            async for m in s.receive():
                sc = m.server_content
                if sc and sc.input_transcription and sc.input_transcription.text:
                    delar.append(sc.input_transcription.text)
                if sc and sc.model_turn and any(p.inline_data for p in sc.model_turn.parts or []):
                    if forsta is None and slut[0] is not None:
                        forsta = time.monotonic() - slut[0]
                    return
                if sc and sc.turn_complete:
                    return

        lyssnare = asyncio.create_task(lyssna())
        tyst = b"\x00\x00" * (RATE * SILENCE_MS // 1000)
        if lage == "drift":  # our activity signals, the audio as the house holds it, silence first
            await s.send_realtime_input(activity_start=types.ActivityStart())
            ljud = pcm + tyst
            for i in range(0, len(ljud), CHUNK * 5):
                await s.send_realtime_input(audio=types.Blob(data=ljud[i:i + CHUNK * 5], mime_type="audio/pcm;rate=16000"))
            slut[0] = time.monotonic() - SILENCE_MS / 1000  # speech ended 800 ms before activityEnd
            await s.send_realtime_input(activity_end=types.ActivityEnd())
        else:  # Google's detection: real time, no pre-roll, silence after
            ljud = pcm[forrulle:]
            for i in range(0, len(ljud), CHUNK):
                await s.send_realtime_input(audio=types.Blob(data=ljud[i:i + CHUNK], mime_type="audio/pcm;rate=16000"))
                await asyncio.sleep(0.02)
            slut[0] = time.monotonic()
            for _ in range(150):  # up to 3 s of silence while Google decides
                if lyssnare.done():
                    break
                await s.send_realtime_input(audio=types.Blob(data=b"\x00\x00" * (CHUNK // 2), mime_type="audio/pcm;rate=16000"))
                await asyncio.sleep(0.02)
        try:
            await asyncio.wait_for(lyssnare, 15)
        except asyncio.TimeoutError:
            lyssnare.cancel()
    return "".join(delar).strip(), forsta


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--modell", default=os.environ.get("GEMINI_MODEL", "models/gemini-3.8-live"))
    ap.add_argument("--lage", default="drift,google")
    ap.add_argument("--radera", action="store_true", help="delete every saved turn in DIR and stop")
    a = ap.parse_args()
    if a.radera:
        n = 0
        for f in glob.glob(os.path.join(a.dir, "*")):
            if f.endswith((".wav", ".json", ".txt")):
                os.remove(f)
                n += 1
        print(f"deleted {n} files in {a.dir}")
        return
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    sammanfattning = {}
    for lage in a.lage.split(","):
        rader = []
        for wav in sorted(glob.glob(os.path.join(a.dir, "*.wav"))):
            namn = wav[:-4]
            pcm, sant, forrulle = las(namn)
            try:
                hort, forsta = await tur(client, a.modell, lage, pcm, forrulle)
                fel = None
            except Exception as e:  # a refused setup is a result, not a crash
                hort, forsta, fel = "", None, skrubba(repr(e))[:160]
            r = {"lage": lage, "tur": os.path.basename(namn), "sant": sant, "hort": hort,
                 "wer": wer(sant, hort) if hort else None, "fel_sprak": fel_sprak(hort),
                 "forsta_ljud_s": round(forsta, 2) if forsta is not None else None, "fel": fel}
            print(json.dumps(r, ensure_ascii=False), flush=True)
            rader.append(r)
            await asyncio.sleep(1)
        ok = [r for r in rader if r["wer"] is not None]
        t = sorted(r["forsta_ljud_s"] for r in rader if r["forsta_ljud_s"] is not None)
        sammanfattning[lage] = {
            "turer": len(rader), "utan_transkript": len(rader) - len(ok),
            "wer_medel": round(sum(r["wer"] for r in ok) / len(ok), 3) if ok else None,
            "helt_ratt": sum(1 for r in ok if r["wer"] == 0), "fel_sprak": sum(1 for r in rader if r["fel_sprak"]),
            "forsta_ljud_p50": t[len(t) // 2] if t else None, "fel": sum(1 for r in rader if r["fel"])}
    print(json.dumps({"sammanfattning": sammanfattning}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
