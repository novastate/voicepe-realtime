"""Does a streamed Core sentence (Gemini TTS) sound like the Live answer? (raawr US-047 AC-4)

The same sentences are read twice in the voice Charon: by Gemini Live (the model answering) and by
Gemini TTS (what `fraga_core` speaks). Both come back as 24 kHz PCM16. Per sentence it prints
duration, speaking rate, pitch (median F0, range), loudness and brightness, and the difference.
Needs GEMINI_API_KEY (never printed). Run on core with its venv:

    python tools/rostjamforelse.py [--modell models/gemini-3.8-live] [--spara DIR]
"""
import argparse
import asyncio
import json
import os
import sys
import wave

import numpy as np

RATE = 24000
MENINGAR = [
    "Elpriset är lågt just nu, under trettio öre per kilowattimme.",
    "Det blir regn i morgon förmiddag, sedan klarnar det upp.",
    "Jag hittade tre anteckningar om det, den senaste är från i måndags.",
    "Det där vet jag inte säkert, men jag kan titta närmare på det.",
    "Din syster ringde i morse och undrade om ni ska ses på lördag.",
]


def drag(pcm: bytes) -> dict:
    """Duration, rate, pitch, loudness and brightness of 24 kHz mono PCM16."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    lang, hop, ram = len(x) / RATE, RATE // 100, RATE * 40 // 1000
    f0, cent, rms_db = [], [], []
    taket = np.sqrt(np.mean(x ** 2) + 1e-12)
    for i in range(0, len(x) - ram, hop):
        f = x[i:i + ram] * np.hanning(ram)
        rms = np.sqrt(np.mean(f ** 2) + 1e-12)
        if rms < 0.3 * taket:  # silence between words
            continue
        rms_db.append(20 * np.log10(rms))
        sp = np.abs(np.fft.rfft(f))
        fr = np.fft.rfftfreq(ram, 1 / RATE)
        cent.append(float((sp * fr).sum() / (sp.sum() + 1e-9)))
        ac = np.correlate(f, f, "full")[ram - 1:]
        lo, hi = RATE // 400, RATE // 70
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.35 * ac[0]:
            f0.append(RATE / k)
    return {"sekunder": round(lang, 2),
            "f0_median": round(float(np.median(f0)), 1) if f0 else None,
            "f0_spann": round(float(np.percentile(f0, 90) - np.percentile(f0, 10)), 1) if f0 else None,
            "ljudstyrka_db": round(float(np.mean(rms_db)), 1) if rms_db else None,
            "ljushet_hz": round(float(np.mean(cent)), 0) if cent else None}


async def live(client, modell, text) -> bytes:
    from google.genai import types

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction="Läs upp exakt den text du får, ord för ord, på svenska. Säg inget annat och lägg inte till något.",
        speech_config=types.SpeechConfig(language_code="sv-SE", voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon"))))
    ljud = b""
    async with client.aio.live.connect(model=modell, config=config) as s:
        await s.send_realtime_input(text=text)
        async for m in s.receive():
            sc = m.server_content
            if sc and sc.model_turn:
                for p in sc.model_turn.parts or []:
                    if p.inline_data and p.inline_data.data:
                        ljud += p.inline_data.data
            if sc and sc.turn_complete:
                break
    return ljud


def spara(katalog, namn, pcm):
    os.makedirs(katalog, exist_ok=True)
    with wave.open(os.path.join(katalog, namn + ".wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(pcm)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modell", default=os.environ.get("GEMINI_MODEL", "models/gemini-3.8-live"))
    ap.add_argument("--spara")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from google import genai

    from app.early_ack import gemini_tts

    nyckel = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=nyckel)
    rader = []
    for i, text in enumerate(MENINGAR):
        l = await live(client, a.modell, text)
        t = await gemini_tts(text, nyckel, "Charon", cache=False)
        if a.spara:
            spara(a.spara, f"{i}-live", l)
            spara(a.spara, f"{i}-tts", t)
        dl, dt = drag(l), drag(t)
        rader.append({"mening": i, "live": dl, "tts": dt,
                      "skillnad": {k: (round(dt[k] - dl[k], 2) if dt[k] is not None and dl[k] is not None else None) for k in dl}})
        print(json.dumps(rader[-1], ensure_ascii=False), flush=True)
        await asyncio.sleep(7)  # Gemini TTS: 10 requests a minute
    def medel(sida, nyckel_):
        v = [r[sida][nyckel_] for r in rader if r[sida][nyckel_] is not None]
        return round(sum(v) / len(v), 2) if v else None
    print(json.dumps({"medel": {s: {k: medel(s, k) for k in rader[0]["live"]} for s in ("live", "tts")}}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
