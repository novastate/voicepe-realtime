"""Can Core's sentences be fed into the Live session one by one, as text Live reads aloud? (raawr US-047 AC-4)

Henrik 2026-10-09: a TTS voice for streamed Core answers is a no ("alla 3 låter som en annan röst"). Everything
spoken must be the Live session's own voice. This opens its OWN Live session (no house, no rostagent) and tries:

  seriell   sentence 1, wait until Live is done, sentence 2, ... (what a simple feeder does)
  tidsatt   a sentence every 1.5 s, as Core writes, with NO_INTERRUPTION (Live is already speaking)
  ko        all sentences sent at once with NO_INTERRUPTION (does Live queue them, in order, uncut?)
  hel       the whole answer as one text, once it is complete (the fallback)

per way: time from sending sentence 1 to the first sound, the gaps between sentences, what Live actually said
(its output transcription, compared with the text) and the audio saved. GEMINI_API_KEY is never printed.

    python tools/live_strom_prov.py [--modell models/gemini-3.8-live] [--spara DIR]
"""
import argparse
import asyncio
import json
import os
import re
import time
import wave

from google import genai
from google.genai import types

RATE = 24000
SVAR = [
    "Elpriset är lågt just nu, under trettio öre per kilowattimme.",
    "Det stiger efter klockan sex i kväll, så kör tvättmaskinen före dess.",
    "Imorgon bitti blir det dyrare igen, runt femtio öre.",
]
FORSKOTT_S = float(os.environ.get("FORSKOTT_S", "0.9"))  # how long before the end of a sentence the next is sent
PROMPT = ("Du är en uppläsare i ett hem i Sverige. Varje meddelande du får är en text som ska läsas upp "
          "ord för ord med din vanliga röst. Läs exakt det som står, säg inget annat och lägg inte till något.")


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


def config(ko):
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"], system_instruction=PROMPT,
        speech_config=types.SpeechConfig(language_code="sv-SE", voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon"))),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            activity_handling=(types.ActivityHandling.NO_INTERRUPTION if ko else types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS)),
    )


async def lyssna(s, t0, handelser, ljud, klart, antal):
    """Collect audio chunks (with time), transcription and turn ends until `antal` turns are complete."""
    fardiga = 0
    while fardiga < antal:  # receive() ends at every turn_complete: ask again for the next turn
        async for m in s.receive():
            sc = m.server_content
            if not sc:
                continue
            if sc.model_turn:
                for p in sc.model_turn.parts or []:
                    if p.inline_data and p.inline_data.data:
                        handelser.append(("ljud", time.monotonic() - t0))
                        ljud.extend(p.inline_data.data)
            if sc.output_transcription and sc.output_transcription.text:
                klart.append(sc.output_transcription.text)
            if sc.turn_complete:
                handelser.append(("slut", time.monotonic() - t0))
                fardiga += 1


def sammanfatta(handelser):
    ljud = [t for k, t in handelser if k == "ljud"]
    slut = [t for k, t in handelser if k == "slut"]
    return {"forsta_ljud_s": round(ljud[0], 2) if ljud else None, "tur_slut_s": [round(t, 2) for t in slut],
            "ljud_stycken": len(ljud)}


async def kor(client, modell, vag, spara):
    ko = vag in ("ko", "tidsatt", "forskott")
    handelser, ljud, klart = [], bytearray(), []
    async with client.aio.live.connect(model=modell, config=config(ko)) as s:
        await asyncio.sleep(1)  # an open, quiet session: what the house has when an answer starts
        t0 = time.monotonic()
        antal = 1 if vag in ("hel", "ko") else len(SVAR)
        lyssnare = asyncio.create_task(lyssna(s, t0, handelser, ljud, klart, antal))
        if vag == "hel":
            await s.send_realtime_input(text=" ".join(SVAR))
        elif vag == "ko":
            for mening in SVAR:
                await s.send_realtime_input(text=mening)
        elif vag == "tidsatt":  # as Core writes: a sentence every 1.5 s, while Live is already speaking
            for mening in SVAR:
                await s.send_realtime_input(text=mening)
                await asyncio.sleep(1.5)
        elif vag == "forskott":  # the next sentence a little BEFORE Live has finished the last (hides its 0.6 s latency)
            for mening in SVAR:
                fore = len([1 for k, _ in handelser if k == "ljud"])
                ts = time.monotonic()
                await s.send_realtime_input(text=mening)
                while len([1 for k, _ in handelser if k == "ljud"]) == fore and time.monotonic() - ts < 10:
                    await asyncio.sleep(0.02)  # wait for this sentence's first sound
                forsta = time.monotonic()
                await asyncio.sleep(max(0.0, len(mening) / 12.0 - FORSKOTT_S - (time.monotonic() - forsta)))
        else:  # seriell: the next sentence when Live has finished the last
            for i, mening in enumerate(SVAR):
                fore = len([1 for k, _ in handelser if k == "slut"])
                await s.send_realtime_input(text=mening)
                for _ in range(300):
                    await asyncio.sleep(0.1)
                    if len([1 for k, _ in handelser if k == "slut"]) > fore:
                        break
        try:
            await asyncio.wait_for(lyssnare, 40)
        except asyncio.TimeoutError:
            lyssnare.cancel()
    sagt = "".join(klart).strip()
    ut = {"vag": vag, **sammanfatta(handelser), "sagt_wer": wer(" ".join(SVAR), sagt), "sagt": sagt,
          "ljud_s": round(len(ljud) / 2 / RATE, 2)}
    if spara:
        os.makedirs(spara, exist_ok=True)
        with wave.open(os.path.join(spara, f"{vag}.wav"), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(RATE)
            f.writeframes(bytes(ljud))
    return ut


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modell", default=os.environ.get("GEMINI_MODEL", "models/gemini-3.8-live"))
    ap.add_argument("--spara")
    ap.add_argument("--vagar", default="seriell,tidsatt,ko,hel")
    a = ap.parse_args()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for vag in a.vagar.split(","):
        try:
            print(json.dumps(await kor(client, a.modell, vag, a.spara), ensure_ascii=False), flush=True)
        except Exception as e:
            nyckel = os.environ.get("GEMINI_API_KEY", "")
            print(json.dumps({"vag": vag, "fel": repr(e).replace(nyckel, "<dold>")[:200]}), flush=True)
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
