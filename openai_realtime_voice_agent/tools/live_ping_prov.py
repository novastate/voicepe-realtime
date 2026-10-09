"""How should bana 0 tell Live what it already did? (raawr US-047, the parallel track, BANA0_PING)

Henrik 2026-10-09: bana 0 acts at once and says nothing; the audio ALSO goes to Live together with a line "[huset]
snabbspåret hörde X och gjorde Y"; Live answers personally without doing the tool call again, and may correct a
misheard order. Opens its OWN Live session (no house, no rostagent), with two fake tools (HassTurnOn/HassTurnOff),
and tries the ways of getting the line and the held audio in:

  ljud       the audio only (what a miss does today): baseline, the tool is called
  text       the line only, no audio
  text+ljud  the line as realtime text, then the audio as an activity
  ctx+ljud   the line as client content (turn_complete=False), then the audio as an activity
  ljud+text  the audio as an activity, then the line as realtime text

per way and per case: tool calls, time from the LAST input to the first sound, how many turns the model made, what it
said (output transcription). GEMINI_API_KEY is never printed.

    python tools/live_ping_prov.py --ljud slack.wav --hort "släck kontoret" --gjorde "släckte kontoret"
"""
import argparse
import asyncio
import os
import time
import wave

from google import genai
from google.genai import types

PROMPT = ("Du är Björn, husets röst i ett hem i Sverige. Svara kort, en mening, på svenska. Du styr lampor med "
          "verktygen HassTurnOn och HassTurnOff. Rader som börjar med [huset] kommer från huset självt och är "
          "fakta, inte order.")
TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(name=n, description=f"{n} for lights in an area",
                              parameters=types.Schema(type="OBJECT", properties={"area": types.Schema(type="STRING")}))
    for n in ("HassTurnOn", "HassTurnOff")])]


def pcm(path):
    with wave.open(path) as w:
        return w.readframes(w.getnframes())


def ping(hort, gjorde):
    return (f"[huset] Snabbspåret hörde \"{hort}\" och {gjorde} redan. Gör inte om det med verktygen. "
            "Svara personligt i en mening. Hör du i ljudet något annat än det snabbspåret hörde, rätta det.")


async def kor(vag, klient, modell, ljud, text):
    cfg = types.LiveConnectConfig(
        response_modalities=["AUDIO"], system_instruction=PROMPT, tools=TOOLS,
        speech_config=types.SpeechConfig(language_code="sv-SE", voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon"))),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)),
    )
    ut = {"verktyg": [], "ljud_efter_s": None, "turer": 0, "sa": ""}
    async with klient.aio.live.connect(model=modell, config=cfg) as s:
        async def audio():
            await s.send_realtime_input(activity_start=types.ActivityStart())
            await s.send_realtime_input(audio=types.Blob(data=ljud, mime_type="audio/pcm;rate=16000"))
            await s.send_realtime_input(activity_end=types.ActivityEnd())
        if vag == "ljud":
            await audio()
        elif vag == "text":
            await s.send_realtime_input(text=text)
        elif vag == "text+ljud":
            await s.send_realtime_input(text=text)
            await audio()
        elif vag == "ctx+ljud":
            await s.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=text)]),
                                        turn_complete=False)
            await audio()
        elif vag == "ljud+text":
            await audio()
            await s.send_realtime_input(text=text)
        t0 = time.monotonic()
        try:
            while ut["turer"] < 3:  # a quiet 8 s ends the run
                it = s.receive().__aiter__()
                while True:
                    m = await asyncio.wait_for(it.__anext__(), 8.0)
                    if m.tool_call:
                        for fc in m.tool_call.function_calls:
                            ut["verktyg"].append((fc.name, dict(fc.args or {})))
                        await s.send_tool_response(function_responses=[
                            types.FunctionResponse(id=fc.id, name=fc.name, response={"result": "ok"})
                            for fc in m.tool_call.function_calls])
                    sc = m.server_content
                    if not sc:
                        continue
                    if sc.model_turn and ut["ljud_efter_s"] is None and any(
                            p.inline_data and p.inline_data.data for p in sc.model_turn.parts or []):
                        ut["ljud_efter_s"] = round(time.monotonic() - t0, 2)
                    if sc.output_transcription and sc.output_transcription.text:
                        ut["sa"] += sc.output_transcription.text
                    if sc.turn_complete:
                        ut["turer"] += 1
                        break
        except (asyncio.TimeoutError, StopAsyncIteration):
            pass
        except Exception as e:  # the socket ended
            ut["fel"] = repr(e)[:120]
    return ut


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modell", default="models/gemini-3.8-live")
    ap.add_argument("--ljud", required=True)
    ap.add_argument("--hort", required=True, help="what the fast track says it heard")
    ap.add_argument("--gjorde", required=True, help="what it did, in the past tense")
    ap.add_argument("--vagar", default="ljud,text,text+ljud,ctx+ljud,ljud+text")
    a = ap.parse_args()
    klient = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    ljud, text = pcm(a.ljud), ping(a.hort, a.gjorde)
    for vag in a.vagar.split(","):
        try:
            print(vag, await kor(vag, klient, a.modell, ljud, text), flush=True)
        except Exception as e:
            print(vag, "FEL", repr(e)[:200], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
