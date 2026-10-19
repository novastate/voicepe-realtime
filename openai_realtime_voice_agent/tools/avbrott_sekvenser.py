"""US-060 (spår C): probe how Gemini Live handles manual activity signal sequences over model audio.

Costs money. Owner approval is required before running this script.
This tool uses a direct Gemini key, opens isolated sessions, and does not talk to house systems.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Callable

try:
    from google import genai
    from google.genai import types as T
except ModuleNotFoundError:  # lets local fake tests run without network deps installed
    genai = None
    T = None

MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-3.8-live")
PROMPT = "Du är en svensk röstassistent. Svara kort och tydligt."
FRAGA = "Berätta en lång saga om en björn, minst en minut."
PCM_MIME = "audio/pcm;rate=16000"
PCM_CHUNK_BYTES = 640  # 20 ms at 16 kHz mono, 16-bit PCM
PCM_CHUNKS_600MS = 30
FIRST_AUDIO_WAIT_S = 12.0
POST_SEQUENCE_LISTEN_S = 8.0
MAX_SESSION_S = 25.0
MAX_AUDIO_S = 8.0
AFTER_FIRST_AUDIO_DELAY_S = 1.5


def _nu(t0: float) -> float:
    return round(time.monotonic() - t0, 3)


def _event(events: list[list[Any]], t0: float, event_type: str, detail: str | None = None) -> None:
    row: list[Any] = [_nu(t0), event_type]
    if detail:
        row.append(detail)
    events.append(row)


def _format_exception(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    text = text[:200]
    code = getattr(exc, "code", None)
    if code is None:
        close_code = getattr(getattr(exc, "close", None), "code", None)
        code = close_code
    if code is not None and str(code) not in text:
        text = f"{text} (code={code})"
    return f"{type(exc).__name__}: {text}"


async def _send_short_pcm(
    session: Any,
    *,
    sleep: Callable[[float], asyncio.Future] = asyncio.sleep,
) -> None:
    blob = _audio_blob(b"\x00" * PCM_CHUNK_BYTES)
    for _ in range(PCM_CHUNKS_600MS):
        await session.send_realtime_input(audio=blob)
        await sleep(0.02)


async def sequence_s1(session: Any, *, sleep: Callable[[float], asyncio.Future] = asyncio.sleep) -> None:
    await session.send_realtime_input(activity_start=_activity_start())
    await _send_short_pcm(session, sleep=sleep)
    await session.send_realtime_input(activity_end=_activity_end())


async def sequence_s2(session: Any, *, sleep: Callable[[float], asyncio.Future] = asyncio.sleep) -> None:
    await session.send_realtime_input(activity_start=_activity_start())
    await sleep(2.0)
    await session.send_realtime_input(activity_start=_activity_start())
    await _send_short_pcm(session, sleep=sleep)
    await session.send_realtime_input(activity_end=_activity_end())


async def sequence_s3(session: Any, *, sleep: Callable[[float], asyncio.Future] = asyncio.sleep) -> None:
    del sleep
    await session.send_realtime_input(activity_start=_activity_start())
    await session.send_realtime_input(activity_end=_activity_end())


async def sequence_s4(session: Any, *, sleep: Callable[[float], asyncio.Future] = asyncio.sleep) -> None:
    del session
    await sleep(0)


SEQUENCES: dict[str, Callable[..., asyncio.Future]] = {
    "S1": sequence_s1,
    "S2": sequence_s2,
    "S3": sequence_s3,
    "S4": sequence_s4,
}


async def run_signal_sequence(
    session: Any,
    sequence_id: str,
    *,
    sleep: Callable[[float], asyncio.Future] = asyncio.sleep,
) -> None:
    if sequence_id not in SEQUENCES:
        raise ValueError(f"Unknown sequence: {sequence_id}")
    await SEQUENCES[sequence_id](session, sleep=sleep)


def _activity_start() -> Any:
    if T is None:
        return object()
    return T.ActivityStart()


def _activity_end() -> Any:
    if T is None:
        return object()
    return T.ActivityEnd()


def _audio_blob(data: bytes) -> Any:
    if T is None:
        return data
    return T.Blob(data=data, mime_type=PCM_MIME)


def _build_config():
    if T is None:
        raise RuntimeError("google-genai is required to run this script.")
    return T.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=PROMPT,
        input_audio_transcription=T.AudioTranscriptionConfig(),
        realtime_input_config=T.RealtimeInputConfig(
            automatic_activity_detection=T.AutomaticActivityDetection(disabled=True)
        ),
    )


async def _run_one(client: Any, sequence_id: str, run_index: int) -> dict[str, Any]:
    item_id = f"{sequence_id}-r{run_index}"
    t0 = time.monotonic()
    events: list[list[Any]] = []
    first_audio = asyncio.Event()
    session_closed = asyncio.Event()
    sequence_done = asyncio.Event()
    got_generation_complete = False
    got_turn_complete = False
    audio_active = False
    total_audio_s = 0.0

    try:
        async with client.aio.live.connect(model=MODEL, config=_build_config()) as session:
            async def receive_loop() -> None:
                nonlocal audio_active, total_audio_s, got_generation_complete, got_turn_complete
                try:
                    async for msg in session.receive():
                        sc = msg.server_content
                        if not sc:
                            continue
                        if sc.model_turn:
                            for part in sc.model_turn.parts or []:
                                if part.inline_data and part.inline_data.data:
                                    if not audio_active:
                                        audio_active = True
                                        _event(events, t0, "audio_start")
                                    first_audio.set()
                                    chunk_s = len(part.inline_data.data) / 48000.0
                                    total_audio_s += chunk_s
                                    if total_audio_s >= MAX_AUDIO_S and audio_active:
                                        audio_active = False
                                        _event(events, t0, "audio_slut", f"{total_audio_s:.3f}")
                                        return
                        if sc.interrupted:
                            _event(events, t0, "interrupted")
                            if audio_active:
                                audio_active = False
                                _event(events, t0, "audio_slut", f"{total_audio_s:.3f}")
                        if sc.generation_complete:
                            got_generation_complete = True
                            _event(events, t0, "generation_complete")
                            if audio_active:
                                audio_active = False
                                _event(events, t0, "audio_slut", f"{total_audio_s:.3f}")
                        if sc.turn_complete:
                            got_turn_complete = True
                            _event(events, t0, "turn_complete")
                            if audio_active:
                                audio_active = False
                                _event(events, t0, "audio_slut", f"{total_audio_s:.3f}")
                        if sc.input_transcription and sc.input_transcription.text:
                            _event(events, t0, "input_transcription", sc.input_transcription.text)
                except Exception as exc:
                    _event(events, t0, "exception", _format_exception(exc))
                finally:
                    if audio_active:
                        _event(events, t0, "audio_slut", f"{total_audio_s:.3f}")
                    _event(events, t0, "stängd")
                    session_closed.set()

            receiver = asyncio.create_task(receive_loop())
            await session.send_realtime_input(text=FRAGA)
            try:
                await asyncio.wait_for(first_audio.wait(), timeout=FIRST_AUDIO_WAIT_S)
            except asyncio.TimeoutError:
                _event(events, t0, "exception", "TimeoutError: no audio_start within wait window")
            else:
                await asyncio.sleep(AFTER_FIRST_AUDIO_DELAY_S)
                await run_signal_sequence(session, sequence_id)
                sequence_done.set()

            deadline = t0 + MAX_SESSION_S
            post_listen_deadline = min(
                deadline,
                time.monotonic() + (POST_SEQUENCE_LISTEN_S if sequence_done.is_set() else 0.0),
            )
            while not session_closed.is_set():
                now = time.monotonic()
                if now >= deadline:
                    break
                if sequence_done.is_set() and now >= post_listen_deadline:
                    break
                if got_turn_complete and got_generation_complete:
                    break
                await asyncio.sleep(0.05)

            if not receiver.done():
                receiver.cancel()
                try:
                    await receiver
                except asyncio.CancelledError:
                    pass
    except Exception as exc:
        _event(events, t0, "exception", _format_exception(exc))
        _event(events, t0, "stängd")

    return {"id": item_id, "sek_session": round(time.monotonic() - t0, 1), "sek_ljud": round(total_audio_s, 1), "handelser": events}


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before running this script.")
    return key


async def _amain(args: argparse.Namespace) -> None:
    if genai is None:
        raise RuntimeError("google-genai is required to run this script.")
    client = genai.Client(api_key=_api_key())
    totalt_s = 0.0
    with open(args.output, "a", encoding="utf-8") as out:
        for run_index in range(1, args.upprepa + 1):
            for sequence_id in ("S1", "S2", "S3", "S4"):
                row = await _run_one(client, sequence_id, run_index)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                totalt_s += row["sek_session"]
                print(f"{row['id']}: {row['sek_session']} s session, {row['sek_ljud']} s ljud (totalt {totalt_s:.1f} s)", flush=True)
                # Avbryt vid fel som inte hör till mätningen: ingen uppkoppling, ingen nyckel, ingen kvot, inget första ljud.
                if any(h[1] == "exception" and not any(x[1] == "audio_start" for x in row["handelser"]) for h in row["handelser"]):
                    print("AVBRYTER: fel före första svarsljudet; inga fler sessioner körs.", flush=True)
                    return


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", help="Output JSONL file, one row per sequence run")
    ap.add_argument("--upprepa", type=int, default=1, help="Run S1-S4 this many times (default: 1)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.upprepa != 1:
        raise SystemExit("--upprepa är låst till 1: högst fyra sessioner (S1-S4), ägarens OK gäller bara dem")
    print(
        "VARNING: detta skript kostar pengar (Gemini Live) och kräver uttryckligt OK från ägaren innan körning.",
        flush=True,
    )
    asyncio.run(_amain(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
