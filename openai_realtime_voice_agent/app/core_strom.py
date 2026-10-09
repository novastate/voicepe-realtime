"""Speak a Core answer while it is being written (raawr US-047 AC-4, US-036).

Core streams the answer to a contract question as SSE frames (`start`, `token`*, `deferred`?,
`done` | `error`). This turns the tokens into sentences and speaks each one as soon as it is
whole, in the engine's own voice, through the same path as every other out-of-band line
(`say`). The model gets the tool `fraga_core`, which returns at once ("checking") while the answer
is spoken here. OFF unless CORE_STREAM_TALA=1: the road in front of Core (Comms forwarding the
stream) is spår A's, so the URL is CORE_STROM_URL until that exists.

The model does not hear what is spoken (it is out of band): a follow-up that refers to it works
only through what the owner remembers. That is the price of speaking before the answer is done.
"""
import json
import logging
import os
import re
import time
import uuid
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

MIN_TECKEN = 25  # a sentence shorter than this waits for the next one ("Ja. Det går.")
# A full stop that is not the end of a sentence: the word before it.
FORKORTNINGAR = ("t.ex", "bl.a", "m.m", "m.fl", "o.s.v", "osv", "dvs", "ca", "kl", "nr", "dr", "jfr", "resp",
                 "fr.o.m", "t.o.m", "s.k", "mfl", "prof", "st")
_GRANS = re.compile(r"[.!?…]+[\"')]*\s+(?=[A-ZÅÄÖ0-9\"'(])")


def core_stream_tala_paa() -> bool:
    return os.environ.get("CORE_STREAM_TALA", "0") == "1"


class Meningsdelare:
    """Tokens in, whole sentences out. `mata()` gives the sentences that are complete now,
    `flush()` the rest when the stream is done."""

    def __init__(self) -> None:
        self.buf = ""

    def mata(self, text: str) -> List[str]:
        self.buf += text
        ut, start = [], 0
        for m in _GRANS.finditer(self.buf):
            kandidat = self.buf[start:m.end()].strip()
            sista = kandidat.lower().rstrip(" .!?…\"')")
            if len(kandidat) < MIN_TECKEN or sista.endswith(FORKORTNINGAR):
                continue
            ut.append(kandidat)
            start = m.end()
        self.buf = self.buf[start:]
        return ut

    def flush(self) -> List[str]:
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []


async def las_ramar(chunks: AsyncIterator[bytes]) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
    """SSE bytes in, (event, data) out. A frame that is not JSON is skipped."""
    buf = ""
    async for chunk in chunks:
        buf += chunk.decode("utf-8", errors="replace")
        while True:
            m = re.search(r"\r?\n\r?\n", buf)
            if not m:
                break
            block, buf = buf[:m.start()], buf[m.end():]
            ev = re.search(r"^event: ?(.*)$", block, re.M)
            da = re.search(r"^data: ?(.*)$", block, re.M)
            if not ev or not da:
                continue
            try:
                yield ev.group(1).strip(), json.loads(da.group(1))
            except ValueError:
                continue


FEL_REPLIK = "Jag fick inget svar från huset just nu."


async def tala_strom(chunks: AsyncIterator[bytes], say: Callable[[str], Awaitable[None]],
                     klocka: Callable[[], float] = time.monotonic) -> Dict[str, Any]:
    """Speak the stream's sentences with `say`. Returns the timing of the run (no words)."""
    t0 = klocka()
    delare = Meningsdelare()
    ut: Dict[str, Any] = {"meningar": 0, "forsta_token_s": None, "forsta_mening_s": None, "slut": None}

    async def tala(text: str) -> None:
        if ut["forsta_mening_s"] is None:
            ut["forsta_mening_s"] = round(klocka() - t0, 3)
        ut["meningar"] += 1
        await say(text)

    async for event, d in las_ramar(chunks):
        if event == "token":
            if ut["forsta_token_s"] is None:
                ut["forsta_token_s"] = round(klocka() - t0, 3)
            for m in delare.mata(str(d.get("text", ""))):
                await tala(m)
        elif event == "done":
            resp = d.get("response") or {}
            if resp.get("type") == "deferred":  # an agent works on it: Core's own line says so
                if resp.get("text"):
                    await tala(str(resp["text"]))
                ut["slut"] = "deferred"
            else:
                for m in delare.flush():
                    await tala(m)
                ut["slut"] = "done"
            return ut
        elif event == "deferred":
            ut["slut"] = "deferred"  # the `done` frame that follows carries the line to speak
        elif event == "error":
            await tala(FEL_REPLIK)
            ut["slut"] = "error"
            return ut
    for m in delare.flush():  # the stream ended without `done`: speak what there is
        await tala(m)
    ut["slut"] = ut["slut"] or "avbruten"
    return ut


def get_fraga_core_definition() -> dict:
    return {
        "type": "function",
        "name": "fraga_core",
        "description": (
            "Ask the house's brain (memory of the owner and the family, research, sub-agents) when a "
            "house tool and recall_memory cannot answer. Returns at once; the answer is then spoken "
            "for you, sentence by sentence, in your voice. After calling it say nothing more yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "The question, with any needed context"}},
            "required": ["question"],
        },
    }


def strom_url() -> str:
    return (os.environ.get("CORE_STROM_URL") or "").strip()


def register_fraga_core(llm, say: Callable[[str], Awaitable[None]], headers: Callable[[], Dict[str, str]],
                        timeout_s: float = 90.0) -> None:
    """Register `fraga_core`: it returns "checking" and speaks the streamed answer in the background."""
    taken = set()

    async def kor(question: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                async with client.stream(
                    "POST", strom_url(), json={"text": question, "language": "sv", "request_id": uuid.uuid4().hex},
                    headers={**headers(), "Accept": "text/event-stream"},
                ) as r:
                    if r.status_code != 200:
                        logger.warning(f"⚠️ fraga_core: the stream answered {r.status_code}")
                        await say(FEL_REPLIK)
                        return
                    ut = await tala_strom(r.aiter_bytes(), say)
            logger.info(f"⏱ core-ström första_token={ut['forsta_token_s']} första_mening={ut['forsta_mening_s']} "
                        f"meningar={ut['meningar']} slut={ut['slut']}")
        except Exception as e:
            logger.warning(f"⚠️ fraga_core failed: {type(e).__name__}")
            try:
                await say(FEL_REPLIK)
            except Exception:
                pass

    async def _fraga(params) -> None:
        import asyncio
        question = ((params.arguments or {}).get("question") or "").strip()
        if not question or not strom_url():
            await params.result_callback({"error": "no question" if not question else "not available"})
            return
        task = asyncio.get_running_loop().create_task(kor(question))
        taken.add(task)
        task.add_done_callback(taken.discard)
        await params.result_callback({"status": "checking; the answer is spoken for you, say nothing more"})

    llm.register_function("fraga_core", _fraga)
