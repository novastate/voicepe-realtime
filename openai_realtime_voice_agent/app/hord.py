"""Raawr US-542 (Henrik 2026-11-09): the owner's yes to a calendar change must be HEARD by the code, not written by the model.

Every utterance the speech-to-text hears is reported to Comms (`<room>/hort`, the room's own key, which the model never
holds) and on to Core. Core, not the model, classifies it. The model cannot reach this road: it is no tool, and the
calendar tools carry no field for the owner's words. The report is fire and forget and never holds up the audio.
"""
import asyncio
import logging
import os
from typing import Optional, Set

import httpx

from app import ha_api

logger = logging.getLogger(__name__)
_uppgifter: Set[asyncio.Task] = set()


def hort_url() -> str:
    """`<comms address of the room>/hort` (the `/api` suffix of HA_API_URL is dropped, as for `/fraga`).
    CORE_HORT_URL overrides it (a test); empty = off."""
    egen = (os.environ.get("CORE_HORT_URL") or "").strip()
    if egen:
        return egen
    bas = ha_api.base()
    return (bas[:-4] if bas.endswith("/api") else bas) + "/hort" if bas else ""


async def skicka(text: str, klient: Optional[httpx.AsyncClient] = None) -> bool:
    url = hort_url()
    if not url or not text.strip():
        return False
    try:
        if klient is not None:
            r = await klient.post(url, json={"text": text[:2000]}, headers=ha_api.headers(), timeout=5.0)
        else:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.post(url, json={"text": text[:2000]}, headers=ha_api.headers())
        return r.status_code == 200
    except Exception as e:
        logger.warning(f"⚠️ hord: could not report ({type(e).__name__})")
        return False


def rapportera(text: str) -> None:
    """Called for each TranscriptionFrame. Order is kept per call by awaiting inside one chain of tasks."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    forra = list(_uppgifter)

    async def kor() -> None:
        for t in forra:  # an utterance is reported after the one before it: Core counts them in order
            try:
                await t
            except BaseException:
                pass
        await skicka(text)

    t = loop.create_task(kor())
    _uppgifter.add(t)
    t.add_done_callback(_uppgifter.discard)
