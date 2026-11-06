"""One journal line per speaker and hour: how many turns, how fast, how often connected (raawr US-489).

    📊 tim kontoret turer=3 utan_ljud=1 median_ms=1200 max_ms=2400 anslutningar=2
    📊 tim koket turer=0 utan_ljud=0 median_ms=- max_ms=- anslutningar=1 tyst

Numbers and the device id only, never a sentence from a conversation. `median_ms`/`max_ms` are the
`enhet=` time of `⏱ tider` (end of speech to first sound at the device), over the turns of the hour
that got sound; `-` when none did. `tyst` is added after 24 hourly lines in a row with zero turns.
The count starts when the process starts, so a restart delays `tyst` by up to a day. A speaker is
known from its first connection or turn and gets a line every hour after that, also with 0 turns.
Find it with: journalctl -u raawr-rostagent --since -2h | grep "tim "
"""
import asyncio
import logging
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)

TYST_TIMMAR = 24
TIMME_S = 3600.0


class TimSammanfattning:
    def __init__(self):
        self._tur = defaultdict(lambda: {"turer": 0, "utan_ljud": 0, "enhet": [], "anslutningar": 0})
        self._tomma_timmar = {}  # device -> hourly lines in a row with 0 turns

    def anslutning(self, device_id: str) -> None:
        self._tur[device_id]["anslutningar"] += 1
        self._tomma_timmar.setdefault(device_id, 0)

    def tur(self, device_id: str, enhet_ms=None, utan_ljud: bool = False) -> None:
        d = self._tur[device_id]
        self._tomma_timmar.setdefault(device_id, 0)
        d["turer"] += 1
        if utan_ljud:
            d["utan_ljud"] += 1
        elif enhet_ms is not None:
            d["enhet"].append(enhet_ms)

    def skriv_timme(self) -> list:
        """Write one line per known speaker, reset the hour, and return the lines."""
        rader = []
        for device_id in sorted(self._tomma_timmar):
            d = self._tur[device_id]
            if d["turer"]:
                self._tomma_timmar[device_id] = 0
            else:
                self._tomma_timmar[device_id] += 1
            tider = d["enhet"]
            median = round(statistics.median(tider)) if tider else "-"
            storst = round(max(tider)) if tider else "-"
            rad = (f"📊 tim {device_id} turer={d['turer']} utan_ljud={d['utan_ljud']} "
                   f"median_ms={median} max_ms={storst} anslutningar={d['anslutningar']}")
            if self._tomma_timmar[device_id] >= TYST_TIMMAR:
                rad += " tyst"
            logger.info(rad)
            rader.append(rad)
        self._tur.clear()
        return rader


SAMMANFATTNING = TimSammanfattning()


async def timloop(sammanfattning: TimSammanfattning = SAMMANFATTNING, every: float = TIMME_S) -> None:
    """Write the hourly lines until cancelled."""
    while True:
        await asyncio.sleep(every)
        try:
            sammanfattning.skriv_timme()
        except Exception as e:
            logger.warning(f"⚠️ tim-sammanfattning failed: {e!r}")
