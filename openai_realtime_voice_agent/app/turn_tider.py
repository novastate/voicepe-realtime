"""The stages of one turn, in milliseconds after the end of speech (raawr US-032/US-047 AC-1).

One line per turn in the journal, no words:

    ⏱ tider kontoret turslut=812 stt=1002 comms=1150 verktyg=1900 modell=2310 enhet=2360

`talets_slut` is the turn end minus the silence the local detector waits for (an estimate, good to
a few tens of ms). The steps that did not happen in a turn are left out: no `stt`/`comms` when bana 0
was not asked, no `verktyg` without a tool call, no `modell` on an engine that does not tell. The
line is written when the first audio reaches the device (`enhet`); a turn that never gets sound is
written at the next turn with `utan_ljud`.
"""
import logging
import time

logger = logging.getLogger(__name__)

STEG = ("turslut", "stt", "comms", "verktyg", "modell", "enhet")


class TurnTider:
    def __init__(self, device_id: str, klocka=time.monotonic):
        self.device_id = device_id
        self._klocka = klocka
        self.t = None

    def start(self, tystnad_s: float) -> None:
        """A turn has ended (the local detector said so); `tystnad_s` is the silence it waited for."""
        if self.t is not None:
            self._skriv(utan_ljud=True)
        nu = self._klocka()
        self.t = {"talets_slut": nu - tystnad_s, "turslut": nu}

    def mark(self, namn: str) -> None:
        if self.t is None or namn in self.t:
            return
        self.t[namn] = self._klocka()
        if namn == "enhet":
            self._skriv()

    def _skriv(self, utan_ljud: bool = False) -> None:
        t0 = self.t["talets_slut"]
        delar = [f"{s}={round((self.t[s] - t0) * 1000)}" for s in STEG if s in self.t]
        logger.info(f"⏱ tider {self.device_id} " + " ".join(delar) + (" utan_ljud" if utan_ljud else ""))
        self.t = None
