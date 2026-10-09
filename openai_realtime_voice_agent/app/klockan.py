"""The clock without the cloud (raawr US-032 AC-7).

"Vad är klockan" is answered from the local STT's text and the agent's own
clock, in the engine's voice from clips rendered ahead -- no cloud engine is
asked, so it answers offline and well under 2 s.

Said the way a Swede says it, and in Björn's voice (the owner 2026-10-07:
"som en svensk gör samt behåller sin karaktär"): "Hon är tjugo över fem."
to the nearest five minutes on a twelve-hour dial, now and then followed by a
dry line for the time of day. Every clip is a whole phrase: Gemini TTS gives
no audio for a bare number ("tjugo", "noll två", 2026-10-07).
"""
import os
import re
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Stockholm")

_TIMMAR = ["tolv", "ett", "två", "tre", "fyra", "fem", "sex", "sju", "åtta", "nio", "tio", "elva"]
# Five-minute slots: (words, whether the hour named is the next one).
_LAGEN = [
    ("{}", False), ("fem över {}", False), ("tio över {}", False),
    ("kvart över {}", False), ("tjugo över {}", False), ("fem i halv {}", True),
    ("halv {}", True), ("fem över halv {}", True), ("tjugo i {}", True),
    ("kvart i {}", True), ("tio i {}", True), ("fem i {}", True),
]
# Björn's dry line, by the hour it is said (first hour of each span).
_KOMMENTARER = [
    (0, "Gå och lägg dig, för fan."),
    (5, "Kaffe först, sen allt annat."),
    (9, "Dagen rullar på."),
    (17, "Snart dags att käka."),
    (22, "Sängdags snart, va?"),
]

# The question, not just the word: "ställ klockan på sju" or "väck mig klockan
# sex" are orders for the model. Whisper writes "var" for "vad" (2026-10-02).
_FRAGA = re.compile(
    r"^(?:(?:hej )?björn )?(?:"
    r"(?:vad|var|va|hur mycket) (?:är|e) (?:klockan|tiden)"
    r"|(?:vet du |kan du säga )?(?:vad|var|va|hur mycket) klockan är"
    r"|hur dags är det"
    r")(?: just)?(?: nu)?(?: då)?(?: björn)?$"
)

# Whisper hears Henrik badly ("Böda klockan", "Börda klockan", "Hur mycket klockan",
# live 2026-10-09). A short phrase that ENDS in "klockan" is the question when it is
# exactly two words (a garbled first word + "klockan") or starts with a question word;
# anything else short that ends there goes to the model as before ("larma mig klockan"
# is an order whose time Whisper dropped), and so does any order word in it (A's review).
_SLUT = {"nu", "då", "just", "björn", "snälla"}
_FRAGEORD = {"vad", "var", "va", "hur", "vet", "har", "säg", "kan", "e", "är"}
_ORDER = {"ställ", "ställa", "väck", "väcka", "väckarklocka", "sätt", "sätta", "timer", "larm", "larma",
          "alarm", "vakna", "ring", "ringa", "påminn", "påminna", "boka", "starta", "stoppa", "pausa",
          "spela", "kör", "öppna", "stäng", "byt", "ändra", "schemalägg", "tänd", "släck", "mig", "oss",
          "på", "vid", "kl", "efter", "runt", "till", "från", "mellan", "klockan"}


def _kort_klockfraga(ord_: list) -> bool:
    while ord_ and ord_[-1] in _SLUT:
        ord_ = ord_[:-1]
    if ord_[:2] == ["hej", "björn"]:
        ord_ = ord_[2:]
    if not (2 <= len(ord_) <= 5 and ord_[-1] == "klockan") or set(ord_[:-1]) & _ORDER:
        return False
    return len(ord_) == 2 or ord_[0] in _FRAGEORD


def klipp_paa() -> bool:
    """The pre-recorded clock answer (AC-7). OFF by default since Henrik's decision
    2026-10-09: fast but static, "not Björn". KLOCKA_KLIPP=1 turns it on again.
    Off: a clock question goes to the model like any other, and no clip is warmed."""
    return os.environ.get("KLOCKA_KLIPP", "0") == "1"


_sa_senast = {"kommentar": True}  # the last answer had a line; the next does not


def ar_klockfraga(text: Optional[str]) -> bool:
    if not text:
        return False
    ren = re.sub(r"[^\wåäö ]+", " ", text.lower())
    ord_ = ren.split()
    return bool(_FRAGA.match(" ".join(ord_))) or _kort_klockfraga(ord_)


def _lage(nu: datetime) -> tuple[int, int]:
    """(slot 0-11, hour 0-23 of that slot's start), to the nearest five minutes."""
    totalt = (nu.hour * 60 + nu.minute + 2) // 5 * 5  # 17:58 -> 18:00
    timme, minut = divmod(totalt % (24 * 60), 60)
    return minut // 5, timme


def tid(lage: int, timme: int) -> str:
    """'Hon är tjugo över fem.' for slot `lage` of hour `timme` (0-23)."""
    ord_, nasta = _LAGEN[lage]
    return f"Hon är {ord_.format(_TIMMAR[(timme + nasta) % 12])}."


def kommentar(timme: int) -> str:
    return [t for h, t in _KOMMENTARER if h <= timme][-1]


def delar(nu: Optional[datetime] = None, med_kommentar: Optional[bool] = None) -> list[str]:
    """The clips that say the time now, in order. Every other answer has a line."""
    nu = (nu or datetime.now(TZ)).astimezone(TZ)
    if med_kommentar is None:
        med_kommentar = not _sa_senast["kommentar"]
        _sa_senast["kommentar"] = med_kommentar
    lage, timme = _lage(nu)
    return [tid(lage, timme)] + ([kommentar(nu.hour)] if med_kommentar else [])


def alla_delar(nu: Optional[datetime] = None) -> list[str]:
    """Every clip delar() can ask for, the soonest needed first: the lines, then
    the dial from now on round. Rendering takes days (Gemini TTS: 100 a day)."""
    nu = (nu or datetime.now(TZ)).astimezone(TZ)
    lage, timme = _lage(nu)
    start = timme * 12 + lage
    ordning, sett = [t for _, t in _KOMMENTARER], set()
    for i in range(24 * 12):
        steg = (start + i) % (24 * 12)
        text = tid(steg % 12, steg // 12)
        if text not in sett:
            sett.add(text)
            ordning.append(text)
    return ordning
