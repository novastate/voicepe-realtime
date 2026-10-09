"""Does the model's own transcription of the start of a turn look like another language than Swedish?

Kitchen 2026-10-09: noise before the wake word came back as Italian ("Ronzio a terra") and the answer began in
Italian. This only decides whether to LOG, so a later case can be seen: it looks at the first few words of the
transcription and compares them with short lists of common words. Nothing is stored and no audio is kept; the
caller logs just those first words.
"""
import re
from typing import Optional, Tuple

FORSTA_ORD = 4
_SV = set("och att det som är jag du en ett på inte har vad hej kan till med för av om den var vi så men ska "
          "tänd släck lampan köket kontoret björn tack ja nej vem hur när varför här där nu bra gärna".split())
_ANDRA = {
    "italienska": set("il la lo gli le di che è non per una un sono come ciao ronzio terra buongiorno grazie "
                      "prego sì anche questo molto".split()),
    "franska": set("le les des est une que pas bonjour vous je merci oui non pour avec dans c'est salut".split()),
    "engelska": set("the and is to of you what this that with hello please okay yes are your play game put turn on music song listen tips".split()),
    "tyska": set("der die das und ist nicht ich ein eine guten morgen danke bitte ja nein sie wir".split()),
    "spanska": set("el los las que una por hola gracias buenos días sí qué cómo está".split()),
}


def forsta_orden(text: str, n: int = FORSTA_ORD) -> list:
    return re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", (text or "").lower())[:n]


def annat_sprak(text: str) -> Optional[Tuple[str, list]]:
    """(language name, the first words) when one of the FIRST TWO words belongs to another language's common
    words and is not Swedish; None for Swedish, unclear or too short a text (fewer than 2 words). It looks at
    the start only: that is where the answer language was decided wrongly."""
    ord_ = forsta_orden(text)
    if len(ord_) < 2:
        return None
    start = ord_[:2]
    traffar = [(n, sum(1 for o in start if o in lista and o not in _SV)) for n, lista in _ANDRA.items()]
    namn, antal = max(traffar, key=lambda x: x[1])
    return (namn, ord_) if antal >= 1 else None


def mest_annat(text: str) -> Optional[str]:
    """The language when a WHOLE text is mostly another one than Swedish and also starts in it: at least two of its
    words are common words of that language (not Swedish ones) and they outnumber the Swedish common words. For gating an action on what the
    model heard, where `annat_sprak` (start only, logging) is too loose."""
    ord_ = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", (text or "").lower())
    sv = sum(1 for o in ord_ if o in _SV)
    namn, antal = max(((n, sum(1 for o in ord_ if o in lista and o not in _SV)) for n, lista in _ANDRA.items()),
                      key=lambda x: x[1])
    # a Swedish request with an English title ("Spela Game of Thrones") starts in Swedish; a foreign turn starts in
    # the foreign language
    return namn if antal >= 2 and antal > sv and ord_[0] in _ANDRA[namn] else None
