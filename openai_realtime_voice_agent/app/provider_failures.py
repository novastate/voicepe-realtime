"""What kind of broken is this, and what is the useful answer?

An engine can fail in ways that mean very different things. Out of money is
permanent until someone pays; a dropped socket is over in a second. Answering
both the same way is how a house ends up either silent or paying on the wrong
account for a day.

The classes below are ordered by how sure we are. MONEY and AUTH are checked
first because their phrasings often also contain the words a transient error
uses -- OpenAI delivers a spent quota as a 429 that reads like a rate limit.
"""
from enum import Enum


class Failure(str, Enum):
    """What to do about an error, not what caused it."""

    MONEY = "money"          # switch engine now; retrying cannot help
    AUTH = "auth"            # switch engine now; this engine will not accept us
    TRANSIENT = "transient"  # try once more on this engine, then switch
    APP = "app"              # our own fault; never switch


# Checked in this order. The first list that matches wins, so the permanent
# failures get to claim a message before the transient patterns see it.
_MONEY = (
    "insufficient_quota",
    "exceeded your current quota",
    "billing hard limit",
    "billing_hard_limit_reached",
    "resource_exhausted",
    "quota exceeded",
    "402",
)
_AUTH = (
    "incorrect api key",
    "invalid_api_key",
    "api key not valid",
    "401",
    "403",
    "unauthorized",
    "permission_denied",
    "unauthenticated",
    # A retired preview model refuses the session before it opens. Nothing on
    # our side can fix it at runtime, so it belongs with the permanent ones.
    "is not found for api version",
    "not supported for bidigeneratecontent",
)
_TRANSIENT = (
    "rate limit",
    "keepalive ping timeout",
    "going away",
    "no close frame",
    "connectionclosed",
    "connection is closed",
    "1011",
    "1001",
    "1006",
    "realtime receive loop",
    "timeout",
    "500",
    "502",
    "503",
    "504",
    "internal error",
)

# Names of the tools this add-on runs itself. An error carrying one of these is
# ours, and swapping engines would only hide it.
_OURS = (
    "play_media",
    "search_home",
    "set_timer",
    "cancel_timer",
    "list_timers",
    "remember",
    "forget",
    "list_memories",
    "voice_enrollment",
    "mark_false_wake",
    "ask_openclaw",
    "recall_memory",
    "web_search",
    "supervisor_token",
)


def classify(message: str) -> Failure:
    """Decide what to do about one error message.

    Args:
        message: The text of the error, from an exception or an ErrorFrame.

    Returns:
        The Failure class that says what the caller should do.
    """
    text = (message or "").lower()
    if not text:
        return Failure.APP
    if any(name in text for name in _OURS):
        return Failure.APP
    if any(marker in text for marker in _MONEY):
        return Failure.MONEY
    if any(marker in text for marker in _AUTH):
        return Failure.AUTH
    if any(marker in text for marker in _TRANSIENT):
        return Failure.TRANSIENT
    return Failure.APP
