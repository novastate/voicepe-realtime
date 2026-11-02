"""US-411: optional shared secret on the device websocket, so only Comms can open it.

Unset `WS_NYCKEL` = no check (as before). Set = the upgrade must carry the same value in `X-Raawr-WS-Nyckel`, else the
socket is closed with 1008 before it is accepted. The value is never logged.
"""
import hmac
import os

from loguru import logger

HEADER = "x-raawr-ws-nyckel"


async def avvisa_utan_nyckel(websocket) -> bool:
    """True if the connection was refused (and closed); False if it may go on."""
    nyckel = os.environ.get("WS_NYCKEL", "")
    if not nyckel:
        return False
    given = websocket.headers.get(HEADER, "") if getattr(websocket, "headers", None) is not None else ""
    if given and hmac.compare_digest(given.encode(), nyckel.encode()):
        return False
    client = getattr(websocket, "client", None)
    logger.warning(f"🛡️ device websocket refused: missing or wrong key (from {getattr(client, 'host', '?')})")
    await websocket.close(code=1008)
    return True
