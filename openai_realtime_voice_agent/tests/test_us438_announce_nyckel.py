"""US-438: the announce door compares its bearer token in constant time and never logs it.

The door (POST /announce on ANNOUNCE_PORT) makes a speaker talk, so its token is the
same kind of secret as US-411's WS_NYCKEL. A plain `!=` leaks how much of the token
matched through timing; a non-ASCII header must still give 401, never a crash.
"""
import asyncio
import hmac
import logging
from pathlib import Path
import sys

import pytest
from aiohttp import ClientSession

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import announce_http

TOKEN = "prov-token-us438-hemligt"


async def _with_server(fn):
    announce_http._recent.clear()
    announce_http._pending.clear()
    spoken = []

    async def announce(message, device_id):
        spoken.append((message, device_id))
        return True

    runner = await announce_http.start_announce_server(0, TOKEN, announce, lambda device_id: True)
    port = next(iter(runner.sites))._server.sockets[0].getsockname()[1]
    try:
        async with ClientSession() as session:
            return await fn(session, f"http://127.0.0.1:{port}/announce"), spoken
    finally:
        await runner.cleanup()


async def _post(session, url, auth):
    headers = {} if auth is None else {"Authorization": auth}
    async with session.post(url, json={"message": "Provvara US-438"}, headers=headers) as r:
        return r.status


def test_ac1_right_token_still_speaks():
    status, spoken = asyncio.run(_with_server(lambda s, u: _post(s, u, f"Bearer {TOKEN}")))
    assert status == 200
    assert spoken == [("Provvara US-438", None)]


@pytest.mark.parametrize("auth", [None, "", "Bearer", f"Bearer {TOKEN}x", f"Bearer {TOKEN[:-1]}", TOKEN,
                                  "Bearer prov-token-us438-hemligå".encode("utf-8").decode("latin-1")])
def test_ac2_wrong_or_missing_token_is_401_and_silent(auth):
    status, spoken = asyncio.run(_with_server(lambda s, u: _post(s, u, auth)))
    assert status == 401
    assert spoken == []


def test_ac2_comparison_is_constant_time(monkeypatch):
    calls = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(announce_http.hmac, "compare_digest", spy)
    status, _ = asyncio.run(_with_server(lambda s, u: _post(s, u, "Bearer fel")))
    assert status == 401
    assert calls, "the token must be compared with hmac.compare_digest"


def test_ac3_token_never_logged(caplog):
    caplog.set_level(logging.DEBUG)

    async def both(s, u):
        return (await _post(s, u, f"Bearer {TOKEN}"), await _post(s, u, "Bearer fel-nyckel-us438"))

    (ok, nej), _ = asyncio.run(_with_server(both))
    assert (ok, nej) == (200, 401)
    assert TOKEN not in caplog.text
    assert "fel-nyckel-us438" not in caplog.text
    assert any("refused" in r.getMessage() for r in caplog.records), "a refusal is logged, without the value"
