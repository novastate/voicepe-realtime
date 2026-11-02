"""US-411: with WS_NYCKEL set, only a caller that presents it (Comms) may open the device socket."""
import asyncio

import pytest

from app import ws_nyckel


class FakeWS:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.closed = None

    async def close(self, code=1000):
        self.closed = code


def test_ac1_utan_inställning_släpps_alla_in(monkeypatch):
    monkeypatch.delenv("WS_NYCKEL", raising=False)
    ws = FakeWS()
    assert asyncio.run(ws_nyckel.avvisa_utan_nyckel(ws)) is False
    assert ws.closed is None


def test_ac2_satt_nyckel_nekar_saknad_och_fel(monkeypatch):
    monkeypatch.setenv("WS_NYCKEL", "hemlig-nyckel-123")
    for headers in ({}, {"x-raawr-ws-nyckel": "fel"}, {"x-raawr-ws-nyckel": ""},
                    {"x-raawr-ws-nyckel": "hemlig-nyckel-12"}, {"authorization": "hemlig-nyckel-123"}):
        ws = FakeWS(headers)
        assert asyncio.run(ws_nyckel.avvisa_utan_nyckel(ws)) is True
        assert ws.closed == 1008


def test_ac3_ratt_nyckel_släpps_in(monkeypatch):
    monkeypatch.setenv("WS_NYCKEL", "hemlig-nyckel-123")
    ws = FakeWS({"x-raawr-ws-nyckel": "hemlig-nyckel-123"})
    assert asyncio.run(ws_nyckel.avvisa_utan_nyckel(ws)) is False
    assert ws.closed is None


def test_ac4_nyckeln_syns_aldrig_i_loggen(monkeypatch):
    from loguru import logger
    rader = []
    id_ = logger.add(rader.append, level="DEBUG")
    try:
        monkeypatch.setenv("WS_NYCKEL", "hemlig-nyckel-123")
        asyncio.run(ws_nyckel.avvisa_utan_nyckel(FakeWS({"x-raawr-ws-nyckel": "gissning-999"})))
    finally:
        logger.remove(id_)
    text = "".join(str(r) for r in rader)
    assert "refused" in text
    assert "hemlig-nyckel-123" not in text and "gissning-999" not in text
