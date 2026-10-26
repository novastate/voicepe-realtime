"""US-225: with EN_HJARNA the voice cannot call its own knowledge tools by name either."""
import pytest

from tests.test_us191_en_hjarna_nekar_kunskap import _kor


@pytest.mark.asyncio
async def test_ac1_web_search_nekas_vid_namn(monkeypatch):
    ran, got = await _kor(monkeypatch, "1", "web_search", {"query": "vem vann"})
    assert ran == [] and "fraga_core" in got


@pytest.mark.asyncio
@pytest.mark.parametrize("verktyg", ["ask_openclaw", "recall_memory", "mcp__comms__recall_memory"])
async def test_ac2_ovriga_kunskapsverktyg_nekas(monkeypatch, verktyg):
    ran, got = await _kor(monkeypatch, "1", verktyg, {"query": "x"})
    assert ran == [] and "fraga_core" in got


@pytest.mark.asyncio
async def test_ac3_en_hjarna_av_och_ovriga_verktyg_ar_oforandrade(monkeypatch):
    assert (await _kor(monkeypatch, "0", "web_search", {"query": "x"}))[0] == ["web_search"]
    assert (await _kor(monkeypatch, "1", "fraga_core", {"fraga": "x"}))[0] == ["fraga_core"]
    assert (await _kor(monkeypatch, "1", "las_skill", {"namn": "hus"}))[0] == ["las_skill"]
