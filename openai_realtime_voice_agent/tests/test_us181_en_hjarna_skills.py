"""US-181: the visible catalogue follows the active Core road, including cache."""
from unittest.mock import AsyncMock

import pytest
from app import core_strom, skills

KNOWLEDGE = ["webb", "sport", "minne", "research"]
HOUSE = ["hus", "musik", "listor", "kalender", "anteckningar"]


@pytest.mark.asyncio
async def test_knowledge_removed_from_catalogue_and_cached_prompt_only_with_core(monkeypatch):
    monkeypatch.setenv("ROST_SKILLS", "c")
    monkeypatch.setenv("EN_HJARNA", "0")
    monkeypatch.setattr(core_strom, "strom_url", lambda: "https://core/fraga")
    katalog = skills.Katalog()
    katalog._hamta = AsyncMock(return_value=[(n, n + " beskrivning") for n in KNOWLEDGE + HOUSE])
    original = await katalog.text("office")
    assert all("- " + n + ":" in original for n in KNOWLEDGE + HOUSE)
    monkeypatch.setenv("EN_HJARNA", "1")
    katalog._hamta.return_value = None
    text = await katalog.text("office")
    assert all("- " + n + ":" not in text for n in KNOWLEDGE)
    assert all("- " + n + ":" in text for n in HOUSE)
    monkeypatch.setattr(core_strom, "strom_url", lambda: "")
    assert await katalog.text("office") == original


def test_knowledge_tool_names_and_namespaces_are_filtered():
    names = ["web_search", "comms__recall_memory", "ask_openclaw", "webb", "sport", "minne", "research", "fraga_core", "styr"]
    assert [t["name"] for t in core_strom.utan_kunskapsverktyg([{"name": n} for n in names])] == ["fraga_core", "styr"]
