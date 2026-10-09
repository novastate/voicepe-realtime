"""The language check on Gemini's own transcription (log only; kitchen 2026-10-09)."""
import logging
from types import SimpleNamespace

import pytest

from app import sprakkoll


@pytest.mark.parametrize("text", ["Ronzio a terraHej Björn, vad tycker du jag ska äta?", "Bonjour du, vad gör du?",
                                  "Ciao Björn come stai"])
def test_ett_annat_sprak_i_bojan_fangas(text):
    hit = sprakkoll.annat_sprak(text)
    assert hit is not None and len(hit[1]) <= sprakkoll.FORSTA_ORD


@pytest.mark.parametrize("text", ["Hej Björn, vad tycker du jag ska äta?", "Tänd lampan i köket", "Släck i kontoret.",
                                  "Vad är klockan", "ja", ""])
def test_svenska_och_for_korta_texter_fangas_inte(text):
    assert sprakkoll.annat_sprak(text) is None


@pytest.mark.asyncio
async def test_tjansten_loggar_bara_de_forsta_orden_en_gang_per_tur(caplog):
    from app.providers import gemini_live
    from app.providers.gemini_live import ResilientGeminiLiveService as S

    anrop = []

    async def bas(self, message):
        anrop.append(1)

    orig = gemini_live.GeminiLiveLLMService._handle_msg_input_transcription
    gemini_live.GeminiLiveLLMService._handle_msg_input_transcription = bas
    try:
        ns = S.__new__(S)
        ns._transkr_start, ns._transkr_kollad = "", False

        def frag(t):
            return SimpleNamespace(server_content=SimpleNamespace(input_transcription=SimpleNamespace(text=t)))

        with caplog.at_level(logging.WARNING):
            for t in ["Ronzio ", "a terra", "Hej Björn, ", "vad tycker du jag ska äta?"]:
                await ns._handle_msg_input_transcription(frag(t))
        assert anrop == [1, 1, 1, 1]  # always passed on
        rader = [r for r in caplog.records if "another language" in r.getMessage()]
        assert len(rader) == 1 and "ronzio a terrahej" in rader[0].getMessage()
        assert "vad tycker du" not in rader[0].getMessage()  # only the first words
    finally:
        gemini_live.GeminiLiveLLMService._handle_msg_input_transcription = orig
