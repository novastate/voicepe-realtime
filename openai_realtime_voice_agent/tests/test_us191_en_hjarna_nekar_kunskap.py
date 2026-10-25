"""US-191: with EN_HJARNA the voice cannot reach knowledge through comms' las_skill / anvand_verktyg."""
import pytest

from tests.test_fraga_inte_order import _call, _service


async def _kor(monkeypatch, en_hjarna, namn, arg):
    monkeypatch.setenv("EN_HJARNA", en_hjarna)
    monkeypatch.setattr("app.core_strom.strom_url", lambda: "https://core/fraga")
    service = _service()
    ran = []

    async def tool(params):
        ran.append(params.function_name)
        await params.result_callback({"result": "ok"})

    service.register_function(namn, tool)
    got = await _call(service, namn, arg)
    return ran, str(got)


@pytest.mark.asyncio
@pytest.mark.parametrize("skill", ["webb", "sport", "minne", "research"])
async def test_ac1_kunskapsskill_nekas_vid_namn(monkeypatch, skill):
    ran, got = await _kor(monkeypatch, "1", "las_skill", {"namn": skill})
    assert ran == [] and "fraga_core" in got


@pytest.mark.asyncio
@pytest.mark.parametrize("verktyg", ["webbsok", "webbsida", "minne_las", "minne_skriv", "research_bestall", "research_hamta"])
async def test_ac2_kunskapsverktyg_nekas_via_anvand_verktyg(monkeypatch, verktyg):
    ran, got = await _kor(monkeypatch, "1", "anvand_verktyg", {"namn": verktyg, "argument": {}})
    assert ran == [] and "fraga_core" in got


@pytest.mark.asyncio
async def test_ac3_huset_och_en_hjarna_av_ar_oforandrade(monkeypatch):
    assert (await _kor(monkeypatch, "1", "las_skill", {"namn": "hus"}))[0] == ["las_skill"]
    assert (await _kor(monkeypatch, "1", "anvand_verktyg", {"namn": "styr", "argument": {}}))[0] == ["anvand_verktyg"]
    assert (await _kor(monkeypatch, "0", "las_skill", {"namn": "webb"}))[0] == ["las_skill"]
