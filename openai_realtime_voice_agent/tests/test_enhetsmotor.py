"""One engine per device for a side-by-side day (DEVICE_PROVIDERS), with a way back when the pinned engine fails."""
import pytest

from app import providers
from app.provider_router import ProviderRouter
from app.providers import EnhetsRouter, device_provider


@pytest.fixture(autouse=True)
def _rent(monkeypatch):
    providers._ENHET_FEL.clear()
    monkeypatch.setenv("DEVICE_PROVIDERS", "koket=openai_live,kontoret=gemini,kallaren=finnsinte")


def test_en_enhet_kan_fastnaglas_pa_en_motor():
    assert device_provider("koket") == "openai_live" and device_provider("kontoret") == "gemini"


def test_okand_enhet_och_okand_motor_foljer_routern():
    assert device_provider("vardagsrummet") is None and device_provider("kallaren") is None


def test_utan_inställning_foljer_alla_routern(monkeypatch):
    monkeypatch.delenv("DEVICE_PROVIDERS")
    assert device_provider("koket") is None


@pytest.mark.asyncio
async def test_misslyckas_motorn_lamnar_enheten_till_routerns_motor_en_stund(monkeypatch):
    router = ProviderRouter("gemini", "xai", probe=lambda e: True)
    r = EnhetsRouter(router, "koket")
    assert await r.report_failure("openai_live", "socket closed") == "gemini"
    assert device_provider("koket") is None  # paused: the rebuild gets the router's engine, not the pin again
    monkeypatch.setenv("DEVICE_PROVIDER_PAUS_MIN", "0")
    assert device_provider("koket") == "openai_live"  # the pause is over
