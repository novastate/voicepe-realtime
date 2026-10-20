"""G's review 2026-10-18, finding 20: HassTurnOn/HassTurnOff with only an area or floor switch everything there.

HA acts on every entity in the area that supports the intent when nothing narrows it (a light, a switch, a fan, a cover:
"open"). The tool schema requires no field, so the model could send {area}. The tool layer now refuses it and says how to
narrow it; a call that names a domain, a name or a device class is untouched.
"""
import pytest

from app import bana0
from tests.test_ha_misslyckades import _call, _service


@pytest.mark.parametrize("namn,arg", [("intent__HassTurnOn", {"area": "Kontor"}), ("HassTurnOff", {"area": "Kök"}),
                                      ("intent__HassTurnOff", {"floor": "Övervåningen"}),
                                      ("rost__anvand_verktyg", {"namn": "HassTurnOn", "argument": {"area": "Kontor"}}),
                                      ("intent__HassTurnOn", {"area": "Kontor", "domain": []}),
                                      ("intent__HassTurnOn", {"area": "Kontor", "name": "  "}),
                                      ("intent__HassTurnOn", {"area": "Kontor", "domain": [""]}),
                                      ("intent__HassTurnOn", {"area": "Kontor", "name": ["", " "]}),
                                      ("rost__anvand_verktyg", {"namn": "HassTurnOn", "argument": '{"area": "Kontor"}'})])
def test_ett_omrade_utan_doman_nekas(namn, arg):
    assert bana0.omrade_utan_doman(namn, arg) is not None


@pytest.mark.parametrize("namn,arg", [("intent__HassTurnOn", {"area": "Kontor", "domain": ["light"]}),
                                      ("HassTurnOff", {"area": "Kök", "device_class": ["outlet"]}),
                                      ("intent__HassTurnOn", {"name": "Fönsterlampan"}),
                                      ("intent__HassTurnOn", {"area": "Kontor", "name": "Skrivbordslampan"}),
                                      ("rost__anvand_verktyg", {"namn": "HassTurnOn", "argument": {"area": "Kontor", "domain": ["light"]}}),
                                      ("intent__GetLiveContext", {"area": "Kontor"}),
                                      ("intent__HassLightSet", {"area": "Kontor", "brightness": 50}),
                                      ("styr", {"vad": "kontoret"}),
                                      ("intent__HassTurnOn", {}),
                                      ("rost__anvand_verktyg", {"namn": "HassTurnOn", "argument": '{"area": "Kontor", "domain": ["light"]}'})])
def test_ett_smalt_eller_annat_anrop_ar_orort(namn, arg):
    assert bana0.omrade_utan_doman(namn, arg) is None


@pytest.mark.asyncio
async def test_verktygslagret_kor_inte_ett_omrade_utan_doman():
    service = _service()
    ran = []

    async def tool(params):
        ran.append(params.function_name)
        await params.result_callback("ok")

    service.register_function("intent__HassTurnOn", tool)
    got = await _call(service, "intent__HassTurnOn", {"area": "Kontor"})
    assert ran == [] and "EVERYTHING" in str(got)
    await _call(service, "intent__HassTurnOn", {"area": "Kontor", "domain": ["light"]})
    assert ran == ["intent__HassTurnOn"]
