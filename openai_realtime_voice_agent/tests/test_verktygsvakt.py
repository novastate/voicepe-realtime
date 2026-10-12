"""US-060: his own voice coming back through the mic must not run a house tool (Kontoret 2026-10-12 20:29: the loop
made Björn call HassTurnOn on his own question 'Vilken dörr vill du låsa?')."""
import pytest

from app import eko_vakt
from tests.test_ha_misslyckades import _call, _service

SVAR = "Vilken dörr vill du låsa? Är det hallen du vill att jag ska titta på?"


def test_his_own_words_back_are_caught_but_a_short_real_order_is_not():
    assert eko_vakt.vakt_verktyg("Vilken dörr vill du låsa?", SVAR, None) is not None
    assert eko_vakt.vakt_verktyg("Är det hallen du vill", SVAR, 30.0) is not None
    assert eko_vakt.vakt_verktyg("tänd lampan", SVAR, None) is None            # too few words: never an echo
    assert eko_vakt.vakt_verktyg("släck lampan i köket tack", SVAR, 2.0) is None  # other words: a person
    assert eko_vakt.vakt_verktyg("Vilken dörr vill du låsa?", "", None) is None   # he has said nothing: not an echo


def test_unclear_speech_right_after_an_interruption_is_held_but_later_it_is_not():
    assert eko_vakt.vakt_verktyg("", SVAR, 3.0) is not None
    assert eko_vakt.vakt_verktyg("", SVAR, 30.0) is None
    assert eko_vakt.vakt_verktyg("", SVAR, None) is None
    assert eko_vakt.vakt_verktyg("släck lampan", SVAR, 3.0) is None               # clear words of its own: let through


@pytest.mark.asyncio
async def test_a_tool_does_not_run_when_the_guard_says_no_and_runs_when_it_does_not():
    service = _service()
    ran = []

    async def tool(params):
        ran.append(params.function_name)
        await params.result_callback("ok")

    service.register_function("intent__HassTurnOn", tool)
    service.verktygsvakt = lambda: "ignored: echo"
    got = await _call(service, "intent__HassTurnOn", {"area": "hall"})
    assert ran == [] and "ignored" in str(got)
    service.verktygsvakt = lambda: None
    await _call(service, "intent__HassTurnOn", {"area": "hall"})
    assert ran == ["intent__HassTurnOn"]
