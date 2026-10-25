"""US-215: calls without a usable name are counted as okänt, and the unknown-tool line is capped."""
import logging
from types import SimpleNamespace

import pytest

from app.providers import OPENAI
from test_tool_loop_guard import _service
from test_us203_verktyg_i_journalen import call


def _rader(caplog):
    return [r.message for r in caplog.records if r.message.startswith("🧰")]


@pytest.mark.asyncio
@pytest.mark.parametrize("funktion,args,forvantat", [
    ("comms__anvand_verktyg", {"argument": {}}, "🧰 verktyg okänt"),
    ("comms__anvand_verktyg", {"namn": "", "argument": {}}, "🧰 verktyg okänt"),
    ("comms__las_skill", {"namn": "   "}, "🧰 skill okänt"),
    ("comms__las_skill", {"namn": 7}, "🧰 skill okänt"),
])
async def test_ac1_anrop_utan_anvandbart_namn_loggas_som_okant(caplog, funktion, args, forvantat):
    caplog.set_level(logging.INFO)
    await call(_service(OPENAI), funktion, args)
    assert _rader(caplog) == [forvantat]


@pytest.mark.asyncio
async def test_ac2_okant_verktyg_har_tak_men_far_svar(caplog):
    caplog.set_level(logging.INFO)
    svar = []

    async def result_callback(result, **kwargs):
        svar.append(result)

    lang = "z" * 200
    handler = _service(OPENAI)._functions[None].handler
    await handler(SimpleNamespace(function_name=lang, arguments={}, result_callback=result_callback))
    rad = [r.message for r in caplog.records if r.message.startswith("okänt verktyg")]
    assert len(rad) == 1 and "z" * 65 not in rad[0] and "z" * 64 in rad[0]
    assert svar and "okänt verktyg" in svar[0]["error"]


@pytest.mark.asyncio
async def test_ac3_vanliga_namn_som_forut(caplog):
    caplog.set_level(logging.INFO)
    s = _service(OPENAI)
    await call(s, "comms__anvand_verktyg", {"namn": "webbsok", "argument": {}})
    await call(s, "comms__las_skill", {"namn": "hus"})
    await call(s, "fraga_core", {"fraga": "x"})
    assert _rader(caplog) == ["🧰 verktyg webbsok", "🧰 skill hus", "🧰 verktyg fraga_core"]
