"""US-206: a made-up tool name is logged as okänt, and the counter never drops a line."""
import json
import logging
import subprocess
import sys

import pytest

from app.providers import OPENAI
from test_tool_loop_guard import _service
from test_matvarden import rad, matvarden
from test_us203_verktyg_i_journalen import call


def _rader(caplog):
    return [r.message for r in caplog.records if r.message.startswith("🧰")]


@pytest.mark.asyncio
@pytest.mark.parametrize("namn", ["x" * 65, "web sok", "web;sok", "väder", "a\nb"])
async def test_ac1_konstigt_namn_loggas_som_okant(caplog, namn):
    caplog.set_level(logging.INFO)
    await call(_service(OPENAI), "comms__anvand_verktyg", {"namn": namn, "argument": {}})
    assert _rader(caplog) == ["🧰 verktyg okänt"]
    assert namn not in caplog.text


@pytest.mark.asyncio
async def test_ac2_vanliga_namn_loggas_som_forut(caplog):
    caplog.set_level(logging.INFO)
    s = _service(OPENAI)
    await call(s, "comms__anvand_verktyg", {"namn": "webbsok", "argument": {}})
    await call(s, "comms__las_skill", {"namn": "hus"})
    await call(s, "fraga_core", {"fraga": "x"})
    assert _rader(caplog) == ["🧰 verktyg webbsok", "🧰 skill hus", "🧰 verktyg fraga_core"]


def test_ac3_matvarden_raknar_okant_och_tappar_ingen_rad():
    logg = [
        rad("10:00:01,000", "🧭 kontoret: LISTENING -> THINKING (thinking: phase)"),
        rad("10:00:02,000", "🧰 verktyg okänt"),
        rad("10:00:03,000", "🧰 verktyg foo bar"),  # old line with a blank in the name
        rad("10:00:04,000", "🧰 skill okänt"),
        rad("10:00:05,000", "🧰 verktyg webbsok"),
    ]
    ut = subprocess.run([sys.executable, matvarden.__file__, "--json"], input="\n".join(logg),
                        text=True, capture_output=True, check=True).stdout
    d = json.loads(ut)["kontoret"]
    assert d["anvand_verktyg"] == {"okänt": 2, "webbsok": 1}
    assert d["las_skill"] == {"okänt": 1}
