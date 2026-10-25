"""US-203: tool choices in the journal, without payloads, and daily room counts."""
import json
import logging
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app.providers import OPENAI
from test_tool_loop_guard import _service
from test_matvarden import LOGG, matvarden, rad


async def call(service, name, arguments):
    async def handler(params):
        await params.result_callback({"result": "hemligt svar"})

    async def result_callback(result, **kwargs):
        pass

    service.register_function(name, handler)
    await service._functions[name].handler(SimpleNamespace(
        function_name=name, arguments=arguments, result_callback=result_callback))


@pytest.mark.asyncio
async def test_ac1_webbsok_loggas_utan_argument_eller_svar(caplog, monkeypatch):
    monkeypatch.setenv("EN_HJARNA", "0")
    caplog.set_level(logging.INFO)
    args = {"namn": "webbsok", "argument": {"fraga": "hemlig sokning"}}
    await call(_service(OPENAI), "comms__anvand_verktyg", args)
    assert [r.message for r in caplog.records if r.message.startswith("🧰")] == [
        "🧰 verktyg webbsok"]
    assert len([r for r in caplog.records if r.message.startswith("⏱ tool comms__anvand_verktyg ")
                and r.message.endswith(" ok")]) == 1
    assert "hemlig" not in caplog.text
    assert "argument" not in caplog.text


@pytest.mark.asyncio
async def test_ac2_hus_skill_och_fraga_core_loggas(caplog):
    caplog.set_level(logging.INFO)
    service = _service(OPENAI)
    await call(service, "comms__las_skill", {"namn": "hus"})
    await call(service, "fraga_core", {"fraga": "hemlig fraga"})
    assert [r.message for r in caplog.records if r.message.startswith("🧰")] == [
        "🧰 skill hus", "🧰 verktyg fraga_core"]
    assert "hemlig" not in caplog.text


def test_ac3_matvarden_raknar_per_rum_och_gammal_journal_ger_noll():
    logg = [
        rad("10:00:00,000", "🧰 verktyg fraga_core"),  # no room yet
        rad("10:00:01,000", "🧭 kontoret: LISTENING -> THINKING (thinking: phase)"),
        rad("10:00:02,000", "🧰 verktyg fraga_core"),
        rad("10:00:03,000", "🧰 verktyg fraga_core"),
        rad("10:00:04,000", "🧰 skill hus"),
        rad("10:00:05,000", "🧰 verktyg webbsok"),
        rad("10:00:06,000", "⏱ tool comms__anvand_verktyg 10 ok"),
        rad("10:00:07,000", "🧭 koket: LISTENING -> THINKING (thinking: phase)"),
        rad("10:00:08,000", "🧰 skill musik"),
        rad("10:00:09,000", "🧰 verktyg styr"),
    ]
    for journal in (logg, LOGG):
        result = subprocess.run([sys.executable, matvarden.__file__, "--json"],
                                input="\n".join(journal), text=True, capture_output=True, check=True)
        data = json.loads(result.stdout)
        expected = ({"fraga_core": 2, "las_skill": {"hus": 1}, "anvand_verktyg": {"webbsok": 1}}
                    if journal is logg else {"fraga_core": 0, "las_skill": {}, "anvand_verktyg": {}})
        assert {key: data["kontoret"][key] for key in expected} == expected
        if journal is logg:
            assert data["koket"]["fraga_core"] == 0
            assert data["koket"]["las_skill"] == {"musik": 1}
            assert data["koket"]["anvand_verktyg"] == {"styr": 1}
        text = subprocess.run([sys.executable, matvarden.__file__], input="\n".join(journal),
                              text=True, capture_output=True, check=True).stdout
        assert f"fraga_core: {expected['fraga_core']}" in text
        assert f"las_skill: {expected['las_skill']}" in text
        assert f"anvand_verktyg: {expected['anvand_verktyg']}" in text
