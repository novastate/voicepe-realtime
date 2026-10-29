"""US-254 (raawr): ett nej från Comms provas genom röstagentens verktygslager, inte bara genom pipecats _call_tool.

Handlern registreras som i drift (`service.register_function`, alltså med verktygslagrets egna omslag: tid, journal, vakter) och
anropas som en modell anropar den. Sessionen står i för Comms: den nekar låset med `isError` och noterar varje anrop som
faktiskt utförs. Ett nej betyder att inget utförs och att modellen får meningen; ett ja utför exakt en gång, så provet inte
är grönt av att inget någonsin anropas.
"""
import types
from types import SimpleNamespace

import pytest

from app.providers import OPENAI
from test_tool_loop_guard import _service

NEJ = "Det får jag inte göra härifrån."


class _Comms:
    """Comms stand-in: låset nekas, allt annat utförs och noteras."""

    def __init__(self, neka_lasning):
        self.neka_lasning = neka_lasning
        self.utfort = []

    async def call_tool(self, namn, arguments=None):
        mal = (arguments or {}).get("namn")
        if mal == "lock_unlock" and self.neka_lasning:
            return types.SimpleNamespace(isError=True, content=[types.SimpleNamespace(text=NEJ)])
        self.utfort.append(mal)
        return types.SimpleNamespace(isError=False, content=[types.SimpleNamespace(text="Klart.")])


async def _anrop(neka_lasning, namn):
    from pipecat.services.mcp_service import MCPClient

    comms = _Comms(neka_lasning)
    klient = MCPClient.__new__(MCPClient)
    klient._tools_output_filters = {}

    async def handler(params):
        await klient._call_tool(comms, "anvand_verktyg", params.arguments, params.result_callback)

    service = _service(OPENAI)
    service.register_function("comms__anvand_verktyg", handler)
    fick = []

    async def result_callback(result, **kwargs):
        fick.append(result)

    await service._functions["comms__anvand_verktyg"].handler(SimpleNamespace(
        function_name="comms__anvand_verktyg", arguments={"namn": namn, "argument": {}}, result_callback=result_callback))
    return comms, fick


@pytest.mark.asyncio
async def test_ac1_ett_nej_fran_comms_kor_aldrig_handlern():
    comms, fick = await _anrop(neka_lasning=True, namn="lock_unlock")
    assert comms.utfort == []
    assert fick == [NEJ]


@pytest.mark.asyncio
async def test_ac2_ett_ja_fran_comms_utfor_exakt_en_gang():
    comms, fick = await _anrop(neka_lasning=False, namn="lock_unlock")
    assert comms.utfort == ["lock_unlock"]
    assert fick == ["Klart."]
