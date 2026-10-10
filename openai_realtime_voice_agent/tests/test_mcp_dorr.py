"""comms' own MCP door as a dynamic second tool source (raawr US-021, spår A's request).

The agent holds no list of the door's tools: `tools/list` is asked each time a session is built, so a change in comms
shows on the next wake with no change here.
"""
import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from app import mcp_service
from tests.test_provider_selection import _bare_app


def _schema(*namn):
    return ToolsSchema(standard_tools=[FunctionSchema(name=n, description=f"{n} d", properties={}, required=[])
                                       for n in namn])


class _Klient:
    def __init__(self, *scheman):
        self.scheman, self.registrerat, self.anrop = list(scheman), [], 0

    async def get_tools_schema(self):
        self.anrop += 1
        s = self.scheman[min(self.anrop - 1, len(self.scheman) - 1)]
        if isinstance(s, Exception):
            raise s
        return s

    async def register_tools_schema(self, schema, service):
        self.registrerat.append([f.name for f in schema.standard_tools])


def _app(ha, dorr=None):
    app = _bare_app("gemini")
    app.mcp_client, app.dorr_client = ha, dorr
    return app


def test_dorren_ar_av_om_inget_sagts(monkeypatch):
    monkeypatch.setenv("HA_API_URL", "http://c:3500/kanal/rost/kontoret/api")
    monkeypatch.delenv("COMMS_MCP_DORR", raising=False)
    assert mcp_service.dorr_url() is None


@pytest.mark.parametrize("bas,url", [("http://c:3500/kanal/rost/kontoret/api", "http://c:3500/kanal/rost/kontoret/mcp"),
                                     ("http://c:3500/kanal/rost/kontoret/api/", "http://c:3500/kanal/rost/kontoret/mcp"),
                                     ("http://c:3500/annat", None)])
def test_dorrens_adress_hamtas_bredvid_api_inte_under_det(monkeypatch, bas, url):
    monkeypatch.setenv("COMMS_MCP_DORR", "1")
    monkeypatch.setenv("HA_API_URL", bas)
    monkeypatch.delenv("COMMS_MCP_URL", raising=False)
    assert mcp_service.dorr_url() == url
    monkeypatch.setenv("COMMS_MCP_URL", "http://x/mcp")
    assert mcp_service.dorr_url() == "http://x/mcp"


@pytest.mark.asyncio
async def test_listan_hamtas_av_dorren_varje_gang_och_en_andring_syns_direkt():
    ha = _Klient(_schema("HassTurnOn"))
    dorr = _Klient(_schema("lage_drift", "folj_upp"), _schema("lage_drift", "folj_upp", "ny_sak_i_comms"))
    app = _app(ha, dorr)
    forsta = await app._fetch_ha_tools_schema()
    assert [f.name for f in forsta.standard_tools] == ["HassTurnOn", "lage_drift", "folj_upp"]
    andra = await app._fetch_ha_tools_schema()  # comms got a tool; the agent was not touched
    assert "ny_sak_i_comms" in [f.name for f in andra.standard_tools] and dorr.anrop == 2


@pytest.mark.asyncio
async def test_utan_dorr_ar_det_som_forr():
    ha = _Klient(_schema("HassTurnOn"))
    out = await _app(ha, None)._fetch_ha_tools_schema()
    assert [f.name for f in out.standard_tools] == ["HassTurnOn"]


@pytest.mark.asyncio
async def test_en_dorr_som_inte_svarar_kostar_inte_havs_verktyg():
    ha = _Klient(_schema("HassTurnOn"))
    app = _app(ha, _Klient(RuntimeError("503")))
    out = await app._fetch_ha_tools_schema()
    assert [f.name for f in out.standard_tools] == ["HassTurnOn"] and not hasattr(out, "dorr_namn")


@pytest.mark.asyncio
async def test_samma_namn_i_bada_ger_havs():
    ha = _Klient(_schema("GetLiveContext"))
    app = _app(ha, _Klient(_schema("GetLiveContext", "lage_drift")))
    out = await app._fetch_ha_tools_schema()
    assert [f.name for f in out.standard_tools] == ["GetLiveContext", "lage_drift"]
    assert out.dorr_namn == frozenset({"lage_drift"})


@pytest.mark.asyncio
async def test_varje_verktyg_anropas_pa_klienten_det_kom_fran():
    ha, dorr = _Klient(_schema("HassTurnOn")), _Klient(_schema("lage_drift"))
    app = _app(ha, dorr)
    schema = await app._fetch_ha_tools_schema()
    await app._register_ha_handlers(object(), schema, "kontoret")
    assert ha.registrerat == [["HassTurnOn"]] and dorr.registrerat == [["lage_drift"]]


@pytest.mark.asyncio
async def test_dorrens_verktyg_erbjuds_modellen():
    app = _app(_Klient(_schema("HassTurnOn")), _Klient(_schema("lage_drift")))
    schema = await app._fetch_ha_tools_schema()
    namn = [t["name"] for t in app._ha_tool_definitions(schema)]
    assert "lage_drift" in namn and "HassTurnOn" in namn


class _Sen:
    def __init__(self, dröjer, schema):
        self.dröjer, self.schema = dröjer, schema

    async def get_tools_schema(self):
        import asyncio
        await asyncio.sleep(self.dröjer)
        return self.schema


@pytest.mark.asyncio
async def test_dorren_far_bara_den_tid_ha_lamnade_av_en_gemensam_tidsgrans():
    import time

    ha, dorr = _Sen(0.2, _schema("HassTurnOn")), _Sen(60, _schema("lage_drift"))  # HA slow, the door hangs
    t0 = time.monotonic()
    out = await _app(ha, dorr)._fetch_ha_tools_schema(timeout=0.3)
    assert time.monotonic() - t0 < 0.4  # was 0.5 s (0.2 + a full 0.3 again) before the shared deadline
    assert [f.name for f in out.standard_tools] == ["HassTurnOn"]


@pytest.mark.asyncio
async def test_tva_sessioner_far_var_sin_uppdelning():
    """The door names travel with the schema they came with (G's review of #50, finding b)."""
    app = _app(_Klient(_schema("HassTurnOn")), _Klient(_schema("lage_drift")))
    a = await app._fetch_ha_tools_schema()
    app.dorr_client = None
    b = await app._fetch_ha_tools_schema()  # a later fetch without the door
    ha, dorr = _Klient(), _Klient()
    app.mcp_client, app.dorr_client = ha, dorr
    await app._register_ha_handlers(object(), a, "kontoret")
    assert dorr.registrerat == [["lage_drift"]] and ha.registrerat == [["HassTurnOn"]]
