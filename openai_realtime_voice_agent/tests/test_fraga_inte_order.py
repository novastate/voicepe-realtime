"""A question about the house is never an order (raawr US-098, live 2026-10-17 00:20 UTC+2).

'Är det tänt i kontoret?' -- heard right by bana 0's speech-to-text -- made the model call
HassTurnOn {area: kontoret, domain: light}: 'tänt' and 'tänd' sound alike, the engine wrote no transcript of
its own, and the prompt says to use the tools directly. Three lights went on. The tool layer now refuses a
switching tool when the turn was a question, and bana 0 no longer reads a question as a light order.
"""
import pytest

from app import bana0
from tests.test_ha_misslyckades import _call, _service


@pytest.mark.parametrize("text", ["Är det tänt i kontoret?", "Vilka lampor är tända?", "Är lampan på i köket?",
                                  "Lyser det i hallen?", "Har du tänt i köket?", "Hur många lampor är på?",
                                  # B on #91: no question word first
                                  "Kan du säga om det är tänt i kontoret?", "Tänt i kontoret?", "Brinner det i köket?",
                                  "Funkar lampan?", "Lampan på i köket?", "Släckt i hallen?",
                                  "Syns det ljus i köket?"])  # B on #92
def test_en_fraga_om_laget_ar_ingen_order(text):
    assert bana0.ar_fraga(text)
    assert bana0.verb_ur(text) is None


@pytest.mark.parametrize("text", ["Tänd det i kontoret", "Kan du tända lampan?", "Skulle du kunna släcka köket?",
                                  "Var snäll och tänd i hallen", "Är du snäll och släcker", "släck kontoret",
                                  "Tänd lampan?", "Kan du tända i köket?", "Släck i hallen, tack?", "Slå på lampan i köket?"])
def test_en_order_ar_ingen_fraga(text):
    assert not bana0.ar_fraga(text)
    assert bana0.verb_ur(text) is not None


@pytest.mark.parametrize("namn,arg", [("intent__HassTurnOn", {"area": "kontoret", "domain": ["light"]}),
                                      ("HassTurnOff", {"area": "kontoret"}), ("light__HassLightSet", {"brightness": 50}),
                                      ("styr", {"vad": "kontoret", "atgard": "pa"}),
                                      ("rost__anvand_verktyg", {"namn": "styr", "argument": {"vad": "kontoret"}})])
def test_ett_skrivande_verktyg_kors_inte_pa_en_fraga(namn, arg):
    assert bana0.fraga_inte_order("Är det tänt i kontoret?", namn, arg) is not None
    assert bana0.fraga_inte_order("Tänd det i kontoret", namn, arg) is None
    assert bana0.fraga_inte_order("", namn, arg) is None  # no local text: nothing to judge by


@pytest.mark.parametrize("namn,arg", [("homeassistant__GetLiveContext", {"area": "kontoret"}), ("hus_lage", {}),
                                      ("rost__anvand_verktyg", {"namn": "hus_lage", "argument": {}})])
def test_ett_lasande_verktyg_kors_aven_pa_en_fraga(namn, arg):
    # through anvand_verktyg comms' own word decides (raawr US-103): here it has said hus_lage only reads
    assert bana0.fraga_inte_order("Är det tänt i kontoret?", namn, arg, {"hus_lage": True}) is None


@pytest.mark.asyncio
async def test_verktygslagret_tander_inte_pa_en_fraga():
    service = _service()
    ran = []

    async def tool(params):
        ran.append(params.function_name)
        await params.result_callback("ok")

    service.register_function("intent__HassTurnOn", tool)
    service.fragevakt = lambda: "Är det tänt i kontoret?"
    got = await _call(service, "intent__HassTurnOn", {"area": "kontoret", "domain": ["light"]})
    assert ran == [] and "question" in str(got)
    service.fragevakt = lambda: "Tänd det i kontoret"
    await _call(service, "intent__HassTurnOn", {"area": "kontoret", "domain": ["light"]})
    assert ran == ["intent__HassTurnOn"]


@pytest.mark.asyncio
async def test_sessionen_far_fragevakten_med_det_bana0_horde():
    from tests.test_ha_tools_recovery import _connect
    from tests.test_provider_selection import _bare_app

    connection, service = await _connect(_bare_app("openai"))
    connection.bana0_heard = "Är det tänt i kontoret?"
    assert service.fragevakt() == "Är det tänt i kontoret?"
    assert bana0.fraga_inte_order(service.fragevakt(), "intent__HassTurnOn", {}) is not None


def test_en_begaran_om_ett_lage_ar_ingen_fraga():
    assert not bana0.ar_fraga("Kan du se till att det är tänt i köket?")
    assert bana0.fraga_inte_order("Kan du se till att det är tänt i köket?", "intent__HassTurnOn", {}) is None


def test_comms_markning_vinner_over_namnlistan():
    """raawr US-103: comms says per tool whether it only reads; the name list is the fallback."""
    las = {"styr": False, "hus_lage": True, "lage_drift": True, "kasta_in": False}
    fraga = "Är det tänt i kontoret?"
    assert bana0.fraga_inte_order(fraga, "kasta_in", {}, las) is not None          # comms: changes
    assert bana0.fraga_inte_order(fraga, "lage_drift", {}, las) is None            # comms: reads
    assert bana0.fraga_inte_order(fraga, "rost__anvand_verktyg", {"namn": "hus_lage"}, las) is None
    assert bana0.fraga_inte_order(fraga, "rost__anvand_verktyg", {"namn": "styr"}, las) is not None
    # not described: through anvand_verktyg it is taken to change something, a direct one goes by the name list
    assert bana0.fraga_inte_order(fraga, "rost__anvand_verktyg", {"namn": "okant"}, las) is not None
    assert bana0.fraga_inte_order(fraga, "intent__HassTurnOn", {}, las) is not None
    assert bana0.fraga_inte_order(fraga, "homeassistant__GetLiveContext", {}, las) is None


def test_skrivverktyg_ar_alltid_andrande_trots_readonlyhint():
    assert bana0.andrar("HassTurnOn", {}, {"HassTurnOn": True})
    assert bana0.andrar("intent__HassTurnOn", {}, {"HassTurnOn": True})
    assert bana0.andrar("rost__anvand_verktyg", {"namn": "styr", "argument": {}}, {"styr": True})
    assert not bana0.andrar("homeassistant__GetLiveContext", {}, {"GetLiveContext": True})


def test_las_skill_svaret_lar_vakten():
    from app import skills
    svar = '{"skill": "hus", "instruktion": "x", "verktyg": [{"namn": "hus_lage", "readOnlyHint": true}, ' \
           '{"namn": "styr", "readOnlyHint": false}, {"namn": "utan"}]}'
    assert skills.lasande_ur(svar) == {"hus_lage": True, "styr": False}
    assert skills.lasande_ur({"result": svar}) == {"hus_lage": True, "styr": False}
    assert skills.lasande_ur("Fel [Comms]: skillen finns inte") == {}


def test_las_skill_readonlyhint_for_skrivverktyg_ignoreras_och_varnas(caplog):
    from app import skills
    caplog.set_level("WARNING")
    svar = {"verktyg": [{"namn": "HassTurnOff", "readOnlyHint": True}, {"namn": "GetLiveContext", "readOnlyHint": True},
                        {"namn": "styr", "readOnlyHint": False}]}
    assert skills.lasande_ur(svar) == {"GetLiveContext": True, "styr": False}
    assert [r.message for r in caplog.records if r.levelname == "WARNING"] == [
        "las_skill påstod readOnlyHint för skrivande verktyg HassTurnOff: ignorerat"
    ]


@pytest.mark.asyncio
async def test_dorrens_tools_list_ger_markningen():
    import types
    from app.mcp_service import LasandeMCPClient

    def verktyg(namn, las):
        ann = None if las is None else types.SimpleNamespace(readOnlyHint=las)
        return types.SimpleNamespace(name=namn, description="d", inputSchema={"type": "object", "properties": {}},
                                     annotations=ann)

    class _Session:
        async def list_tools(self):
            return types.SimpleNamespace(tools=[verktyg("hus_lage", True), verktyg("styr", False), verktyg("x", None)])

    k = LasandeMCPClient.__new__(LasandeMCPClient)
    k._tools_filter = None
    schema = await k._list_tools_helper(_Session())
    assert k.las == {"hus_lage": True, "styr": False}
    assert [f.name for f in schema.standard_tools] == ["hus_lage", "styr", "x"]


@pytest.mark.asyncio
async def test_verktygslagret_lar_av_las_skill():
    service = _service()
    ran = []

    async def las_skill(params):
        await params.result_callback('{"verktyg": [{"namn": "hus_lage", "readOnlyHint": true}]}')

    async def anvand(params):
        ran.append(params.arguments["namn"])
        await params.result_callback("ok")

    service.register_function("rost__las_skill", las_skill)
    service.register_function("rost__anvand_verktyg", anvand)
    service.fragevakt = lambda: "Är det tänt i kontoret?"
    await _call(service, "rost__anvand_verktyg", {"namn": "hus_lage", "argument": {}})
    assert ran == []                        # not described yet: through anvand_verktyg it may change something
    await _call(service, "rost__las_skill", {"namn": "hus"})
    await _call(service, "rost__anvand_verktyg", {"namn": "hus_lage", "argument": {}})
    assert ran == ["hus_lage"]


@pytest.mark.asyncio
async def test_sessionen_far_dorrens_markning():
    from tests.test_ha_tools_recovery import _connect
    from tests.test_mcp_dorr import _Klient, _schema
    from tests.test_provider_selection import _bare_app

    app = _bare_app("openai")
    app.mcp_client = _Klient(_schema("HassTurnOn"))
    app.dorr_client = _Klient(_schema("lage_drift", "kasta_in"))
    app.dorr_client.las = {"lage_drift": True, "kasta_in": False}
    _, service = await _connect(app)
    assert service.lasande == {"lage_drift": True, "kasta_in": False}


def test_fraga_core_med_en_order_i_fragetexten_ar_andrande():
    """G on US-098: Core runs 'Tänd ...' in its fast path, so a read turn must not carry it through fraga_core."""
    fraga = "Är det tänt i kontoret?"
    assert bana0.fraga_inte_order(fraga, "fraga_core", {"question": "Tänd fönsterlampan i kontoret."}) is not None
    assert bana0.fraga_inte_order(fraga, "fraga_core", {"question": "Är lampan i kontoret tänd?"}) is None
    assert bana0.fraga_inte_order(fraga, "fraga_core", {"question": "Hur är läget i huset?"}) is None


def test_fraga_core_med_corens_andra_orderord_ar_andrande():
    """G on US-098, round 2: Core's fast path also runs 'godnatt' (off_all) and 'sätt igång köket' (on)."""
    fraga = "Är det tänt i kontoret?"
    for order in ("godnatt", "Sätt igång köket", "Starta kvällsscenen", "Öppna garaget"):
        assert bana0.fraga_inte_order(fraga, "fraga_core", {"question": order}) is not None, order
    for lasning in ("Hur är läget i huset?", "Vad minns du om mig?", "Vad har jag i kalendern i morgon?"):
        assert bana0.fraga_inte_order(fraga, "fraga_core", {"question": lasning}) is None, lasning


# US-253 (G on US-098, round 4; G on #133): a changing tool called before bana 0 has written the turn's text waits for BANA 0's text.

def _vakt(service, bana0, motor=lambda: ""):
    service.fragevakt_bana0 = lambda: bana0["v"]
    service.fragevakt = lambda: bana0["v"] or motor()


@pytest.mark.asyncio
async def test_ett_skrivande_verktyg_fore_texten_vantar_och_nekas_om_det_var_en_fraga():
    import asyncio
    text = {"v": ""}
    service = _service()
    ran = []

    async def tool(params):
        ran.append(params.function_name)
        await params.result_callback("ok")

    service.register_function("intent__HassTurnOn", tool)
    _vakt(service, text)
    asyncio.get_event_loop().call_later(0.1, lambda: text.update(v="Är det tänt i kontoret?"))  # STT lands 100 ms late
    got = await _call(service, "intent__HassTurnOn", {"area": "kontoret", "domain": ["light"]})
    assert ran == [] and "question" in str(got)


@pytest.mark.asyncio
async def test_en_order_som_kommer_fore_texten_kors_nar_texten_ar_en_order():
    import asyncio
    text = {"v": ""}
    service = _service()
    ran = []

    async def tool(params):
        ran.append(params.function_name)
        await params.result_callback("ok")

    service.register_function("intent__HassTurnOn", tool)
    _vakt(service, text)
    asyncio.get_event_loop().call_later(0.1, lambda: text.update(v="Tänd det i kontoret"))
    await _call(service, "intent__HassTurnOn", {"area": "kontoret", "domain": ["light"]})
    assert ran == ["intent__HassTurnOn"]


@pytest.mark.asyncio
async def test_motorns_vaxande_text_dömer_inte_for_tidigt():
    """G on #133: the engine's own transcript grows ('Tänd i kontoret' before 'Är det tänt i kontoret?'); the wait must
    judge bana 0's whole sentence, not the first crumb. And 'Var' (a crumb that reads as a question) must not deny the order
    'Var snäll och tänd i hallen'."""
    import asyncio
    for krummel, hel, ska_koras in (("Tänd i kontoret", "Är det tänt i kontoret?", False),
                                    ("Var", "Var snäll och tänd i hallen", True)):
        bana0_t = {"v": ""}
        motor = {"v": krummel}
        service = _service()
        ran = []

        async def tool(params):
            ran.append(params.function_name)
            await params.result_callback("ok")

        service.register_function("intent__HassTurnOn", tool)
        _vakt(service, bana0_t, lambda: motor["v"])
        asyncio.get_event_loop().call_later(0.1, lambda: bana0_t.update(v=hel))
        await _call(service, "intent__HassTurnOn", {"area": "kontoret", "domain": ["light"]})
        assert (ran == ["intent__HassTurnOn"]) is ska_koras, (krummel, hel, ran)


@pytest.mark.asyncio
async def test_utan_text_i_tid_kors_anropet_efter_taket_och_en_lasning_vantar_aldrig(monkeypatch):
    import time
    monkeypatch.setenv("BANA0_VANTA_TEXT_MS", "150")
    tom = lambda: ""
    t0 = time.monotonic()
    assert await bana0.fraga_inte_order_vanta(tom, "styr", {"vad": "kontoret", "atgard": "pa"}, bana0_text=tom) is None
    assert 0.12 <= time.monotonic() - t0 < 0.6            # waited about the cap, then ran (fail-open for an order)
    t0 = time.monotonic()
    assert await bana0.fraga_inte_order_vanta(tom, "hus_lage", {}, {"hus_lage": True}, bana0_text=tom) is None
    assert time.monotonic() - t0 < 0.05                    # a read tool never waits
    fraga = lambda: "Är det tänt i kontoret?"
    t0 = time.monotonic()
    assert await bana0.fraga_inte_order_vanta(tom, "styr", {}, bana0_text=fraga) is not None
    assert time.monotonic() - t0 < 0.05                    # bana 0's text already there: no wait
    t0 = time.monotonic()
    assert await bana0.fraga_inte_order_vanta(tom, "styr", {}) is None
    assert time.monotonic() - t0 < 0.05                    # a room without bana 0 (GEMINI_EGEN_TUR): nothing to wait for
    monkeypatch.setenv("BANA0_VANTA_TEXT_MS", "0")
    t0 = time.monotonic()
    assert await bana0.fraga_inte_order_vanta(tom, "styr", {}, bana0_text=tom) is None
    assert time.monotonic() - t0 < 0.05                    # 0 = off


def test_vantetaket_klipps_vid_tva_sekunder_och_skräp_ger_standard():
    assert bana0.vanta_pa_text_tak_s({"BANA0_VANTA_TEXT_MS": "inf"}) == 0.3
    assert bana0.vanta_pa_text_tak_s({"BANA0_VANTA_TEXT_MS": "nan"}) == 0.3
    assert bana0.vanta_pa_text_tak_s({"BANA0_VANTA_TEXT_MS": "x"}) == 0.3
    assert bana0.vanta_pa_text_tak_s({"BANA0_VANTA_TEXT_MS": "99999"}) == 2.0
    assert bana0.vanta_pa_text_tak_s({"BANA0_VANTA_TEXT_MS": "-5"}) == 0.0
    assert bana0.vanta_pa_text_tak_s({}) == 0.3


@pytest.mark.asyncio
async def test_sessionen_far_bana0_lasaren_utom_i_ett_rum_dar_motorn_kor_turerna(monkeypatch):
    from tests.test_ha_tools_recovery import _connect
    from tests.test_provider_selection import _bare_app

    connection, service = await _connect(_bare_app("openai"))
    connection.bana0_heard = "Tänd i hallen"
    assert service.fragevakt_bana0() == "Tänd i hallen"
    monkeypatch.setenv("GEMINI_EGEN_TUR", "1")
    connection2, service2 = await _connect(_bare_app("openai"))
    assert service2.fragevakt_bana0 is None   # bana 0 writes nothing there: the guard must not wait for it
