"""Gemini takes the same tools, in its own shape."""

from app.providers import ProviderOptions, build_service
from app.providers.gemini_live import to_gemini_tools

OPENAI_SHAPE = [
    {
        "type": "function",
        "name": "search_home",
        "description": "Search the house.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "list_timers",
        "description": "List timers.",
        # A tool with no arguments at all. Gemini refuses a declaration with no
        # parameters block, so one must be invented.
    },
]


def _options(**over):
    base = dict(
        api_key="AIza-test",
        model="models/gemini-3.1-flash-live-preview",
        voice="Charon",
        instructions="Du är Björn.",
        max_output_tokens=1024,
        language="sv-SE",
    )
    base.update(over)
    return ProviderOptions(**base)


def test_tools_keep_their_name_description_and_parameters():
    decls = to_gemini_tools(OPENAI_SHAPE)
    assert [d["name"] for d in decls] == ["search_home", "list_timers"]
    assert decls[0]["parameters"]["properties"]["query"]["type"] == "string"
    assert decls[0]["description"] == "Search the house."


def test_a_tool_without_parameters_gets_an_empty_object():
    decls = to_gemini_tools(OPENAI_SHAPE)
    assert decls[1]["parameters"] == {"type": "object", "properties": {}}


def test_the_openai_only_type_key_is_dropped():
    # "type": "function" is OpenAI Realtime's envelope, not part of the schema.
    for d in to_gemini_tools(OPENAI_SHAPE):
        assert "type" not in d


def test_the_service_is_built_with_our_model_voice_and_instructions():
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    assert service.model_name == "models/gemini-3.1-flash-live-preview"
    assert service._voice_id == "Charon"
    # pipecat stashes the raw string until the context is initialized on
    # connect; it never lands under a plain `_system_instruction` attribute.
    assert "Björn" in service._system_instruction_from_init


def test_the_dead_default_model_is_never_used():
    # pipecat defaults to models/gemini-2.0-flash-live-001, which no longer
    # exists and refuses the socket with 1008. Our default must be a live one.
    from app.providers.gemini_live import DEFAULT_MODEL
    assert DEFAULT_MODEL != "models/gemini-2.0-flash-live-001"
    assert DEFAULT_MODEL.startswith("models/gemini-")


def test_the_language_is_resolved_to_a_gemini_language_enum():
    # InputParams.language is typed Optional[Language] (a pipecat StrEnum), not
    # a bare string. Passing "sv-SE" straight through would fail pydantic
    # validation the first time someone actually spoke in the house.
    from pipecat.transcriptions.language import Language

    service = build_service("gemini", _options(language="sv-SE"), OPENAI_SHAPE)
    assert service._language == Language.SV_SE


def test_an_unresolvable_language_falls_back_instead_of_crashing():
    from pipecat.transcriptions.language import Language

    service = build_service(
        "gemini", _options(language="not-a-real-language"), OPENAI_SHAPE
    )
    assert isinstance(service._language, Language)
