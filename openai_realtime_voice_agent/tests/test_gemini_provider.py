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


# --- the shape google-genai actually accepts --------------------------------

def test_the_service_gets_tools_wrapped_the_way_the_api_wants_them():
    """One wrapper deeper than the declarations themselves.

    Handing pipecat the bare declarations made google-genai reject every field
    of every tool as "extra inputs are not permitted", and the session never
    opened. Found in the house, not by the suite — the old tests only checked
    what `to_gemini_tools` returned, never what was done with it.
    """
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    tools = service._tools_from_init
    assert isinstance(tools, list) and len(tools) == 1
    declarations = tools[0]["function_declarations"]
    assert [d["name"] for d in declarations] == ["search_home", "list_timers"]


def test_gemini_never_sees_additional_properties():
    """Gemini rejects the keyword outright, and Home Assistant generates
    schemas that carry it — one such tool would take the whole session down."""
    noisy = [{
        "type": "function",
        "name": "noisy",
        "description": "Has the keyword at three depths.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "nested": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"deep": {"type": "string"}},
                },
                "listed": {
                    "type": "array",
                    "items": {"type": "object", "additionalProperties": True},
                },
            },
        },
    }]
    params = to_gemini_tools(noisy)[0]["parameters"]
    assert "additionalProperties" not in params
    assert "additionalProperties" not in params["properties"]["nested"]
    assert "additionalProperties" not in params["properties"]["listed"]["items"]
    assert params["properties"]["nested"]["properties"]["deep"]["type"] == "string"


def test_the_session_always_carries_turn_detection_settings():
    """Live 2026-09-09: the first Gemini session sent no realtime_input_config
    at all, so Google ran its automatic activity detection at its own default
    (START_SENSITIVITY_HIGH). The device answered room noise, its own speaker
    echo and half-words nobody said -- "Och?", "Ja.", "Ne?", and one whole
    sentence in Portuguese. pipecat only attaches the config when at least one
    field is set, so every field is sent."""
    from google.genai.types import EndSensitivity, StartSensitivity

    service = build_service("gemini", _options(), OPENAI_SHAPE)
    vad = service._vad_params
    assert vad is not None
    assert vad.start_sensitivity == StartSensitivity.START_SENSITIVITY_LOW
    assert vad.end_sensitivity == EndSensitivity.END_SENSITIVITY_LOW
    assert vad.prefix_padding_ms == 300
    assert vad.silence_duration_ms == 800


def test_the_sensitivities_follow_the_add_on_settings():
    from google.genai.types import EndSensitivity, StartSensitivity

    service = build_service(
        "gemini",
        _options(
            gemini_vad_start_sensitivity="high",
            gemini_vad_end_sensitivity="high",
            gemini_vad_prefix_padding_ms=120,
            gemini_vad_silence_duration_ms=400,
        ),
        OPENAI_SHAPE,
    )
    vad = service._vad_params
    assert vad.start_sensitivity == StartSensitivity.START_SENSITIVITY_HIGH
    assert vad.end_sensitivity == EndSensitivity.END_SENSITIVITY_HIGH
    assert vad.prefix_padding_ms == 120
    assert vad.silence_duration_ms == 400


def test_an_unreadable_sensitivity_falls_back_to_low_not_to_googles_default():
    """A typo in add-on config must not silently hand the room back to
    Google's HIGH default -- that is the exact failure this setting exists to
    prevent, and it is inaudible until the assistant starts answering the
    television."""
    from google.genai.types import StartSensitivity

    service = build_service(
        "gemini", _options(gemini_vad_start_sensitivity="lowish"), OPENAI_SHAPE
    )
    assert service._vad_params.start_sensitivity == StartSensitivity.START_SENSITIVITY_LOW


def test_a_negative_padding_is_clamped_rather_than_sent():
    service = build_service(
        "gemini", _options(gemini_vad_prefix_padding_ms=-50), OPENAI_SHAPE
    )
    assert service._vad_params.prefix_padding_ms == 0


def test_proactive_audio_is_off_unless_asked_for():
    service = build_service("gemini", _options(), OPENAI_SHAPE)
    assert not service._settings.get("proactivity")


def test_proactive_audio_needs_a_native_audio_model():
    """Google's guide is explicit: proactive audio and affective dialog are
    not supported on Gemini 3.1 Flash Live. Sending the config anyway gets the
    whole session refused, so an operator who ticks the box without moving the
    model must lose the feature, not the assistant."""
    service = build_service(
        "gemini",
        _options(gemini_proactive_audio=True),  # still the 3.1 preview model
        OPENAI_SHAPE,
    )
    assert not service._settings.get("proactivity")


def test_proactive_audio_reaches_a_native_audio_session():
    service = build_service(
        "gemini",
        _options(
            model="models/gemini-2.5-flash-native-audio-latest",
            gemini_proactive_audio=True,
        ),
        OPENAI_SHAPE,
    )
    assert service._settings["proactivity"].proactive_audio is True
