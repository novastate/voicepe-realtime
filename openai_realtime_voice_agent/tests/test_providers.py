"""One narrow door to every engine."""

import pytest

from app.providers import (
    ProviderOptions,
    build_service,
    input_sample_rate,
    self_heals,
)

TOOLS = [{
    "type": "function",
    "name": "search_home",
    "description": "Search the house.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}]


def _options(**over):
    base = dict(
        api_key="sk-test",
        model="gpt-realtime-2",
        voice="cedar",
        instructions="Du är Björn.",
        max_output_tokens=1024,
        speed=1.0,
        noise_reduction="",
        turn_detection_type="semantic_vad",
        vad_eagerness="medium",
        vad_threshold=0.5,
        vad_prefix_padding_ms=300,
        vad_silence_duration_ms=500,
        semantic_vad_create_response=True,
        interrupt_response=True,
        transcription_model="gpt-4o-transcribe",
        transcription_language="sv",
    )
    base.update(over)
    return ProviderOptions(**base)


def test_the_device_rate_is_kept_for_gemini_and_raised_for_openai():
    # The Voice PE microphone produces 16 kHz. OpenAI wants 24 kHz, Gemini
    # wants exactly what the device already sends.
    assert input_sample_rate("openai") == 24000
    assert input_sample_rate("gemini") == 16000


def test_only_gemini_repairs_its_own_connection():
    # pipecat's OpenAI service has no reconnect logic at all; the Gemini one
    # has _reconnect and session resumption. ConnectionRecovery must know.
    assert self_heals("openai") is False
    assert self_heals("gemini") is True


def test_an_unknown_engine_is_refused_loudly():
    with pytest.raises(ValueError, match="unknown provider"):
        build_service("claude", _options(), TOOLS)
    with pytest.raises(ValueError, match="unknown provider"):
        input_sample_rate("claude")


def test_the_openai_service_carries_the_tools_and_the_instructions():
    service = build_service("openai", _options(), TOOLS)
    props = service._session_properties
    assert props.tools == TOOLS
    assert props.instructions.startswith("Du är Björn.")
    assert props.audio.output.voice == "cedar"


def test_the_openai_service_still_no_ops_the_truncate():
    # The fork disables truncation-on-interruption; moving the class must not
    # quietly drop that. See the class docstring for what it cost.
    from app.providers.openai_realtime import SafeRealtimeLLMService
    service = build_service("openai", _options(), TOOLS)
    assert isinstance(service, SafeRealtimeLLMService)
