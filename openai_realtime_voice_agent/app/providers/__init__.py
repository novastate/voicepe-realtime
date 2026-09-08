"""One narrow door to every voice engine.

Everything that differs between OpenAI Realtime and Gemini Live lives behind
this module. No code outside it may know which engine is running -- except the
router, which only ever handles names.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# The engine names used in the add-on config, in logs and by the router.
OPENAI = "openai"
GEMINI = "gemini"
PROVIDERS = (OPENAI, GEMINI)

# What each engine wants the microphone audio to be. The device produces
# 16 kHz; OpenAI needs it raised, Gemini takes it as it is. Both answer with
# 24 kHz, which is what the pipeline already plays.
_INPUT_RATE = {OPENAI: 24000, GEMINI: 16000}

# pipecat's OpenAI Realtime service has no reconnect logic; a dead socket
# floods ErrorFrames forever. The Gemini service has _reconnect,
# _handle_connection_error and session resumption, so it repairs itself and
# ConnectionRecovery must keep its hands off.
_SELF_HEALS = {OPENAI: False, GEMINI: True}


@dataclass
class ProviderOptions:
    """Every knob the add-on can set, for whichever engine reads it.

    Fields an engine does not have are simply ignored by that engine's module.
    Gemini has no playback speed and no noise reduction; OpenAI has no thinking
    budget. Keeping one shape means the caller never branches on the engine.
    """

    api_key: str
    model: str
    voice: str
    instructions: str
    max_output_tokens: Optional[int] = None
    # OpenAI only
    speed: float = 1.0
    noise_reduction: str = ""
    turn_detection_type: str = "semantic_vad"
    vad_eagerness: str = "medium"
    vad_threshold: float = 0.5
    vad_prefix_padding_ms: int = 300
    vad_silence_duration_ms: int = 500
    semantic_vad_create_response: bool = True
    interrupt_response: bool = True
    transcription_model: str = "gpt-4o-transcribe"
    transcription_language: str = ""
    # Gemini only
    language: str = "sv-SE"


def _known(provider: str) -> str:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")
    return provider


def input_sample_rate(provider: str) -> int:
    """The microphone sample rate this engine wants, in Hz."""
    return _INPUT_RATE[_known(provider)]


def self_heals(provider: str) -> bool:
    """Whether this engine reconnects its own dead socket."""
    return _SELF_HEALS[_known(provider)]


def build_service(provider: str, options: ProviderOptions, tools: List[Dict[str, Any]]):
    """Build a configured, unconnected session for one device.

    Args:
        provider: "openai" or "gemini".
        options: Every knob; each engine reads the ones it has.
        tools: Tool definitions in OpenAI Realtime shape. The Gemini module
            converts them; nothing outside providers/ needs two shapes.

    Returns:
        A pipecat LLMService.
    """
    if _known(provider) == OPENAI:
        from app.providers import openai_realtime
        return openai_realtime.build(options, tools)
    from app.providers import gemini_live
    return gemini_live.build(options, tools)
