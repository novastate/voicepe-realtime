"""Google Gemini Live, wearing the same shape as the OpenAI session.

Proven against the live account 2026-09-08: all 52 tools accepted at once --
this add-on's 8 plus Home Assistant's 44 -- and the model picked search_home by
itself when asked about the car's battery, then answered in Swedish.

⛔ pipecat 0.0.97 defaults to models/gemini-2.0-flash-live-001, which the API
refuses with `1008 ... not supported for bidiGenerateContent`. Every live model
is a preview name and will be retired in turn, so the model is a setting and
this default is only the one that worked on the day it was written.
"""
import logging
from typing import Any, Dict, List

from google.genai.types import (
    EndSensitivity,
    HttpOptions,
    ProactivityConfig,
    StartSensitivity,
)
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    GeminiVADParams,
    InputParams,
)
from pipecat.transcriptions.language import Language

logger = logging.getLogger(__name__)

# Verified live 2026-09-08. The other working one is
# models/gemini-2.5-flash-native-audio-latest, which additionally supports
# thinking and affective dialog.
DEFAULT_MODEL = "models/gemini-3.1-flash-live-preview"
DEFAULT_VOICE = "Charon"

# Falls back here if the configured language string isn't a Language member.
# Swedish because that's the house this add-on runs in.
FALLBACK_LANGUAGE = Language.SV_SE

# Google's automatic activity detection decides when a user turn starts and
# ends. Pass it nothing and the API runs it at START_SENSITIVITY_HIGH, which
# is why the first live session answered room noise, its own speaker echo and
# stray half-words ("Och?", "Ja.", "Né?", one whole sentence in Portuguese) as
# if they were questions. These maps turn the add-on's plain "low"/"high"
# strings into the enums google-genai wants.
_START_SENSITIVITY = {
    "low": StartSensitivity.START_SENSITIVITY_LOW,
    "high": StartSensitivity.START_SENSITIVITY_HIGH,
}
_END_SENSITIVITY = {
    "low": EndSensitivity.END_SENSITIVITY_LOW,
    "high": EndSensitivity.END_SENSITIVITY_HIGH,
}

# Proactive audio (the model deciding for itself whether it was spoken to)
# lives behind API version v1beta and only on the native-audio models. The
# preview model this add-on defaults to, gemini-3.1-flash-live-preview, does
# NOT support it: Google's own guide says so, and the session is refused
# rather than degraded. Turning it on without moving the model just breaks
# the engine, so a mismatch is refused here, loudly, instead of at 1008.
PROACTIVE_AUDIO_API_VERSION = "v1beta"
PROACTIVE_AUDIO_MODEL_MARKER = "native-audio"


def to_gemini_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI Realtime tool definitions to Gemini declarations.

    The two differ in exactly two ways: OpenAI wraps each tool in a
    ``"type": "function"`` envelope, and it allows a tool with no parameters at
    all, where Gemini insists on an object schema even when it is empty.

    Args:
        tools: Tool definitions in OpenAI Realtime shape.

    Returns:
        Function declarations for Gemini Live.
    """
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": _strip_additional_properties(
                tool.get("parameters") or {"type": "object", "properties": {}}
            ),
        }
        for tool in tools
    ]


def _strip_additional_properties(schema: Any) -> Any:
    """Remove every "additionalProperties" key, at any depth.

    Gemini rejects the keyword outright. Home Assistant generates its tool
    schemas, and some of them carry it, so one such tool would take the whole
    session down with it. pipecat's own adapter strips it for exactly this
    reason -- see GeminiLLMAdapter.to_provider_tools_format.
    """
    if isinstance(schema, dict):
        return {
            key: _strip_additional_properties(value)
            for key, value in schema.items()
            if key != "additionalProperties"
        }
    if isinstance(schema, list):
        return [_strip_additional_properties(item) for item in schema]
    return schema


def _resolve_language(language: str) -> Language:
    """Turn the add-on's plain language string into pipecat's enum.

    ``InputParams.language`` is typed ``Optional[Language]`` (a pipecat
    StrEnum), not a bare string -- passing "sv-SE" straight through fails
    pydantic validation the moment a session actually starts. An unresolvable
    value falls back rather than crashing the whole session over a typo in
    add-on config.

    Args:
        language: A BCP-47-ish string like "sv-SE", from ProviderOptions.

    Returns:
        The matching Language member, or FALLBACK_LANGUAGE if none matches.
    """
    try:
        return Language(language)
    except ValueError:
        logger.warning(
            f"⚠️ Unknown Gemini language {language!r}; falling back to {FALLBACK_LANGUAGE}"
        )
        return FALLBACK_LANGUAGE


def _build_vad_params(options) -> GeminiVADParams:
    """Turn the add-on's turn-detection knobs into Gemini's VAD config.

    Every field is always sent. pipecat only attaches
    ``realtime_input_config`` when at least one field is set, and a
    half-filled config would leave the rest at Google's defaults -- which is
    the state this function exists to get away from. An unknown string falls
    back to LOW rather than to Google's HIGH default: the failure mode of
    "slightly too hard to trigger" is a missed word, the failure mode of the
    default is the assistant talking to the room.

    Args:
        options: The ProviderOptions carrying the four gemini_vad_* knobs.

    Returns:
        A fully populated GeminiVADParams.
    """
    start = (options.gemini_vad_start_sensitivity or "low").strip().lower()
    end = (options.gemini_vad_end_sensitivity or "low").strip().lower()
    if start not in _START_SENSITIVITY:
        logger.warning(f"⚠️ Unknown gemini_vad_start_sensitivity {start!r}; using 'low'")
        start = "low"
    if end not in _END_SENSITIVITY:
        logger.warning(f"⚠️ Unknown gemini_vad_end_sensitivity {end!r}; using 'low'")
        end = "low"
    return GeminiVADParams(
        start_sensitivity=_START_SENSITIVITY[start],
        end_sensitivity=_END_SENSITIVITY[end],
        prefix_padding_ms=max(0, int(options.gemini_vad_prefix_padding_ms)),
        silence_duration_ms=max(0, int(options.gemini_vad_silence_duration_ms)),
    )


def _proactivity_for(options, model: str):
    """Whether this session asks Google to judge if it was spoken to.

    Args:
        options: The ProviderOptions carrying gemini_proactive_audio.
        model: The resolved model name, which decides whether the feature
            can be asked for at all.

    Returns:
        A (ProactivityConfig, HttpOptions) pair, or (None, None) when the
        feature is off or the model cannot carry it.
    """
    if not options.gemini_proactive_audio:
        return None, None
    if PROACTIVE_AUDIO_MODEL_MARKER not in model:
        logger.warning(
            f"⚠️ gemini_proactive_audio is on but {model} does not support it "
            f"(it needs a *-{PROACTIVE_AUDIO_MODEL_MARKER}-* model, e.g. "
            f"models/gemini-2.5-flash-native-audio-latest) — starting WITHOUT it "
            f"rather than letting the session be refused"
        )
        return None, None
    logger.info("🤫 Proactive audio ON — the model may stay silent when not addressed")
    return (
        ProactivityConfig(proactive_audio=True),
        HttpOptions(api_version=PROACTIVE_AUDIO_API_VERSION),
    )


def build(options, tools: List[Dict[str, Any]]) -> GeminiLiveLLMService:
    """Build a configured Gemini Live session for one device.

    Args:
        options: The ProviderOptions carrying every knob from the add-on config.
            Speed and noise reduction have no equivalent here and are ignored.
            Turn detection DOES have one -- Google's automatic activity
            detection -- but it is configured through the gemini_vad_* knobs,
            not the OpenAI semantic_vad ones, so those are ignored too.
        tools: Tool definitions in OpenAI Realtime shape.

    Returns:
        A GeminiLiveLLMService, not yet connected.
    """
    model = options.model or DEFAULT_MODEL
    voice = options.voice or DEFAULT_VOICE
    language = _resolve_language(options.language or "sv-SE")
    vad = _build_vad_params(options)
    proactivity, http_options = _proactivity_for(options, model)
    params = InputParams(
        max_tokens=options.max_output_tokens or 4096,
        language=language,
        vad=vad,
        proactivity=proactivity,
    )
    logger.info(
        f"🔧 Gemini Live session: model={model} voice={voice} "
        f"lang={language} tools={len(tools)}"
    )
    logger.info(
        f"🎚️ Gemini turn detection: start={vad.start_sensitivity} "
        f"end={vad.end_sensitivity} prefix={vad.prefix_padding_ms}ms "
        f"silence={vad.silence_duration_ms}ms"
    )
    return GeminiLiveLLMService(
        api_key=options.api_key,
        model=model,
        voice_id=voice,
        system_instruction=options.instructions,
        # One wrapper deeper than it looks: the Live API takes a LIST OF TOOLS,
        # each of which carries its function declarations. Handing it the bare
        # declarations makes google-genai reject every one of them as an extra
        # field, and the session never opens. pipecat's own adapter wraps them
        # the same way -- see GeminiLLMAdapter.to_provider_tools_format.
        tools=[{"function_declarations": to_gemini_tools(tools)}] if tools else None,
        start_audio_paused=False,
        params=params,
        # Only passed when proactive audio asked for it; None otherwise leaves
        # google-genai on its own default version.
        http_options=http_options,
    )
