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

from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, InputParams
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


def build(options, tools: List[Dict[str, Any]]) -> GeminiLiveLLMService:
    """Build a configured Gemini Live session for one device.

    Args:
        options: The ProviderOptions carrying every knob from the add-on config.
            Speed, noise reduction and the OpenAI turn-detection knobs have no
            equivalent here and are ignored.
        tools: Tool definitions in OpenAI Realtime shape.

    Returns:
        A GeminiLiveLLMService, not yet connected.
    """
    model = options.model or DEFAULT_MODEL
    voice = options.voice or DEFAULT_VOICE
    language = _resolve_language(options.language or "sv-SE")
    params = InputParams(
        max_tokens=options.max_output_tokens or 4096,
        language=language,
    )
    logger.info(
        f"🔧 Gemini Live session: model={model} voice={voice} "
        f"lang={language} tools={len(tools)}"
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
    )
