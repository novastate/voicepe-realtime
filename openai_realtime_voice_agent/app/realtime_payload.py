"""Compatibility transforms for OpenAI Realtime client-event payloads."""


GPT_TRANSCRIPTION_MODELS_WITH_LANGUAGE_HINTS = {
    "gpt-live-transcribe",
    "gpt-transcribe",
}


def transform_gpt_transcription_language(payload: dict) -> None:
    """Adapt Pipecat's singular language field for GPT transcription models."""
    if payload.get("type") != "session.update":
        return

    transcription = payload.get("session", {}).get("input_audio_transcription")
    if (
        not transcription
        or transcription.get("model") not in GPT_TRANSCRIPTION_MODELS_WITH_LANGUAGE_HINTS
    ):
        return

    language = transcription.pop("language", None)
    if language:
        transcription["languages"] = [language]
