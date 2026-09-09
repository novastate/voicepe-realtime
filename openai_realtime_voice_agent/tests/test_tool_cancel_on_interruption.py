"""A tool call must survive the interruption caused by asking for it.

Measured live 2026-09-09 22:23 on Gemini: the user transcript arrives AFTER
the model has called the tool, pipecat's user aggregator turns that late
transcript into an emulated "user started speaking", and the interruption
cancels the in-flight call 81 ms in. play_media died; the assistant still
said it was playing. The OpenAI service had been immune since it was written;
the Gemini one had never been given the same rule.
"""

import pytest

from app.providers import GEMINI, OPENAI, ProviderOptions, build_service


def _options(provider):
    if provider == GEMINI:
        return ProviderOptions(
            api_key="AIza-test",
            model="models/gemini-3.1-flash-live-preview",
            voice="Charon",
            instructions="Du är Björn.",
            language="sv-SE",
        )
    return ProviderOptions(
        api_key="sk-test",
        model="gpt-realtime-2",
        voice="cedar",
        instructions="Du är Björn.",
    )


async def _handler(params):  # pragma: no cover - never called here
    return None


@pytest.mark.parametrize("provider", [OPENAI, GEMINI])
def test_registered_tools_are_not_cancelled_on_interruption(provider):
    service = build_service(provider, _options(provider), [])
    service.register_function("play_media", _handler)

    entry = service._functions["play_media"]
    assert entry.cancel_on_interruption is False, (
        f"{provider}: a user interruption would kill this tool mid-flight"
    )


@pytest.mark.parametrize("provider", [OPENAI, GEMINI])
def test_caller_cannot_re_enable_cancellation(provider):
    """Even an explicit True is refused — the rule is not a default."""
    service = build_service(provider, _options(provider), [])
    service.register_function("set_timer", _handler, cancel_on_interruption=True)

    assert service._functions["set_timer"].cancel_on_interruption is False
