"""Restoring a cached conversation on reconnect must actually reach the
model -- or the assistant "forgets" every time the speaker reconnects, which
happens often (closed follow-up windows, add-on updates, dropped network).

Two independent reviews, tracing pipecat 0.0.97, concluded that
`ContextInitializer` (session_manager.py) never delivers: it pushes an
`LLMMessagesUpdateFrame` via `context_aggregator.user().push_frame(...)`, but
`FrameProcessor.push_frame` forwards a frame to `self._next` -- it does NOT
invoke the target processor's own `process_frame`. That sends the frame
straight past the aggregator to whatever is next in the pipeline (the LLM
service), and neither `OpenAIRealtimeLLMService` nor `GeminiLiveLLMService`
has a branch for `LLMMessagesUpdateFrame` -- both just forward it on
unchanged. The log line `📤 Sent cached context (N messages)` fires
unconditionally either way, which is exactly why nobody noticed.

This is the same defect class Task 6b found and fixed for a different
feature (telling the model who is speaking) -- see
`_send_gemini_note_silently` and `make_speaker_note` in websocket_handler.py,
and `tests/test_speaker_injection.py`'s docstring for the fully-traced
history of that fix. The fix here follows the same shape: bypass the broken
frame-based channel and talk to each engine's own session directly (see
`app/context_restore.py`'s `restore_context_silently` and its two
per-engine helpers).

Both tests below build a REAL, wired 3-stage pipecat pipeline --
`[context_aggregator.user(), <engine service>, ContextInitializer]` --
using real `Pipeline`/`PipelineTask`/`PipelineRunner`, faking only the
websocket/session (no network). This reproduces production topology closely
enough that, against the pre-fix code, it proves the actual defect: a frame
pushed from the aggregator does reach the service (it's the very next
processor), but the service does nothing useful with it, and nothing is ever
sent. (Verified directly against the pre-fix `ContextInitializer` before this
fix was written -- `fake_ws.sent` came back empty.)
"""

import asyncio

import pytest
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

from app.providers import GEMINI, OPENAI, ProviderOptions, build_service
from app.session_manager import ContextInitializer

RESTORED_MESSAGES = [
    {"role": "user", "content": "what time is it in the kitchen"},
    {"role": "assistant", "content": "it is two o'clock"},
]


def _openai_options(**over):
    base = dict(
        api_key="sk-test",
        model="gpt-realtime-2",
        voice="cedar",
        instructions="You are Bjorn.",
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


def _gemini_options(**over):
    base = dict(
        api_key="AIza-test",
        model="models/gemini-3.1-flash-live-preview",
        voice="Charon",
        instructions="You are Bjorn.",
        max_output_tokens=1024,
        language="sv-SE",
    )
    base.update(over)
    return ProviderOptions(**base)


class _FakeOpenAIWebSocket:
    """Stands in for the OpenAI Realtime websocket. Records outbound sends;
    nothing touches the network. Mirrors test_speaker_injection.py's fake.
    `close()` is needed here (unlike the speaker-injection fake) because
    driving a real pipeline to EndFrame triggers a real disconnect."""

    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        pass


class _FakeGeminiSession:
    """Stands in for the Gemini Live `AsyncSession`. Records calls; nothing
    touches the network. Mirrors test_speaker_injection.py's fake."""

    def __init__(self):
        self.calls = []

    async def send_client_content(self, **kwargs):
        self.calls.append(kwargs)

    async def close(self):
        pass


async def _drive_wired_pipeline(components, settle=0.15):
    """Run a real pipecat pipeline just long enough for StartFrame to
    propagate through every stage and any awaited work it triggers to
    complete, then shut it down cleanly with EndFrame.

    This is the same pattern pipecat's own `pipecat.tests.utils.run_test`
    uses (a short head-start sleep before queuing the next control frame),
    just built by hand because we need a multi-stage pipeline (aggregator ->
    service -> ContextInitializer -> sink) rather than `run_test`'s fixed
    3-stage [source, processor_under_test, sink] shape.
    """
    pipeline = Pipeline(components)
    task = PipelineTask(pipeline, cancel_on_idle_timeout=False)

    async def push():
        await asyncio.sleep(settle)
        await task.queue_frame(EndFrame())

    runner = PipelineRunner(handle_sigint=False)
    await asyncio.gather(runner.run(task), push())


def _build_restore_aggregator():
    """Build the same shape SessionManager.create_context_aggregator does
    when cached context exists: a fresh LLMContext pre-seeded with the
    restored messages, wrapped in a real LLMContextAggregatorPair."""
    context = LLMContext(messages=list(RESTORED_MESSAGES))
    return LLMContextAggregatorPair(context), context


@pytest.mark.asyncio
async def test_restored_context_reaches_a_real_openai_service_without_starting_a_turn():
    """The proof: a real, wired OpenAIRealtimeLLMService must actually
    receive the FULL restored conversation (both turns, not just one -- this
    is REPLACE semantics: the whole cached conversation, not a single
    injected line), and must NOT receive a response.create -- the assistant
    must wait for the person to speak."""
    service = build_service(OPENAI, _openai_options(), [])
    fake_ws = _FakeOpenAIWebSocket()
    service._websocket = fake_ws
    assert service._llm_needs_conversation_setup is True  # sanity: real default

    aggregator_pair, context = _build_restore_aggregator()
    initializer = ContextInitializer(
        cached_context=context,
        client_id="kitchen",
        service=service,
        provider=OPENAI,
    )

    await _drive_wired_pipeline([aggregator_pair.user(), service, initializer])

    assert len(fake_ws.sent) > 0, (
        "the restored conversation never reached the OpenAI Realtime "
        "websocket at all"
    )
    joined = " ".join(fake_ws.sent)
    assert "kitchen" in joined and "two o'clock" in joined, (
        "something was sent, but it doesn't contain the FULL restored "
        "conversation (both the user's and the assistant's turn)"
    )
    assert not any('"type": "response.create"' in payload for payload in fake_ws.sent), (
        "a response.create was sent -- the assistant would speak unprompted "
        "on a restored connection, which is exactly what run_llm=False exists "
        "to prevent"
    )
    assert service._llm_needs_conversation_setup is False, (
        "the conversation-setup flag was left True, so the FIRST real turn's "
        "own _create_response() will resend this exact same conversation a "
        "second time as duplicate conversation items"
    )


@pytest.mark.asyncio
async def test_restored_context_reaches_a_real_gemini_service_without_starting_a_turn():
    """Same proof for Gemini: the restored conversation must reach the real
    session via send_client_content, with turn_complete=False (no immediate
    reply), and the service's own pending-turn-close flag must end up True
    so the ALREADY-WIRED _handle_user_stopped_speaking silently closes the
    turn the next time the user's real speech ends -- never before."""
    service = build_service(GEMINI, _gemini_options(), [])
    fake_session = _FakeGeminiSession()
    service._session = fake_session
    assert service._needs_turn_complete_message is False  # sanity: real default

    aggregator_pair, context = _build_restore_aggregator()
    initializer = ContextInitializer(
        cached_context=context,
        client_id="kitchen",
        service=service,
        provider=GEMINI,
    )

    await _drive_wired_pipeline([aggregator_pair.user(), service, initializer])

    assert len(fake_session.calls) > 0, (
        "the restored conversation never reached the Gemini Live session at all"
    )
    call = fake_session.calls[0]
    assert call["turn_complete"] is False, (
        "turn_complete=True was sent -- the assistant would speak unprompted "
        "on a restored connection"
    )
    assert not any(c.get("turn_complete") is True for c in fake_session.calls), (
        "no call in this restore should ever close the turn immediately"
    )
    turns = call["turns"]
    assert len(turns) == len(RESTORED_MESSAGES), (
        "REPLACE semantics: the FULL restored conversation (both turns) must "
        "be sent, not a single collapsed or dropped message"
    )
    all_text = " ".join(
        part.text or "" for content in turns for part in content.parts
    )
    assert "kitchen" in all_text and "two o'clock" in all_text, (
        "the turns sent don't contain the restored conversation's actual content"
    )
    assert service._needs_turn_complete_message is True, (
        "the silent turn-close bridge was never armed, so the restored turn "
        "will never close on its own"
    )
