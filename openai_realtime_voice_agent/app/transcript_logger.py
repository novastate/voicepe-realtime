"""Log the conversation transcript (assistant + user) into the add-on log.

WHY: with gpt-realtime the model hears the user's audio natively and bursts the
whole spoken reply as audio — the only window into *what it actually said* used to
be the OpenAI tool-call arguments. This processor surfaces the assistant's spoken
text (and, when input transcription is enabled, the user's transcript) as plain
INFO lines so the add-on log alone explains a turn.

How the text reaches us (verified against pipecat 0.0.97's *real*
`pipecat.services.openai.realtime.llm.OpenAIRealtimeLLMService` — NOT the older
`openai_realtime_beta` module, which pushes different frames):

  - Assistant reply, AUDIO modality (what we use): the service handles
    `response.output_audio_transcript.delta` and pushes a **`TTSTextFrame`** per
    chunk (NOT `LLMTextFrame` — that's only for the text modality). The whole
    response is bracketed by `LLMFullResponseStartFrame` /
    `LLMFullResponseEndFrame`. We accumulate the chunks and log one line on the
    End frame. (We also match `LLMTextFrame` so a text-modality run still logs.)
    These flow DOWNSTREAM out of the service, so the "assistant" tap sits AFTER
    it in the pipeline.

  - User transcript: `conversation.item.input_audio_transcription.completed`
    pushes a `TranscriptionFrame` — but UPSTREAM (toward the input, so the user
    context aggregator can consume it) and ONLY when input transcription is
    configured (main.py: transcription is None unless TRANSCRIPTION_LANGUAGE is
    set). A tap placed AFTER the service never sees it; the "user" tap therefore
    sits BEFORE the service (between the user aggregator and the LLM).

Because the two transcripts travel in opposite directions past the service, no
single position sees both — so the pipeline wires TWO instances of this class,
one per role (see websocket_handler.build_pipeline). It is pure instrumentation:
it never transforms or drops a frame. (Listed for removal under CLAUDE.md
roadmap #5 once the system is stable.)
"""
import logging

from pipecat.frames.frames import (
    Frame,
    TTSTextFrame,
    LLMTextFrame,
    LLMFullResponseEndFrame,
    BotStoppedSpeakingFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

logger = logging.getLogger(__name__)


class TranscriptLogger(FrameProcessor):
    """Forward-only processor that logs assistant and/or user transcript lines.

    Args:
        capture: which side to log — "assistant" (TTS/LLM reply text, place AFTER
            the LLM), "user" (TranscriptionFrame, place BEFORE the LLM), or "both".
    """

    def __init__(self, capture: str = "both", on_user=None, on_assistant=None, **kwargs):
        super().__init__(**kwargs)
        self._capture = capture
        self._on_user = on_user
        self._on_assistant = on_assistant  # raawr US-542: what the speech-to-text heard goes to Comms/Core, never what the model wrote
        self._assistant_buf: list[str] = []
        self._vantar_uppspelning = ""  # raawr US-542: finished reply text, delivered only when the audio has played to its end

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._capture in ("assistant", "both"):
            # Accumulate the reply text chunks (audio modality -> TTSTextFrame;
            # text modality -> LLMTextFrame), then log once per response on the
            # End bracket so it's one readable line instead of one per chunk.
            if isinstance(frame, StartInterruptionFrame):
                # raawr US-542 (Codex P2): the owner cut the speech off, also AFTER the text was finished. Nothing waiting is delivered.
                self._assistant_buf = []
                self._vantar_uppspelning = ""
            elif isinstance(frame, (TTSTextFrame, LLMTextFrame)):
                if frame.text:
                    self._assistant_buf.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame):
                text = "".join(self._assistant_buf).strip()
                self._assistant_buf = []
                if text:
                    logger.info(f"🤖 assistant: {text}")
                    # Delivered means HEARD: the text counts only once the output audio stops (BotStoppedSpeakingFrame) without an
                    # interruption in between. A reply whose audio had already ended before this frame is never reported (fail-closed).
                    self._vantar_uppspelning = text
            elif isinstance(frame, BotStoppedSpeakingFrame) and self._vantar_uppspelning:
                text, self._vantar_uppspelning = self._vantar_uppspelning, ""
                if self._on_assistant is not None:
                    try:
                        self._on_assistant(text)
                    except Exception as e:  # reporting must never stop the audio
                        logger.warning(f"⚠️ levererat: {type(e).__name__}")

        if self._capture in ("user", "both"):
            if isinstance(frame, TranscriptionFrame):
                text = (frame.text or "").strip()
                if text:
                    logger.info(f"🗣️ user: {text}")
                    if self._on_user is not None:
                        try:
                            self._on_user(text)
                        except Exception as e:  # reporting must never stop the audio
                            logger.warning(f"⚠️ hord: {type(e).__name__}")

        await self.push_frame(frame, direction)
