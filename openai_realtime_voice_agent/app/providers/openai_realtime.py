"""OpenAI Realtime engine: the SafeRealtimeLLMService fixes and the session builder.

Everything here is specific to OpenAI's Realtime API. Nothing outside
`app/providers/` should import from this module directly except the
`app.providers` boundary itself.
"""
import asyncio
import logging
import os

from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from app.realtime_payload import transform_gpt_transcription_language

logger = logging.getLogger(__name__)


class SafeRealtimeLLMService(OpenAIRealtimeLLMService):
    """OpenAIRealtimeLLMService with audio-truncation-on-interruption disabled.

    pipecat's `_truncate_current_audio_response()` (called by `_handle_interruption`
    on EVERY interruption — both our device "stop" AND pipecat's own server-VAD
    barge-in when the user wakes/speaks mid-reply) sends a
    `conversation.item.truncate` with `audio_end_ms = wall-clock ms since audio
    start`. But OpenAI BURSTS the reply faster than real-time, so that elapsed
    value massively overshoots the audio that actually exists, and OpenAI rejects
    it with `invalid_request_error("Audio content of N ms is already shorter than
    M ms")`. That errored truncate wedges the realtime session, so the user's very
    next turn gets NO response — the recurring "interrupt, then immediately ask
    again → silence" bug (confirmed in logs: session goes quiet right after
    `_truncate_current_audio_response`).

    The device stops playback authoritatively on its own, so server-side
    truncation buys us nothing. No-op it. (Cost: OpenAI's conversation history
    keeps the full assistant text the user may not have fully heard — purely
    cosmetic for context.)
    """

    async def _truncate_current_audio_response(self):  # type: ignore[override]
        return

    async def send_client_event(self, event):  # type: ignore[override]
        """Serialize GPT transcription models with their required `languages` field.

        Pipecat 0.0.97 only exposes the legacy singular `language` field on
        InputAudioTranscription. OpenAI rejects that field for newer GPT
        transcription models, which instead expect `languages: [<ISO code>]`.
        """
        payload = event.model_dump(exclude_none=True)
        transform_gpt_transcription_language(payload)
        await self._ws_send(payload)

    # Per-response cost accounting (fork). The API reports exact token usage in
    # every response.done; pipecat only pushes it as metrics frames. Log the
    # breakdown + estimated $ (measured 2026-07-12: warm turn ≈ $0.003-0.013,
    # cold session-first turn ≈ $0.019 — the 4.4k-token instruction+tool prefix
    # uncached) and publish a daily cost sensor to HA.
    _RATES = (  # $/1M tokens: (text_in, text_out, audio_in, audio_out, cached)
        (0.60, 2.40, 10.0, 20.0, 0.06)
        if "mini" in (os.environ.get("OPENAI_MODEL_CUSTOM") or os.environ.get("OPENAI_MODEL") or "")
        else (4.0, 24.0, 32.0, 64.0, 0.40)
    )

    async def _handle_evt_response_done(self, evt):  # type: ignore[override]
        try:
            u = evt.response.usage
            itd = getattr(u, "input_token_details", None)
            otd = getattr(u, "output_token_details", None)
            ctd = getattr(itd, "cached_tokens_details", None)
            in_text = getattr(itd, "text_tokens", 0) or 0
            in_audio = getattr(itd, "audio_tokens", 0) or 0
            cached = getattr(itd, "cached_tokens", 0) or 0
            c_text = getattr(ctd, "text_tokens", 0) or 0
            c_audio = getattr(ctd, "audio_tokens", 0) or 0
            out_text = getattr(otd, "text_tokens", 0) or 0
            out_audio = getattr(otd, "audio_tokens", 0) or 0
            ti, to, ai, ao, ca = self._RATES
            cost = (max(0, in_text - c_text) * ti + max(0, in_audio - c_audio) * ai
                    + cached * ca + out_text * to + out_audio * ao) / 1e6
            logger.info(
                f"💰 usage: in {in_text}txt+{in_audio}aud (cached {cached}) "
                f"out {out_text}txt+{out_audio}aud ≈ ${cost:.4f}"
            )
            from app.ha_sensors import PUBLISHER
            asyncio.get_running_loop().create_task(PUBLISHER.usage(cost, {
                "in_text": in_text, "in_audio": in_audio, "cached": cached,
                "out_text": out_text, "out_audio": out_audio}))
        except Exception as e:
            logger.debug(f"usage accounting failed: {e!r}")
        await super()._handle_evt_response_done(evt)

    async def reset_conversation(self):  # type: ignore[override]
        """Reconnect WITHOUT forcing a response on the reconnected session.

        pipecat's reset_conversation() (used by ConnectionRecovery on a 60-min cap
        / keepalive drop) reconnects and leaves `_llm_needs_conversation_setup =
        True`. The collision: if a turn was mid-flight when the WS dropped,
        `_create_response()` had already set `_run_llm_when_api_session_ready =
        True` (because `_api_session_ready` went False on disconnect). After the
        reconnect, the `session.updated` handler sees that flag and fires
        `_create_response()` — but under semantic_vad (`create_response=true`) the
        SERVER also auto-creates a response for the user's next turn. Two
        response.create events collide → `conversation_already_has_active_response`,
        and that turn gets no answer (observed: first turn right after a reconnect
        fails, ~1 in 20 reconnects — whenever the user happens to speak in the few
        seconds just after a reconnect).

        Fix: after the normal reconnect, clear `_run_llm_when_api_session_ready` so
        the reconnected session does NOT self-create a response, and set
        `_llm_needs_conversation_setup = False` (same as the startup pre-seed) — the
        server-VAD drives every user-turn response, so we never need to create one
        ourselves on reconnect.

        CORRECTED (Task 9b fix round 1): this comment used to claim "the live
        context is untouched (it's restored by the SessionManager on the next
        real turn)". That was false, and remained false after Task 9b's first
        pass at this file — the SessionManager's restore path
        (ContextInitializer) only ever runs once, on a brand-new client
        WebSocket connection; it never re-fires for an in-place reconnect of
        this SAME service instance, because no new StartFrame is produced.
        Meanwhile `self._context` (this Python object) is untouched by
        reset — it still holds the full conversation up to the disconnect —
        but the FRESH OpenAI-side session this reconnect just opened knows
        NOTHING about it, and clearing `_llm_needs_conversation_setup` above
        stops `_create_response()`'s own lazy "send initial messages" loop
        from ever sending it either. Net effect before this fix: every
        60-minute session cap silently wiped the model's memory of the
        conversation. `_reseed_context_after_reset()` below re-sends
        `self._context`'s own messages onto the freshly reconnected session,
        the same way `ContextInitializer` seeds a brand-new one — using this
        service's own live context (more current than SessionManager's cache,
        which only snapshots on disconnect), not the SessionManager's cache.
        """
        self._resetting_conversation = True
        try:
            await super().reset_conversation()
            try:
                self._run_llm_when_api_session_ready = False
                self._llm_needs_conversation_setup = False
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"⚠️ could not clear post-reconnect response flags: {e!r}")
            await self._reseed_context_after_reset()
        finally:
            self._resetting_conversation = False

    async def _reseed_context_after_reset(self):
        """Re-seed the freshly reconnected session with the conversation it
        just lost (see `reset_conversation`'s docstring for why this is
        needed at all).

        Uses this service's OWN `_context` — the live, ongoing conversation
        pipecat has been tracking for this connection all along — not the
        SessionManager's cache (that cache is keyed by client_id for a
        brand-new WebSocket connection; this is an in-place reconnect of the
        same service, which the SessionManager never even sees). Tool calls
        are stripped first for the same reason `_strip_tool_plumbing` exists
        for the client-reconnect path: their ids belong to the conversation
        that just ended, and the fresh session rejects a replayed id.

        Known timing caveat, not verifiable without a live API key: this
        sends immediately after `_connect()` returns, i.e. as soon as the
        websocket handshake completes, which may be before OpenAI's
        `session.created` event has arrived. `_create_response()`'s own
        lazy conversation-setup loop (the path this replaces) only ever ran
        once pipecat's aggregator finished a real turn — comfortably after
        `session.created` in practice — so this is untested territory this
        fix introduces. If OpenAI rejects conversation.item.create sent this
        early, the fix would need to defer until `_api_session_ready` (the
        same flag `_run_llm_when_api_session_ready` already waits on).
        """
        context = getattr(self, "_context", None)
        if context is None:
            return
        from app.context_restore import _strip_tool_plumbing, restore_context_silently
        from app.providers import OPENAI

        messages = _strip_tool_plumbing(context.get_messages())
        if not messages:
            return
        try:
            restored = await restore_context_silently(OPENAI, self, messages)
        except Exception as e:
            logger.warning(f"⚠️ Failed to re-seed context after reset: {e!r}")
            return
        if restored:
            logger.info(
                f"📤 Re-seeded {len(messages)} message(s) onto the reconnected "
                f"session after reset_conversation (waiting for user)"
            )

    # Error codes that must NOT kill the realtime session. pipecat 0.0.97's
    # _receive_task_handler does `_handle_evt_error(evt); return` on EVERY
    # error event — the reader task dies, the in-flight reply cuts off
    # mid-sentence and the session is deaf until the next connection death.
    # Observed live (2026-06-10): semantic_vad split one utterance into two
    # turns, the server's auto-created second response collided with the
    # first → conversation_already_has_active_response → the playing reply
    # stopped at 4.4 s and the session wedged. These codes are harmless
    # protocol races; the right move is to keep reading.
    BENIGN_ERROR_CODES = {
        # The server auto-created a response while one was still active
        # (VAD split a sentence into two turns). The active response keeps
        # streaming — nothing is broken.
        "conversation_already_has_active_response",
        # response.cancel landed after the response already finished (device
        # "stop" / the post-interrupt racing-response kill) — nothing to
        # cancel, nothing broken.
        "response_cancel_not_active",
        # input_audio_buffer.commit raced our input_audio_buffer.clear (device
        # "stop"): an empty commit is exactly the outcome we wanted.
        "input_audio_buffer_commit_empty",
    }

    async def _maybe_handle_evt_retrieve_conversation_item_error(self, evt):  # type: ignore[override]
        """Generic benign-error filter, hooked into pipecat's receive loop.

        pipecat's `_receive_task_handler` treats a True return from this
        method as "error handled — keep the receive loop alive"; every other
        error event kills the reader task (`_handle_evt_error` + `return`).
        It is the ONLY surviving path, so besides the original retrieve-item
        case (super()), we declare our benign protocol races handled here
        instead of letting them cut off live audio and wedge the session.
        """
        if await super()._maybe_handle_evt_retrieve_conversation_item_error(evt):
            return True
        code = getattr(getattr(evt, "error", None), "code", None)
        if code in self.BENIGN_ERROR_CODES:
            logger.warning(
                f"⚠️ benign realtime error ignored (session stays alive): {code}"
            )
            return True
        return False

    def register_function(self, function_name, handler, start_callback=None, *,
                          cancel_on_interruption: bool = True):  # type: ignore[override]
        """Force cancel_on_interruption=False for every tool registration.

        pipecat cancels in-flight function-call tasks on EVERY user-speech
        interruption — and semantic_vad fires one per utterance fragment, so
        merely continuing your own sentence kills the tool call your previous
        fragment started. By then the HTTP request to Home Assistant has
        usually already been SENT: the action executes, but its result never
        reaches the model, which then tells the user it failed (observed
        live: the lights turned ON while the assistant claimed they
        wouldn't). Our tools are all short-lived (HA service calls, one web
        search), so letting them finish and report the truth always beats
        killing them halfway. This single override covers every registration
        path (MCP tools via pipecat's MCPClient, web_search, disconnect).

        The handler is also wrapped to tick its connection's liveness around its run, so
        the PhaseEmitter's thinking-watchdog knows a tool is in flight and a
        slow tool (web search: 10-20 s of pipeline silence) is never mistaken
        for a dead turn. All our handlers use the single-param
        FunctionCallParams signature, so the wrapper does too (pipecat
        inspects the signature to pick the calling convention).
        """
        async def liveness_tracked(params):
            # Speaker gate (fork): tools listed in male_only_tools only execute
            # when the last voice-type verdict is "male". Enforced HERE — below
            # the model — so prompt tricks can't bypass it. Fails closed on
            # uncertain/stale/absent verdicts. This is convenience gating on a
            # voice-type heuristic, not biometric auth.
            if self.male_only_tools and function_name in self.male_only_tools:
                speaker = self.speaker_probe.gate_speaker() if self.speaker_probe else "unknown"
                if speaker != "male":
                    owner = (self.speaker_probe.male_name if self.speaker_probe else "") or "the owner"
                    logger.info(f"⛔ speaker gate blocked '{function_name}' (speaker={speaker})")
                    await params.result_callback({
                        "error": (
                            f"Not available: this capability is reserved for {owner}, "
                            f"and the current speaker's voice was not recognized as {owner}. "
                            f"Relay this politely."
                        )
                    })
                    return
            self.turn_liveness.tool_started()
            try:
                return await handler(params)
            finally:
                self.turn_liveness.tool_finished()

        super().register_function(
            function_name, liveness_tracked, start_callback, cancel_on_interruption=False
        )

    async def _receive_task_handler(self):  # type: ignore[override]
        """Surface OpenAI reader death as an ErrorFrame so recovery can act.

        pipecat's receive loop can end without producing ANY ErrorFrame: a
        silent server-side close ends the `async for` normally, and a network
        drop raises ConnectionClosed, which the task manager merely LOGS
        ("unexpected exception"). Nothing reaches ConnectionRecovery either
        way, so the session sat deaf for HOURS until the next user utterance
        hit the dead socket — losing that utterance (observed live twice).
        Wrap the loop and report its end; ConnectionRecovery treats the
        message as a reconnect trigger.
        """
        try:
            await super()._receive_task_handler()
        except asyncio.CancelledError:
            raise  # our own disconnect/reset tearing the task down — not a death
        except Exception as e:
            await self.push_error(error_msg=f"realtime receive loop died: {e!r}")
            return
        # reset_conversation() intentionally closes the old reader before
        # connecting the replacement session. That normal close is not a
        # recoverable failure and may arrive after the processor has stopped.
        if getattr(self, "_resetting_conversation", False):
            return
        # Loop ended without an exception: a clean server-side close, or the
        # fatal-error path (which already pushed its own ErrorFrame —
        # duplicates collapse in ConnectionRecovery's cooldown/guard).
        await self.push_error(error_msg="realtime receive loop ended — connection closed")


def build(options, tools):
    """Build a configured OpenAI Realtime session for one device.

    Args:
        options: The ProviderOptions carrying every knob from the add-on config.
        tools: Tool definitions in OpenAI Realtime shape.

    Returns:
        A SafeRealtimeLLMService, not yet connected.
    """
    from pipecat.services.openai.realtime.events import (
        AudioConfiguration,
        AudioInput,
        AudioOutput,
        InputAudioNoiseReduction,
        InputAudioTranscription,
        SemanticTurnDetection,
        SessionProperties,
        TurnDetection,
    )

    # Turn detection: semantic_vad (recommended — semantic end-of-turn,
    # echo-resistant, doesn't cut the user off) or classic server_vad.
    if options.turn_detection_type == "semantic_vad":
        turn_detection = SemanticTurnDetection(
            eagerness=options.vad_eagerness,
            # create_response=True (default): the SERVER creates a
            # response on every detected end-of-turn. This is required for
            # multi-turn conversation. Pipecat 0.0.97's
            # OpenAIRealtimeLLMService._handle_context only auto-creates a
            # response for the FIRST context (turn 1) and after tool
            # results (its else-branch just updates the context); a plain
            # 2nd/3rd user turn therefore gets NO response unless the
            # server makes it. We previously set this False to stop a
            # turn-1 double-response (server + Pipecat first-context both
            # creating → `conversation_already_has_active_response`), but
            # that silently broke every turn after the first (device hung
            # in "thinking"). True is the correct trade: the server drives
            # all user-turn responses; Pipecat still creates the post-tool
            # response via _process_completed_function_calls. To stop the
            # turn-1 double (server + Pipecat-first-context both creating →
            # conversation_already_has_active_response), run() seeds
            # self._context once at startup with a kickoff LLMRunFrame, so
            # the user's first real turn hits the else-branch too.
            create_response=options.semantic_vad_create_response,
            interrupt_response=options.interrupt_response,
        )
    else:
        turn_detection = TurnDetection(
            type="server_vad",
            threshold=options.vad_threshold,
            prefix_padding_ms=options.vad_prefix_padding_ms,
            silence_duration_ms=options.vad_silence_duration_ms,
        )

    # Optionally pin the input-transcription language to stop the model
    # drifting between languages (e.g. "nl"). Empty -> auto-detect.
    # transcription_model picks the STT used for the transcript text.
    transcription = (
        InputAudioTranscription(
            model=options.transcription_model,
            language=options.transcription_language,
        )
        if options.transcription_language
        else None
    )

    # Optional near/far-field input noise reduction (helps the VAD reject
    # background noise / residual speaker leak). None = off (default).
    noise_reduction = (
        InputAudioNoiseReduction(type=options.noise_reduction)
        if options.noise_reduction
        else None
    )

    session_properties = SessionProperties(
        instructions=options.instructions,
        # Cap the reply length: bounds runaway monologues + per-response
        # output-token cost. None = unlimited (the API default "inf").
        max_output_tokens=options.max_output_tokens,
        audio=AudioConfiguration(
            input=AudioInput(
                turn_detection=turn_detection,
                transcription=transcription,
                noise_reduction=noise_reduction,
            ),
            # speed is a post-generation playback rate (0.25-1.5, 1.0 = normal).
            output=AudioOutput(voice=options.voice, speed=options.speed),
        ),
        tools=tools,
    )

    if options.turn_detection_type == "semantic_vad":
        logger.info(
            f"🎚️ Turn detection: semantic_vad (eagerness={options.vad_eagerness}, "
            f"create_response={options.semantic_vad_create_response}, "
            f"interrupt_response={options.interrupt_response})"
            + (f", transcription={options.transcription_model} (lang={options.transcription_language})" if options.transcription_language else " (transcription off)")
        )
    else:
        logger.info(
            f"🎚️ Turn detection: server_vad (threshold={options.vad_threshold}, "
            f"silence_duration_ms={options.vad_silence_duration_ms})"
            + (f", transcription={options.transcription_model} (lang={options.transcription_language})" if options.transcription_language else " (transcription off)")
        )

    return SafeRealtimeLLMService(
        api_key=options.api_key,
        model=options.model,
        session_properties=session_properties,
        start_audio_paused=False,
    )
