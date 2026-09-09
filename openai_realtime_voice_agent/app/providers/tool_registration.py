"""How a tool gets registered, identically on every engine.

This used to live inside `SafeRealtimeLLMService`, so it protected the OpenAI
path only. The house then moved to Gemini and the protection stayed behind:
measured live 2026-09-09 22:23, `play_media` was cancelled 81 ms after it
started while the assistant said "jajemän, fixar det" and nothing played.
Same for `GetLiveContext` and `vaderprognos`, all evening.

Keeping it in one mixin is the point. A rule that has to be remembered twice
is a rule that protects one engine.
"""
import logging

logger = logging.getLogger(__name__)


class ToolRegistrationMixin:
    """Registers every tool the same way, whichever engine is running.

    Mix in BEFORE the pipecat service class so this `register_function` wins:

        class Foo(ToolRegistrationMixin, SomeLLMService): ...
    """

    # main.py sets these on the service right after build_service(). The
    # defaults exist so a tool registered before that never raises.
    speaker_probe = None
    male_only_tools: set = set()
    turn_liveness = None

    def register_function(self, function_name, handler, start_callback=None, *,
                          cancel_on_interruption: bool = True):  # type: ignore[override]
        """Force cancel_on_interruption=False for every tool registration.

        pipecat cancels in-flight function-call tasks on EVERY user-speech
        interruption. Both engines produce one per utterance, for different
        reasons: OpenAI's semantic_vad fires on each utterance fragment, so
        merely continuing your own sentence kills the tool call your previous
        fragment started; on Gemini the user transcript arrives AFTER the
        model has already called the tool, and the user aggregator turns that
        late transcript into an emulated "user started speaking" — an
        interruption caused by the very sentence that asked for the tool.

        By then the HTTP request to Home Assistant has usually already been
        SENT: the action executes, but its result never reaches the model,
        which then tells the user it failed (observed live: the lights turned
        ON while the assistant claimed they wouldn't). Or the request is cut
        off mid-flight and nothing happens at all, while the assistant still
        says it did. Our tools are all short-lived (HA service calls, one web
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
            if self.turn_liveness is not None:
                self.turn_liveness.tool_started()
            try:
                return await handler(params)
            finally:
                if self.turn_liveness is not None:
                    self.turn_liveness.tool_finished()

        super().register_function(
            function_name, liveness_tracked, start_callback, cancel_on_interruption=False
        )
