"""How a tool gets registered, identically on every engine.

This used to live inside `SafeRealtimeLLMService`, so it protected the OpenAI
path only. The house then moved to Gemini and the protection stayed behind:
measured live 2026-09-09 22:23, `play_media` was cancelled 81 ms after it
started while the assistant said "jajemän, fixar det" and nothing played.
Same for `GetLiveContext` and `vaderprognos`, all evening.

Keeping it in one mixin is the point. A rule that has to be remembered twice
is a rule that protects one engine.
"""
import asyncio
import json
import re
import inspect
import logging
import os
import time

from pipecat.frames.frames import FunctionCallResultProperties

from app import bana0, core_strom, skills
from app.early_ack import paa as early_ack_paa

logger = logging.getLogger(__name__)


def _result_max_chars() -> int:
    try:
        return max(0, int(os.environ.get("TOOL_RESULT_MAX_CHARS", "6000")))
    except ValueError:
        return 6000


def _tool_timeout_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("TOOL_TIMEOUT_SECONDS", "30")))
    except ValueError:
        return 30.0


def cap_tool_result(result, max_chars: int):
    """Shorten a tool result that would flood the conversation.

    Every later turn re-bills everything in the conversation. One unfiltered
    GetLiveContext is the whole house, 16 kB / ~6,300 tokens: measured live
    2026-10-02, a turn went 8,767 -> 15,026 input tokens and the next
    sentence hit the 40k TPM limit. 6,000 chars still fits a whole-domain
    query (all lights: 5.1 kB). The note tells the model how to get the rest.
    """
    if not max_chars or result is None:
        return result
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    if len(text) <= max_chars:
        return result
    cut = text[:max_chars]
    cut = cut[: max(cut.rfind("\n"), cut.rfind("\\n"), max_chars // 2)]
    return (cut + f"\n[TRUNCATED: {len(text) - len(cut)} of {len(text)} characters cut. "
            "Ask again with a narrower filter (name, area or domain).]")


def _early_ack_ms() -> int:
    if not early_ack_paa():
        return 0
    try:
        return max(0, int(os.environ.get("EARLY_ACK_MS", "700")))
    except ValueError:
        return 700


def _silence_ack_ms() -> int:
    if not early_ack_paa():
        return 0
    try:
        return max(0, int(os.environ.get("EARLY_ACK_SILENCE_MS", "1500")))
    except ValueError:
        return 1500


def _followup_ack_ms() -> int:
    if not early_ack_paa():
        return 0
    try:
        return max(0, int(os.environ.get("EARLY_ACK_FOLLOWUP_MS", "3000")))
    except ValueError:
        return 3000


def ack_delay_ms(wake_ms: int, liveness) -> int:
    """How long to wait before "jag kollar" in this turn; 0 = never.

    The owner, 2026-10-02 21:16: the ack is good on the first question after
    the wake word, not on follow-ups in the same conversation, unless the wait
    is really long. A follow-up turn (no wake since the last idle) waits
    EARLY_ACK_FOLLOWUP_MS (3000, 0 = never) for both triggers.
    """
    if wake_ms <= 0 or liveness is None or liveness.from_wake():
        return wake_ms
    return _followup_ack_ms()


# Loop guard (0.25.1), every engine: Grok called GetLiveContext up to 48
# times in one turn with made-up arguments when the answer was not there
# (probe 2026-10-02). Past the limit HA is not called; the model is told to
# answer with what it has. 0.25.6: IDENTICAL calls are counted (same tool,
# same arguments) -- "tänd kontoret, köket, hallen och sovrummet" is four
# HassTurnOn and all four must run.
MAX_SAME_TOOL_PER_TURN = 3
MAX_TOOLS_PER_TURN = 12


def _norm(value):
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    return value


def _bana0_for_tur(f, tur):
    """US-256: bana 0's text for the turn the call arrived in; a reader without turn ids is called as before."""
    return f if f is None or tur is None else (lambda: f(tur))


def _call_key(function_name, arguments) -> str:
    """The tool and its arguments, normalized (key order, case, spaces)."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass
    return function_name + " " + json.dumps(_norm(arguments or {}), sort_keys=True,
                                            ensure_ascii=False, default=str)


def loop_stop(liveness, function_name, arguments=None):
    """Count this run; return the stop text if it is over the turn's limit."""
    if liveness is None:
        return None
    counts = liveness.tool_counts
    key = _call_key(function_name, arguments)
    same, total = counts.get(key, 0), sum(counts.values())
    if same >= MAX_SAME_TOOL_PER_TURN or total >= MAX_TOOLS_PER_TURN:
        logger.info(f"⏱ tool-loop stopp {function_name} {same + 1}")
        if same >= MAX_SAME_TOOL_PER_TURN:
            what = f"{function_name} med samma argument {same} gånger"
        else:
            what = f"verktyg {total} gånger"
        return (f"Stopp: du har redan anropat {what} i den här turen. "
                "Svara nu med det du vet, eller säg ärligt att du inte hittar det.")
    counts[key] = same + 1
    return None


def _lista(v):
    return list(v) if isinstance(v, (list, tuple)) else ([v] if v else [])


def media_paus(function_name, arguments):
    """HassTurnOff aimed at a media player -> (HassMediaPause's name, its arguments), else None.

    Voice PE speakers cannot be turned off: HA answers `failed: media_player.*` and the radio plays on
    (kontoret 2026-10-11 'Stäng av radion'). Off for music means pause.
    """
    if function_name.rsplit("__", 1)[-1] != "HassTurnOff":
        return None
    if isinstance(arguments, str):  # some engines hand the arguments over as a JSON string
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if not isinstance(arguments, dict):
        return None
    args = dict(arguments)
    domaner = [str(d) for d in _lista(args.get("domain"))]
    if not domaner or any(d != "media_player" for d in domaner):
        return None  # 'turn off the lights and the radio' must still turn the lights off: only a pure speaker request is a pause
    return "HassMediaPause", {k: args[k] for k in ("name", "area", "floor") if args.get(k)}


def _namn(post):
    return str(post.get("name") or post.get("id") or post) if isinstance(post, dict) else str(post)


def ha_utfall(result):
    """(success, failed) names from a Home Assistant intent result, ([], []) when it has none or another shape."""
    def leta(o, djup=0):
        if djup > 4:
            return None
        if isinstance(o, str):
            t = o.strip()
            if t[:1] in "{[":
                try:
                    return leta(json.loads(t), djup + 1)
                except ValueError:
                    return None
            return None
        if isinstance(o, dict):
            if isinstance(o.get("failed"), list) or isinstance(o.get("success"), list):
                return o
            for v in o.values():
                r = leta(v, djup + 1)
                if r is not None:
                    return r
        if isinstance(o, list):
            for v in o:
                r = leta(v, djup + 1)
                if r is not None:
                    return r
        return None
    hit = leta(result)
    if hit is None:
        return [], []
    return [_namn(x) for x in hit.get("success") or []], [_namn(x) for x in hit.get("failed") or []]


def ha_misslyckades(result):
    """A result that tells the model plainly when HA says failed is not empty, else None."""
    ok, fel = ha_utfall(result)
    if not fel:
        return None
    if ok:
        text = (f"Delvis: klart för {', '.join(ok)}. INTE klart för {', '.join(fel)}. "
                "Säg ifrån om det som inte gick, kvittera inte allt som klart.")
    else:
        text = (f"Det gick INTE: {', '.join(fel)} misslyckades. Säg ärligt till användaren att det inte gick. "
                "Säg inte att det är klart.")
    return {"error": text, "failed": fel, "success": ok}


NAMN_OK = re.compile(r"[A-Za-z0-9_]{1,64}")


def _is_error(result) -> bool:
    return isinstance(result, dict) and "error" in result


class ToolRegistrationMixin:
    """Registers every tool the same way, whichever engine is running.

    Mix in BEFORE the pipecat service class so this `register_function` wins:

        class Foo(ToolRegistrationMixin, SomeLLMService): ...
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        async def unknown_tool(params):
            logger.warning(f"okänt verktyg: {str(params.function_name)[:64]!r}")
            await params.result_callback({"error": f"okänt verktyg: {params.function_name}"})

        # Pipecat dispatches unregistered names to None, preserving the call ID
        # and sending the result through the normal context/result pipeline.
        super().register_function(None, unknown_tool, cancel_on_interruption=False)

    # main.py sets these on the service right after build_service(). The
    # defaults exist so a tool registered before that never raises.
    speaker_probe = None
    male_only_tools: set = set()
    turn_liveness = None
    # async (started, recent=None, tool=None) -> None: speaks a short phrase for `tool`
    # ("Jag söker på nätet.") on this connection's device. Set by main.py; None (tests, no device) means stay silent.
    early_ack = None
    _silence_ack_task = None

    def arm_silence_ack(self) -> None:
        """The model was just handed the user's turn; fill a long silence.

        Called where each engine is asked to answer: Gemini's activityEnd,
        OpenAI's speech_stopped (or bana 0's miss). Never on a bana 0 hit,
        which never asks the model. Live 2026-10-02 15:19:44 Gemini took
        3.7 s to its function call and 5.5 s to its first audio, and the
        tool ack never fired because the tool itself took 0.2 s. If the
        model has said nothing after EARLY_ACK_SILENCE_MS (1500, 0 = off),
        the same early ack plays: once per turn, never over the model's own
        audio or a new utterance, never in its history.
        """
        self.cancel_silence_ack()
        ms = ack_delay_ms(_silence_ack_ms(), self.turn_liveness)
        if self.early_ack is None or ms <= 0:
            return
        asked = time.monotonic()

        async def ack_if_silent():
            await asyncio.sleep(ms / 1000.0)
            liveness = self.turn_liveness
            if liveness is None or not liveness.claim_silence_ack(asked):
                return
            try:
                await self.early_ack(asked, 0.0)
            except Exception as e:
                logger.warning(f"⚠️ silence ack failed: {e!r}")

        self._silence_ack_task = asyncio.ensure_future(ack_if_silent())

    def cancel_silence_ack(self) -> None:
        if self._silence_ack_task is not None:
            self._silence_ack_task.cancel()
            self._silence_ack_task = None

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
        says it did. Pipecat therefore never owns cancellation. TurnLiveness
        instead cancels the wrapper task only for a real user-speech start;
        Gemini's emulated start leaves it alone. This single override covers
        every registration path (MCP tools via pipecat's MCPClient, web_search,
        disconnect).

        The handler is also wrapped to tick its connection's liveness around its run, so
        the PhaseEmitter's thinking-watchdog knows a tool is in flight and a
        slow tool (web search: 10-20 s of pipeline silence) is never mistaken
        for a dead turn. All our handlers use the single-param
        FunctionCallParams signature, so the wrapper does too (pipecat
        inspects the signature to pick the calling convention).
        """
        if not hasattr(self, "_ra_handlers"):
            self._ra_handlers = {}
        self._ra_handlers[function_name] = handler

        async def run_liveness_tracked(params):
            # One line per call, every engine, every tool (MCP included):
            # `⏱ tool <name> <ms> ok|fel`, timed to the moment the result is
            # handed back -- that is what the person in the room waits for.
            started = time.monotonic()
            tur_nr = getattr(self, "tur_nr", None)
            tur = tur_nr() if callable(tur_nr) else None   # US-256: the turn this call belongs to, fixed on arrival
            call_key = _call_key(function_name, getattr(params, "arguments", None))
            current_task = asyncio.current_task()
            logged = False
            utfall = "ok"
            ack_speaking = False
            if getattr(self, "tider", None) is not None:
                self.tider.mark("verktyg")

            def log_timing(status: str) -> None:
                nonlocal logged
                if not logged:
                    logged = True
                    ms = round((time.monotonic() - started) * 1000)
                    logger.info(f"⏱ tool {function_name} {ms} {status}")
                    bas = function_name.rsplit("__", 1)[-1]
                    if bas == "fraga_core":
                        logger.info("🧰 verktyg fraga_core")
                    elif bas in ("las_skill", "anvand_verktyg"):
                        arguments = getattr(params, "arguments", None)
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except (ValueError, TypeError):
                                arguments = None
                        namn = arguments.get("namn") if isinstance(arguments, dict) else None
                        # Only the selected name, never the payload or answer. A missing, empty
                        # or made-up name (too long, odd characters) is logged as okänt, so every
                        # call is counted (US-206, US-215).
                        if not (isinstance(namn, str) and NAMN_OK.fullmatch(namn)):
                            namn = "okänt"
                        typ = "skill" if bas == "las_skill" else "verktyg"
                        logger.info(f"🧰 {typ} {namn}")

            async def ack_if_slow():
                # The owner, 2026-10-02: a smart agent says it has understood
                # but needs to check, instead of going quiet. Fast tools (HA:
                # 0.1-0.8 s) finish before this fires and stay silent.
                nonlocal ack_speaking
                await asyncio.sleep(ack_ms / 1000.0)
                liveness = self.turn_liveness
                if liveness is not None and not liveness.claim_ack(started):
                    return
                ack_speaking = True
                try:
                    await self.early_ack(started, tool=function_name)
                except Exception as e:
                    logger.warning(f"⚠️ early ack failed: {e!r}")

            ack_task = None
            ack_ms = ack_delay_ms(_early_ack_ms(), self.turn_liveness)
            if self.early_ack is not None and ack_ms > 0:
                ack_task = asyncio.ensure_future(ack_if_slow())

            def stop_waiting_ack() -> None:
                # Only a pending ack is dropped; one already speaking finishes
                # its sentence (the device queues the reply after it).
                if ack_task is not None and not ack_speaking:
                    ack_task.cancel()

            # Capped here, below every engine and every tool, MCP included.
            original_callback = params.result_callback
            callback_used = False
            max_chars = _result_max_chars()

            async def tracked_callback(result, *args, **kwargs):
                nonlocal callback_used
                if (self.turn_liveness is not None
                        and self.turn_liveness.tool_was_cancelled(current_task)):
                    return None
                callback_used = True
                # shielded: a cap or a barge-in landing mid-delivery must not leave the call half-answered.
                # Accepted risk (Hemma 2026-10-21): a barge-in landing in the milliseconds while the handler's OWN result is
                # being handed over lets that result through with default properties, so the model may speak after the
                # interruption. A whole, true answer beats a half-delivered one.
                return await asyncio.shield(original_callback(result, *args, **kwargs))

            async def capped_callback(result, *args, **kwargs):
                nonlocal utfall
                stop_waiting_ack()
                if function_name.rsplit("__", 1)[-1] == "las_skill":  # what anvand_verktyg may run on a question
                    self.lasande = {**(getattr(self, "lasande", None) or {}), **skills.lasande_ur(result)}
                fel = ha_misslyckades(result)
                if fel is not None:
                    logger.warning(f"⚠️ {function_name}: Home Assistant says failed: {', '.join(fel['failed'])}")
                    result = fel
                utfall = "fel" if _is_error(result) else "ok"
                log_timing(utfall)
                return await tracked_callback(cap_tool_result(result, max_chars), *args, **kwargs)

            params.result_callback = capped_callback
            if self.turn_liveness is not None:
                self.turn_liveness.tool_started(call_key, current_task)
            try:
                if (self.turn_liveness is not None
                        and self.turn_liveness.interrupted_recently(call_key)):
                    await params.result_callback({
                        "result": "Anropet avbröts av användaren och körs inte igen. Fråga vad han vill."
                    })
                    return
                stop = loop_stop(self.turn_liveness, function_name, getattr(params, "arguments", None))
                if stop is not None:
                    await params.result_callback({"result": stop})
                    return
                omvag = core_strom.kunskap_nekas(function_name, getattr(params, "arguments", None))   # US-191
                if omvag:
                    logger.warning(f"🛡️ tool {function_name} NOT run: {omvag}")
                    await params.result_callback({"result": omvag})
                    return
                vakt = getattr(self, "verktygsvakt", None)   # echo guard (US-060): his own voice must not run a tool
                skal = vakt() if vakt is not None else None
                if inspect.isawaitable(skal):   # GEMINI_EGEN_TUR: the guard may wait a moment for the turn's text
                    skal = await skal
                if skal:
                    logger.warning(f"🛡️ tool {function_name} NOT run: {skal}")
                    await params.result_callback({"result": skal})
                    return
                fraga = getattr(self, "fragevakt", None)  # a question is no order (raawr US-098, 2026-10-17)
                nej = await bana0.fraga_inte_order_vanta(fraga, function_name, getattr(params, "arguments", None),
                                                         getattr(self, "lasande", None),
                                                         bana0_text=_bana0_for_tur(getattr(self, "fragevakt_bana0", None), tur)) if fraga else None   # US-253
                if nej:
                    logger.warning(f"🛡️ tool {function_name} NOT run: {nej}")
                    await params.result_callback({"result": nej})
                    return
                for_brett = bana0.omrade_utan_doman(function_name, getattr(params, "arguments", None))
                if for_brett:  # G finding 20: an area-wide turn on/off without a domain switches everything there
                    logger.warning(f"🛡️ tool {function_name} NOT run: {for_brett}")
                    await params.result_callback({"result": for_brett})
                    return
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
                target = handler
                paus = media_paus(function_name, getattr(params, "arguments", None))
                if paus is not None:
                    ny = next((n for n in (f"media_player__{paus[0]}", paus[0]) if n in self._ra_handlers), None)
                    if ny is not None:
                        logger.info(f"🎵 {function_name} on a speaker -> {ny} (a speaker cannot be turned off, it is paused)")
                        params.function_name, params.arguments = ny, paus[1]
                        target = self._ra_handlers[ny]
                timeout_seconds = _tool_timeout_seconds()
                try:
                    async with asyncio.timeout(timeout_seconds or None) as cap:  # 0 = no cap
                        return await target(params)
                except TimeoutError:
                    if not cap.expired():
                        raise  # the handler's own timeout, not our cap: the generic error path answers it
                    utfall = "fel"
                    log_timing("fel")
                    if not callback_used:
                        await params.result_callback({
                            "error": (
                                f"Verktyget svarade inte inom {timeout_seconds:g} sekunder. "
                                "Säg att det inte gick att bekräfta att det blev gjort, och "
                                "påstå aldrig att det är klart."
                            )
                        })
                    return
            except Exception as e:
                utfall = "fel"
                logger.warning(f"⚠️ {function_name} kastade: {e!r}", exc_info=True)  # the model only gets the type name
                if not callback_used:
                    try:
                        await params.result_callback(
                            {"error": f"Verktyget gav ett fel ({type(e).__name__}). Säg till användaren att det inte gick att bekräfta att det blev gjort, och påstå aldrig att det är klart."}
                        )
                    except Exception as cb_error:
                        logger.warning(f"⚠️ {function_name}: result_callback failed after tool exception: {cb_error!r}")
                        log_timing("fel")
                        raise e
                    return  # capped_callback has logged the outcome ("fel") once
                log_timing("fel")
                raise
            except BaseException:
                utfall = "fel"
                log_timing("fel")
                if (self.turn_liveness is not None
                        and self.turn_liveness.tool_was_cancelled(current_task)):
                    # A real barge-in. Pipecat keeps the call IN_PROGRESS until it gets a result,
                    # so close it quietly (run_llm False: the model must not start talking).
                    # A write may already have gone out, so never claim it did not happen.
                    if not callback_used:
                        try:
                            await asyncio.shield(original_callback(
                                {"result": "Avbröts av användaren. Det är inte bekräftat om det hann utföras."},
                                properties=FunctionCallResultProperties(run_llm=False),
                            ))
                        except Exception as cb_error:
                            logger.warning(f"⚠️ {function_name}: closing the interrupted call failed: {cb_error!r}")
                    return
                raise
            finally:
                stop_waiting_ack()
                if utfall != "fel":
                    log_timing("ok")
                if self.turn_liveness is not None:
                    self.turn_liveness.tool_finished(current_task)

        async def liveness_tracked(params):
            # Never register/cancel Pipecat's caller task (or a test's task).
            # The handler wrapper owns a dedicated task which a real barge-in
            # can safely cancel without affecting its caller's other work.
            return await asyncio.create_task(run_liveness_tracked(params))

        super().register_function(
            function_name, liveness_tracked, start_callback, cancel_on_interruption=False
        )
