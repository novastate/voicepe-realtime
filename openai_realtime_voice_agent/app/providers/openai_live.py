"""OpenAI Live (gpt-live-1) as a voice engine (raawr US-025, 0.28.0).

Henrik's OpenAI key allows Live only; Realtime (openai_realtime.py) is shut
off on it. Live is a different protocol: full duplex, no turns, the model
decides itself when to speak, and tools run through "Responses delegation".
Verified against the docs 2026-10-04 (developers.openai.com primary-websocket
and live-delegation, Microsoft Foundry's GPT-Live event reference):

    client: session.start {session}  -> server: session.started
    client: session.input_audio.append {audio: base64 PCM16 mono}
    server: session.output_audio.delta {delta, start_ms, end_ms}   (24 kHz)
    server: session.input_transcript.delta / session.output_transcript.delta
    server: response.event {delegation_id, event: {type, ...}}
            -> event response.output_item.done {item: function_call}
    client: response.item.create {item: function_call_output} + response.create
    client: session.close -> server: session.closed

There is no "output audio done" event, so a gap of OPENAI_LIVE_REPLY_GAP_MS
without output audio ends a reply.

Kept like the other engines: asleep until the wake (SovlageMixin), bana 0
first (the turn's audio is held until the fast path has decided, as on
Gemini), local turn detection. On top, Henrik's two rules for this engine:
one OpenAI session at a time in the process (_LAS), and its own daily cap,
OPENAI_MAX_MINUTER_PER_DAG (6), so his 10 USD/month ceiling is never hit
mid-conversation.
"""
import asyncio
import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from pipecat.frames.frames import (
    FunctionCallResultFrame,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService

from app.providers.local_turns import LocalTurns, LocalTurnsMixin
from app.providers.sovlage import Budget, SovlageMixin
from app.providers.tool_registration import ToolRegistrationMixin

logger = logging.getLogger(__name__)

LIVE_URL = "wss://api.openai.com/v1/live/sessions"
DEFAULT_MODEL = "gpt-live-1"
OUT_RATE = 24000
IN_RATE = 16000


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def max_sekunder_openai() -> float:
    """OPENAI_MAX_MINUTER_PER_DAG (6): this engine's own ceiling, on top of the shared one."""
    return max(0.0, _env_float("OPENAI_MAX_MINUTER_PER_DAG", 6.0)) * 60.0


OPENAI_BUDGET = Budget(path=os.environ.get("MOLN_LEDGER_OPENAI", "/data/moln_minuter_openai.json"))


class _Las:
    """One OpenAI Live session in the whole process (Henrik's key allows one)."""

    agare = None

    @classmethod
    def ta(cls, service) -> bool:
        if cls.agare is not None and cls.agare is not service:
            return False
        cls.agare = service
        return True

    @classmethod
    def slapp(cls, service) -> None:
        if cls.agare is service:
            cls.agare = None


def session_config(options, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The whole session.start body, in one place.

    ponytail: the audio input format keys and the delegation block are
    written from the docs, not yet from a live probe. If session.start is
    refused ("unknown field"), fix it HERE and nowhere else.
    """
    functions = [
        {"type": "function", "name": t["name"], "description": t.get("description", ""),
         "parameters": t.get("parameters") or {"type": "object", "properties": {}}}
        for t in tools if t.get("name")
    ]
    responses = {
        "model": os.environ.get("OPENAI_LIVE_DELEGATION_MODEL", "").strip() or "gpt-5.5",
        "instructions": "Använd husets verktyg när det behövs. Svara kort och på svenska.",
        "tools": functions,
        "tool_choice": "auto",
        "max_output_tokens": int(options.max_output_tokens or 1024),
    }
    return {
        "model": options.model or DEFAULT_MODEL,
        "instructions": options.instructions,
        "audio": {
            "input": {"format": {"type": "audio/pcm", "rate": IN_RATE}},
            "output": {"voice": options.voice or "marin"},
        },
        "delegation": {"type": "responses", "responses": responses},
    }


class OpenAILiveService(SovlageMixin, LocalTurnsMixin, ToolRegistrationMixin, LLMService):
    """gpt-live-1 behind the same doors as the other engines."""

    # Gemini helpers look for these and skip quietly when the session is None:
    # Live has no silent context restore and no speaker note (yet).
    _session = None
    _context = None
    _needs_turn_complete_message = False

    # Bana 0 hooks, same as Gemini (see gemini_live.py).
    on_user_turn_start = None
    on_user_turn_end = None
    _turn_end_task = None
    _held: Optional[List[bytes]] = None
    _activity_open = False

    def __init__(self, *, api_key: str, config: Dict[str, Any], url: str = LIVE_URL, **kwargs):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._config = config
        self._url = url
        self._ws = None
        self._started = asyncio.Event()
        self._reader = None
        self._reply_open = False
        self._reply_end_task = None
        self._user_text = ""
        self._on_turn_complete = None
        self._closing = False
        self.vagran = None  # why the last wake was refused: "las" / "budget"

    # --- lifecycle -------------------------------------------------------------

    async def start(self, frame):
        await super().start(frame)
        await self._connect()  # a no-op while asleep (SovlageMixin)

    async def stop(self, frame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        if self.sover or self._ws is not None:
            return
        import websockets

        self._started.clear()
        self._closing = False
        try:
            self._ws = await websockets.connect(
                self._url, additional_headers={"Authorization": f"Bearer {self._api_key}"},
                max_size=None,
            )
            await self._send({"type": "session.start", "session": self._config})
            self._reader = asyncio.get_running_loop().create_task(self._read_loop())
        except Exception as e:
            logger.error(f"❌ OpenAI Live connect failed: {e!r}")
            await self._close_socket()

    async def _disconnect(self):
        _Las.slapp(self)
        self._closing = True
        if self._ws is not None and self._started.is_set():
            try:
                await self._send({"type": "session.close"})
            except Exception:
                pass
        await self._close_socket()

    async def _close_socket(self):
        ws, self._ws = self._ws, None
        self._started.clear()
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        reader, self._reader = self._reader, None
        if reader is not None and reader is not asyncio.current_task():
            reader.cancel()

    # --- SovlageMixin --------------------------------------------------------------

    async def _ateranslut(self, forut: bool) -> None:
        await self._connect()  # Live has no resumption: always a fresh session

    async def _ar_uppkopplad(self, timeout: float = 3.0) -> bool:
        # Under ConnectionRecovery.VAKNA_TIMEOUT_S (5 s), so a session that never
        # starts lands in SovlageMixin's "asleep again" here, not in a cancel.
        try:
            await asyncio.wait_for(self._started.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return self._ws is not None

    def over_budget(self) -> bool:
        return super().over_budget() or (
            OPENAI_BUDGET.anvant() + self.oppen_tid() >= max_sekunder_openai()
        )

    async def vakna(self) -> bool:
        if not self.sover:
            return False
        self.vagran = None
        if OPENAI_BUDGET.anvant() >= max_sekunder_openai():
            self.vagran = "budget"
            logger.warning(f"💸 OpenAI Live: today's {max_sekunder_openai() / 60:.0f} min used — not connecting")
            return False
        if not _Las.ta(self):
            self.vagran = "las"
            logger.warning("🔒 OpenAI Live: the other speaker holds the one session — not connecting")
            return False
        ok = await super().vakna()
        if not ok:
            _Las.slapp(self)
        return ok

    async def sova(self, reason: str) -> bool:
        tid = self.oppen_tid()
        slept = await super().sova(reason)
        if slept:
            OPENAI_BUDGET.lagg_till(tid)
        _Las.slapp(self)
        return slept

    # --- wire ------------------------------------------------------------------------

    async def _send(self, event: Dict[str, Any]) -> None:
        if self._ws is None:
            return
        await self._ws.send(json.dumps(event))

    async def _append(self, pcm: bytes) -> None:
        if pcm and self._started.is_set():
            await self._send({"type": "session.input_audio.append",
                              "audio": base64.b64encode(pcm).decode()})

    async def _read_loop(self):
        try:
            async for raw in self._ws:
                try:
                    await self._handle(json.loads(raw))
                except Exception as e:
                    logger.warning(f"⚠️ OpenAI Live event not handled: {e!r}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self.sover:
                logger.warning(f"⚠️ OpenAI Live socket ended: {e!r}")
        if not self.sover and not self._closing:
            # Never reconnect on our own: asleep, so the next wake connects.
            asyncio.get_running_loop().create_task(self.sova("OpenAI Live socket closed"))

    async def _handle(self, ev: Dict[str, Any]) -> None:
        typ = ev.get("type")
        if typ == "session.started":
            self._started.set()
        elif typ == "session.output_audio.delta":
            await self._audio_out(base64.b64decode(ev.get("delta") or ""))
        elif typ == "session.output_transcript.delta":
            if ev.get("delta"):
                await self.push_frame(TTSTextFrame(text=ev["delta"], aggregated_by="sentence"))
        elif typ == "session.input_transcript.delta":
            self._user_text += ev.get("delta") or ""
        elif typ == "response.event":
            inner = ev.get("event") or {}
            item = inner.get("item") or {}
            if inner.get("type") == "response.output_item.done" and item.get("type") == "function_call":
                await self._tool_call(item)
        elif typ == "error":
            logger.warning(f"⚠️ OpenAI Live error: {ev.get('error')}")
        elif typ == "session.closed":
            logger.info(f"OpenAI Live session closed: {ev.get('reason')} {ev.get('usage')}")

    async def _audio_out(self, pcm: bytes) -> None:
        if not pcm:
            return
        if not self._reply_open:
            self._reply_open = True
            self.cancel_silence_ack()
            await self._flush_user_text()
            await self.push_frame(TTSStartedFrame())
            await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(TTSAudioRawFrame(audio=pcm, sample_rate=OUT_RATE, num_channels=1))
        if self._reply_end_task is not None:
            self._reply_end_task.cancel()
        self._reply_end_task = asyncio.get_running_loop().create_task(self._end_reply_later())

    async def _end_reply_later(self):
        await asyncio.sleep(_env_float("OPENAI_LIVE_REPLY_GAP_MS", 1500.0) / 1000.0)
        self._reply_end_task = None
        if getattr(self, "_verktyg_pagar", 0) > 0:
            return  # a tool is still running: the answer comes after it
        self._reply_open = False
        await self.push_frame(TTSStoppedFrame())
        await self.push_frame(LLMFullResponseEndFrame())
        if self._on_turn_complete is not None:
            try:
                await self._on_turn_complete()
            except Exception as e:
                logger.warning(f"⚠️ turn-complete handler failed: {e!r}")

    def set_turn_complete_handler(self, handler) -> None:
        self._on_turn_complete = handler

    async def _flush_user_text(self) -> None:
        text, self._user_text = self._user_text.strip(), ""
        if text:
            await self.push_frame(
                TranscriptionFrame(text=text, user_id="", timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")),
                FrameDirection.UPSTREAM,
            )

    # --- tools -----------------------------------------------------------------------

    async def _tool_call(self, item: Dict[str, Any]) -> None:
        try:
            args = json.loads(item.get("arguments") or "{}")
        except ValueError:
            args = {}
        call = FunctionCallFromLLM(function_name=item.get("name", ""), tool_call_id=item.get("call_id", ""),
                                   arguments=args, context=self._context)
        # Off the reader: a slow tool must not stop the socket being read. While
        # it runs the reply is not over, however long the silence (review of US-025).
        self._verktyg_pagar = getattr(self, "_verktyg_pagar", 0) + 1
        task = asyncio.get_running_loop().create_task(self.run_function_calls([call]))
        self._verktyg_tasks = getattr(self, "_verktyg_tasks", set())
        self._verktyg_tasks.add(task)
        task.add_done_callback(self._verktyg_tasks.discard)

    async def broadcast_frame(self, frame_cls, **kwargs):
        """pipecat hands a tool's result back by broadcasting a frame; send it to Live too."""
        await super().broadcast_frame(frame_cls, **kwargs)
        if frame_cls is FunctionCallResultFrame:
            self._verktyg_pagar = max(0, getattr(self, "_verktyg_pagar", 0) - 1)
            result = kwargs.get("result")
            output = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            await self._send({"type": "response.item.create", "item": {
                "type": "function_call_output", "call_id": kwargs.get("tool_call_id"), "output": output}})
            await self._send({"type": "response.create"})

    # --- the microphone ----------------------------------------------------------------

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            await self._send_user_audio(frame)
        if isinstance(frame, LLMContextFrame):
            return  # Live keeps its own conversation
        await self.push_frame(frame, direction)

    async def _send_user_audio(self, frame) -> None:
        turns = self._turns
        if turns is None:
            return await self._append(frame.audio)
        event = turns.feed(frame.audio, frame.sample_rate)
        if self._held is not None:
            self._held.append(frame.audio)
            if event == "end":
                await self.push_frame(UserStoppedSpeakingFrame())
                self._decide_turn()
            return
        if not self._activity_open:
            if event != "start":
                self._keep_preroll(frame)
                return
            preroll = self._take_preroll()
            await self.push_frame(UserStartedSpeakingFrame())
            if self.on_user_turn_start is not None:
                self.on_user_turn_start()
            if self.on_user_turn_end is not None:
                self._held = [preroll, frame.audio]
                return
            self._activity_open = True
            await self._append(preroll)
        await self._append(frame.audio)
        if event == "end":
            await self.push_frame(UserStoppedSpeakingFrame())
            await self._end_activity()

    async def _end_activity(self) -> None:
        self._activity_open = False
        # ponytail: Live hears only what we append; trailing silence is how it
        # can tell he stopped. 600 ms, tune with OPENAI_LIVE_TAIL_MS after a probe.
        tail = int(_env_float("OPENAI_LIVE_TAIL_MS", 600.0) * IN_RATE / 1000) * 2
        await self._append(b"\x00" * tail)
        self.arm_silence_ack()

    def _decide_turn(self) -> None:
        if self._turn_end_task is None or self._turn_end_task.done():
            self._turn_end_task = asyncio.get_running_loop().create_task(self._run_turn_end())

    async def _run_turn_end(self) -> None:
        try:
            await self.on_user_turn_end()
        finally:
            if self._held:
                await self.answer_turn()

    async def answer_turn(self) -> None:
        """Bana 0 missed: give Live the held turn."""
        held, self._held = self._held, None
        if not held:
            return
        for pcm in held:
            await self._append(pcm)
        await self._end_activity()

    async def drop_turn(self) -> None:
        """Bana 0 hit (or the device dropped the input): Live never hears this turn."""
        self._held = None
        self._activity_open = False
        self._preroll = bytearray()
        self.cancel_silence_ack()
        if self._turns is not None:
            self._turns.reset()

    async def end_audio_stream(self, keep_speech: bool = False) -> str:
        if keep_speech and self._held:
            self._decide_turn()
            return "held turn decided"
        await self.drop_turn()
        return "held audio dropped"


def build(options, tools):
    """Build a configured, unconnected OpenAI Live session for one device."""
    service = OpenAILiveService(api_key=options.api_key, config=session_config(options, tools))
    service._turns = LocalTurns.create(int(options.gemini_turn_silence_ms))
    logger.info(f"🎚️ openai_live: {options.model or DEFAULT_MODEL} voice={options.voice or 'marin'} "
                f"turn detection LOCAL, tools={len(tools)} via Responses delegation")
    return service
