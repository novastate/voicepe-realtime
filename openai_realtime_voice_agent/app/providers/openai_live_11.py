"""gpt-live-1.1, native full duplex, behind its own flag (VOICE_PROVIDER=openai_live_11). Prototype, branch live-gpt-11.

What differs from openai_live.py (gpt-live-1 with local turns and Responses delegation):

- AUDIO-01/CONV-02: the microphone is streamed CONTINUOUSLY while the session is open, also while Björn speaks. No
  local turn detection decides what Live hears, no bana 0 hold, no trailing-silence trick. When the device sends
  nothing (between follow-ups), silence is sent at real-time pace so the stream never stalls.
- Live runs the conversation. The local detector (`_fas_vad`) is only used to tell the device "listening"/"thinking"
  (its follow-up window closes otherwise) and for telemetry -- the same role it has under GEMINI_EGEN_TUR.
- AUDIO-03: Live has no interruption or "audio done" event on the primary WebSocket (live-conversations,
  voice-websockets?api=live). When Live goes silent in the middle of an open reply while the user is speaking, the
  app's own queue is the only thing still talking: it is dropped (pipecat InterruptionFrame, and "listening" makes the
  device clear its ring). Live is never told anything: there is no truncation API to invent.
- TASK-01..03: client delegation (`delegation.type = "client"`). Live decides when to delegate; the app builds the
  task from the user's transcript and hands it to the existing backend, Comms `/fraga` -> Core (app/live_delegering.py).
- CTX-01: a new session within LIVE11_MINNE_S of the last one is seeded with the recent dialogue and the operations'
  state through `session.input` (text-only history, documented for session.start).
- OPS-02: monotonic telemetry lines (`📈 live11 ...`) with ids, durations and lengths, never words.

Verified against the official docs and the OpenAI Python SDK's generated types (openai 3.60.0, openai/types/live/*),
2026-11-28: model "gpt-live-1.1"; audio.format {type: "audio/pcm", rate: 16000|24000}; delegation
{type: "client", acknowledgement_delay_ms}; session.start `input` items; session.delegation.created; thinking/commentary
appends with delegation_id (required, nullable); session.usage.updated; session.closed reasons. wake_policy and
idle_timeout_s are WebRTC-only (MediaSessionConfig) and are not used here.
"""
import asyncio
import base64
import json
import logging
import os
import statistics
import time
from typing import Any, Dict, List, Optional

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from app.live_delegering import Delegering
from app.providers.local_turns import LocalTurns
from app.providers.openai_live import IN_RATE, OUT_RATE, OpenAILiveService, _env_float

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-live-1.1"

# Section 5 of the requirements (Bjorn_Live_Voice_Requirements v2.0), with the three headings the Live prompting
# guide asks to keep. It goes AFTER Björn's own text (the persona from Core, bjorn_karna), never instead of it.
LIVE11_POLICY = """

SAMTAL I HÖGTALAREN: Tala naturlig svenska. Var kort, konkret och lugn. Utveckla när användaren vill ha mer.

Backchannel policy: Använd sparsamma, naturliga lyssningsljud när det hjälper. Ta inte över samtalet medan
användaren formulerar sin tanke. Ett kort mm eller ja från användaren kan vara en bekräftelse, inte en ny fråga.

Interruption policy: När användaren tar ordet, sluta tala och lyssna. Kräv inget stoppord. Vänta genom tankepauser
och upprepade ord. Om användaren rättar eller ändrar sin fråga, utgå från den senaste avsikten. Fortsätt inte
automatiskt det avbrutna svaret. Reagera inte på eget högtalarljud, tv, musik eller prat som inte riktar sig till dig.

Delegation policy: Husets backend kan: styra huset (ljus, media, scener, värme, timers), läsa husets läge,
minnet om familjen, kalendern, väder, nyheter, sport, sökning på nätet och djupare analys. Delegera när frågan
kräver aktuell information, en åtgärd i huset, minne eller analys. Delegera inte enkla samtalsfrågor, småprat eller
frågor om det som just sagts. Fortsätt lyssna medan arbetet pågår. Ge relevanta uppdateringar och återge verifierade
resultat. Påstå aldrig att en åtgärd lyckats innan backend har bekräftat det. Skilj på att sluta prata och att
avbryta en uppgift: ber användaren dig avbryta uppgiften, delegera det. Lås, larm och garage sköter du aldrig själv."""

_SPAR = "📈 live11"


def _p(varden: List[float], q: float) -> Optional[float]:
    if not varden:
        return None
    if len(varden) == 1:
        return varden[0]
    return statistics.quantiles(varden, n=100, method="inclusive")[int(q) - 1]


class Matning:
    """OPS-02: what one session measured, on the monotonic clock. Generated is not played: `spelat_s` is an
    estimate from the push times (the device reports no playback position; see the requirements' AUDIO-03)."""

    def __init__(self):
        self.svarstid: List[float] = []      # end of the user's speech (local detector) -> first sound of the reply
        self.avbrottstid: List[float] = []   # user speech start over a reply -> the app dropped the old audio
        self.avbrott = 0
        self.kastat_s = 0.0
        self.genererat_s = 0.0
        self.fel = 0
        self.mic_under_svar = 0              # AUDIO-01 on the real device: mic frames that came while he spoke

    def rad(self) -> str:
        def f(v):
            return "-" if v is None else f"{v * 1000:.0f}ms"
        return (f"svar n={len(self.svarstid)} p50={f(_p(self.svarstid, 50))} p95={f(_p(self.svarstid, 95))} | "
                f"avbrott n={self.avbrott} p50={f(_p(self.avbrottstid, 50))} p95={f(_p(self.avbrottstid, 95))} | "
                f"genererat={self.genererat_s:.1f}s kastat={self.kastat_s:.1f}s fel={self.fel} "
                f"mic_under_svar={self.mic_under_svar}")


class OpenAILive11Service(OpenAILiveService):
    """gpt-live-1.1: continuous audio, Live's own turn-taking, client delegation to Comms/Core."""

    live11 = True
    PREBUF_ENV, PREBUF_DEFAULT = "LIVE11_PREBUF_S", 0.3   # AUDIO-03: a short queue is one that can be interrupted
    _fas_vad: Optional[LocalTurns] = None
    slapp_ljud = None        # set by the handler: False = nobody woke him and no reply is recent -> silence is sent
    svarar = None            # set by the handler: True while his reply is playing
    on_motor_avbrott = None  # set by the handler: called when the app dropped the old reply's audio
    vakt = None              # set by the handler: text -> reason a delegation must not run now (echo guard), or None

    def __init__(self, *, options, **kwargs):
        super().__init__(**kwargs)
        self._options = options
        self._spar = f"{_SPAR}[{getattr(options, 'rum', '') or '-'}]"   # one process serves every room: tag the lines
        self._turns = None                      # never local turns: Live hears everything
        self.delegering = Delegering(self._send, vakt=lambda t: self.vakt(t) if self.vakt else None,
                                     tagg=getattr(options, "rum", "") or "-")
        self.historik: List[Dict[str, str]] = []   # CTX-01: [{"roll", "text", "status"}], newest last
        self._anv = ""                          # user words since the last delegation
        self._anv_tur = ""                      # user words of the current utterance (for the history)
        self._bjorn = ""                        # Björn's words of the current reply
        self._senast_anv_delta_t = 0.0
        self._senast_aktiv_t = 0.0
        self._senast_append_t = 0.0
        self._fore_start: List[tuple] = []      # (t, pcm) mic audio before session.started (T19), bounded
        self._fyll_task = None
        self._fas_talar = False
        self._tal_start_t = 0.0
        self._tal_slut_t = 0.0
        self._spelstart_t = 0.0
        self._avbrott_t = 0.0                   # we cut him on text alone: no local "end" will close "listening"
        self._delegeringar = set()
        self.matning = Matning()
        self._ateransluten = False

    # --- config ----------------------------------------------------------------------------------------------------

    def session_config(self) -> Dict[str, Any]:
        o = self._options
        cfg: Dict[str, Any] = {
            "model": o.model or DEFAULT_MODEL,
            "instructions": (o.instructions or "") + LIVE11_POLICY,
            "audio": {"format": {"type": "audio/pcm", "rate": IN_RATE}, "output": {"voice": o.voice or "marin"}},
            "delegation": {"type": "client",
                           "acknowledgement_delay_ms": int(_env_float("LIVE11_ACK_MS", 800.0))},
        }
        seed = self._seed()
        if seed:
            cfg["input"] = seed
        return cfg

    def _seed(self) -> List[Dict[str, Any]]:
        """Recent dialogue and the operations' state, as text history (<= 128 messages, 8192 tokens by default)."""
        if not self.historik or time.monotonic() - self._senast_aktiv_t > _env_float("LIVE11_MINNE_S", 600.0):
            return []
        items: List[Dict[str, Any]] = []
        budget = 6000
        for h in reversed(self.historik[-16:]):
            text = h["text"][:600]
            if h.get("status") == "avbrutet":
                text += " [avbröts; exakt vad som hördes är osäkert]"
            budget -= len(text)
            if budget < 0:
                break
            if h["roll"] == "user":
                items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})
            else:
                items.append({"type": "message", "role": "assistant",
                              "content": [{"type": "output_text", "text": text}]})
        items.reverse()
        ops = self.delegering.sammanfattning()
        if ops:
            items.append({"type": "message", "role": "developer", "content": [{"type": "input_text", "text":
                          "Samtalet fortsätter efter en ny anslutning. Uppgifternas läge (fakta från appen):\n" + ops}]})
        return items

    async def _connect(self):
        if self.sover or self._ws is not None:
            return
        self._config = self.session_config()   # rebuilt on every connect: the seed changes
        self.delegering.ny_session()
        await super()._connect()

    # --- server events ---------------------------------------------------------------------------------------------

    async def _handle(self, ev: Dict[str, Any]) -> None:
        typ = ev.get("type")
        if typ == "session.started":
            self._session_t0 = time.monotonic()
            self._started.set()
            logger.info(f"{self._spar} session start gen={self.delegering.gen} seed={len(self._config.get('input') or [])}")
            await self._slapp_fore_start()
            self._starta_fyllning()
        elif typ == "session.output_audio.delta":
            await self._audio_out(base64.b64decode(ev.get("delta") or ""))
        elif typ == "session.output_transcript.delta":
            d = ev.get("delta") or ""
            if d:
                self._bjorn += d
                self._svarsprak(d)
                await self.push_frame(TTSTextFrame(text=d, aggregated_by="sentence"))
        elif typ == "session.input_transcript.delta":
            d = ev.get("delta") or ""
            if d:
                if self._bjorn.strip() and not self._reply_open:
                    self._historik_bjorn("klar")   # an open reply goes in when it ends or is cut
                self._anv += d
                self._anv_tur += d
                self._user_text += d
                self._senast_anv_delta_t = self._senast_aktiv_t = time.monotonic()
                await self._kolla_avbrott("transkript")
        elif typ == "session.delegation.created":
            # Off the reader: the wait for the transcript must not hold the audio back.
            t = asyncio.get_running_loop().create_task(self._delegering(ev))
            self._delegeringar.add(t)
            t.add_done_callback(self._delegeringar.discard)
        elif typ in ("session.commentary.appended", "session.thinking.appended", "session.instructions.appended"):
            logger.debug(f"{self._spar} {typ} {ev.get('client_event_id')}")
        elif typ == "session.usage.updated":
            u = ev.get("usage") or {}
            cw = (ev.get("context_window") or {}).get("usage_ratio")
            logger.info(f"{self._spar} usage s={u.get('seconds')} aktiv={u.get('active_seconds')} kontext={cw}")
        elif typ == "error":
            self.matning.fel += 1
            err = ev.get("error") or {}
            logger.warning(f"⚠️ live11 error code={err.get('code')} param={err.get('param')} "
                           f"event={err.get('client_event_id')}")
            await super()._handle(ev)   # rate-limit retry on session.start stays the parent's
        elif typ == "session.closed":
            logger.info(f"{self._spar} session stängd orsak={ev.get('reason')} usage={ev.get('usage')} | {self.matning.rad()}")
        else:
            self._logga_handelse(ev)

    async def _delegering(self, ev: Dict[str, Any]) -> None:
        d = ev.get("delegation") or {}
        did = str(d.get("id") or "")
        if d.get("target") not in (None, "client") or not did:
            return
        # The event carries no task text. The transcript of what was said can come a moment after it: wait briefly.
        await asyncio.sleep(_env_float("LIVE11_TEXT_VANTA_MS", 400.0) / 1000.0)
        text, self._anv = self._anv.strip(), ""
        await self.delegering.skapad(did, text)

    def tur_text(self) -> str:
        return self._anv_tur

    def _tider_start(self) -> None:
        if self.tider is not None and self._fas_vad is not None:
            self.tider.start(getattr(self._fas_vad, "silence_s", 0.8))

    # --- output ----------------------------------------------------------------------------------------------------

    async def _audio_out(self, pcm: bytes) -> None:
        if pcm and not any(pcm):
            # Live streams exact zeros while it says nothing. In an open reply, zeros while the user speaks = Live
            # has yielded; what is still queued here is the old answer.
            await self._kolla_avbrott("tystnad")
            return
        if pcm and not self._reply_open:
            self._spelstart_t = time.monotonic()
            if self._anv_tur.strip():
                self._historik_anv()
            if self._tal_slut_t and self._tal_slut_t > self._senast_svar_t():
                self.matning.svarstid.append(time.monotonic() - self._tal_slut_t)
        if pcm:
            self.matning.genererat_s += len(pcm) / (2.0 * OUT_RATE)
            self._senast_aktiv_t = time.monotonic()
        await super()._audio_out(pcm)

    def _senast_svar_t(self) -> float:
        return getattr(self, "_forra_svar_t", 0.0)

    async def _ljud_ut(self, pcm: bytes) -> None:
        self._forra_svar_t = time.monotonic()
        await super()._ljud_ut(pcm)

    def _anvandaren_talar(self) -> bool:
        """The user has taken the floor DURING this reply. Speech (or its transcript) that began before the reply's first
        sound is the question being answered: the transcript lags the audio, and the local detector's "end" comes
        silence_ms after the last word. Neither may cut the answer it just started."""
        lokalt = bool(self._fas_vad is not None and getattr(self._fas_vad, "_speaking", False)
                      and self._tal_start_t > self._spelstart_t)
        text = (time.monotonic() - self._senast_anv_delta_t < _env_float("LIVE11_AVBROTT_TEXT_S", 1.0)
                and self._senast_anv_delta_t - self._spelstart_t > _env_float("LIVE11_TEXT_EFTER_S", 0.8))
        return lokalt or text

    async def _kolla_avbrott(self, kalla: str) -> None:
        """Drop the old reply's queued audio when Live has stopped generating it and the user has the floor."""
        if not self._reply_open:
            return
        tyst_s = time.monotonic() - self._sista_ljud_t
        if tyst_s < _env_float("LIVE11_AVBROTT_TYST_MS", 200.0) / 1000.0 or not self._anvandaren_talar():
            return
        await self._avbryt_uppspelning(kalla)

    async def _avbryt_uppspelning(self, kalla: str) -> None:
        nu = time.monotonic()
        # What could have been heard: from the first push plus the device's prebuffer, at real time. An estimate.
        spelat = max(0.0, nu - self._spelstart_t - _env_float("PLAYBACK_PREBUFFER_MS", 150.0) / 1000.0)
        genererat = getattr(self, "_ljud_bytes", 0) / (2.0 * OUT_RATE)
        kastat = max(0.0, genererat - spelat)
        self.matning.avbrott += 1
        self.matning.kastat_s += kastat
        if self._tal_start_t and self._tal_start_t > self._spelstart_t:
            self.matning.avbrottstid.append(nu - self._tal_start_t)
        logger.info(f"{self._spar} avbrott kalla={kalla} genererat={genererat:.1f}s spelat~{spelat:.1f}s kastat~{kastat:.1f}s "
                    f"tal_for={(nu - self._tal_start_t) if self._tal_start_t else -1:.2f}s")
        if self._reply_end_task is not None:
            self._reply_end_task.cancel()
            self._reply_end_task = None
        self._pre, self._pre_s = None, 0.0
        if self._pre_task is not None:
            self._pre_task.cancel()
            self._pre_task = None
        self._reply_open = False
        if self._bjorn.strip():
            self._historik_bjorn("avbrutet")
        if self.on_motor_avbrott is not None:
            self.on_motor_avbrott()
        await self.push_frame(InterruptionFrame())
        if not self._fas_talar:
            self._fas_talar = True
            await self.push_frame(UserStartedSpeakingFrame())   # "listening": the device clears its own ring
        talar = bool(self._fas_vad is not None and getattr(self._fas_vad, "_speaking", False))
        self._avbrott_t = 0.0 if talar else nu

    # --- history (CTX-01) -------------------------------------------------------------------------------------------

    def _historik_anv(self) -> None:
        text, self._anv_tur = self._anv_tur.strip(), ""
        if text:
            self.historik = self.historik[-40:] + [{"roll": "user", "text": text, "status": "hort"}]

    def _historik_bjorn(self, status: str) -> None:
        text, self._bjorn = self._bjorn.strip(), ""
        if text:
            # Generated words. 'avbrutet' = the reply was cut; which of them were heard is not known exactly.
            self.historik = self.historik[-40:] + [{"roll": "assistant", "text": text, "status": status}]

    async def _end_reply_later(self):
        await super()._end_reply_later()
        if not self._reply_open and self._bjorn.strip():
            self._historik_bjorn("klar")

    # --- input -----------------------------------------------------------------------------------------------------

    async def process_frame(self, frame, direction: FrameDirection):
        # Not the parent's: no local-turn path. The LLMService base still sees every frame.
        await super(OpenAILiveService, self).process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            await self._mic(frame)
        if isinstance(frame, LLMContextFrame):
            return
        await self.push_frame(frame, direction)

    async def _mic(self, frame) -> None:
        pcm = frame.audio
        if self._reply_open and any(pcm):
            self.matning.mic_under_svar += 1   # 0 after a long reply = the device's mic is shut while he speaks
        if self.slapp_ljud is not None and not self.slapp_ljud():
            pcm = b"\x00" * len(pcm)   # nobody woke him: the room is not sent, the stream stays continuous
        else:
            await self._fas(frame)
        if not self._started.is_set():
            self._fore_start.append((time.monotonic(), pcm))
            while sum(len(b) for _, b in self._fore_start) > int(_env_float("LIVE11_FORE_START_S", 3.0) * IN_RATE * 2):
                self._fore_start.pop(0)
            return
        await self._append(pcm)

    async def _append(self, pcm: bytes) -> None:
        if len(pcm) % 2:
            pcm = pcm[:-1]   # whole 16-bit samples only (voice-websockets?api=live)
        self._senast_append_t = time.monotonic()
        await super()._append(pcm)

    async def _slapp_fore_start(self) -> None:
        """T19: what was said while the session was starting is not lost. Live drops appends faster than 1.2x real
        time beyond a ~5 s burst (probed on gpt-live-1, 2026-10-10); this is at most LIVE11_FORE_START_S (3 s)."""
        # Only what was said during this start: audio from before an earlier sleep is not this conversation.
        grans = time.monotonic() - _env_float("LIVE11_FORE_START_S", 3.0) - 2.0
        held, self._fore_start = [b for t, b in self._fore_start if t >= grans], []
        for pcm in held:
            await self._append(pcm)
        if held:
            logger.info(f"{self._spar} före start: {sum(len(b) for b in held) / (2.0 * IN_RATE):.2f}s skickat")

    def _starta_fyllning(self) -> None:
        if self._fyll_task is None or self._fyll_task.done():
            self._fyll_task = asyncio.get_running_loop().create_task(self._fyll())

    async def _fyll(self) -> None:
        """Silence at real-time pace whenever the device sends nothing, so Live's input never stalls (AUDIO-01)."""
        steg = 0.1
        tyst = b"\x00" * int(steg * IN_RATE) * 2
        try:
            while self._ws is not None and self._started.is_set():
                await asyncio.sleep(steg)
                if time.monotonic() - self._senast_append_t > 1.5 * steg:
                    await self._append(tyst)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"live11 silence filler stopped: {e!r}")

    async def _fas(self, frame) -> None:
        """The device's phases only (and telemetry): the local detector decides nothing about the conversation."""
        if self._fas_vad is None:
            return
        event = self._fas_vad.feed(frame.audio, frame.sample_rate)
        if (self._fas_talar and self._avbrott_t and not getattr(self._fas_vad, "_speaking", False)
                and time.monotonic() - self._avbrott_t > 1.5):
            # Cut on the transcript alone and the local detector never heard speech: close "listening" anyway.
            self._avbrott_t, self._fas_talar = 0.0, False
            await self.push_frame(UserStoppedSpeakingFrame())
        if event == "start":
            self._avbrott_t = 0.0
            self._tal_start_t = time.monotonic()
            if self._reply_open:
                await self._kolla_avbrott("lokal")   # Live may already have yielded
            elif not self._fas_talar:
                self._fas_talar = True
                await self.push_frame(UserStartedSpeakingFrame())
        elif event == "end":
            self._tal_slut_t = time.monotonic()
            if self._fas_talar:
                self._fas_talar = False
                self._tider_start()
                await self.push_frame(UserStoppedSpeakingFrame())

    # --- bana 0 and the device's flush: nothing to hold, nothing to drop --------------------------------------------

    async def answer_turn(self) -> None:
        return None

    async def drop_turn(self) -> None:
        return None

    async def end_audio_stream(self, keep_speech: bool = False) -> str:
        return "live11: continuous stream, nothing held"

    # --- link loss (T13) ---------------------------------------------------------------------------------------------

    async def _read_loop(self):
        try:
            async for raw in self._ws:
                try:
                    await self._handle(json.loads(raw))
                except Exception as e:
                    logger.warning(f"⚠️ live11 event not handled: {e!r}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self.sover:
                logger.warning(f"⚠️ live11 socket ended: {type(e).__name__}")
        if self.sover or self._closing:
            return
        if self._reply_open:
            await self._avbryt_uppspelning("bortfall")   # no stale audio of a dead session
        aktiv = time.monotonic() - self._senast_aktiv_t < _env_float("LIVE11_ATERANSLUT_AKTIV_S", 15.0)
        if aktiv and not self._ateransluten and os.environ.get("LIVE11_ATERANSLUT", "1") != "0":
            # Mid-conversation: one new session at once, seeded with the dialogue and the operations (no resume in
            # the WebSocket API; fork needs store=true). Pending operations keep running; their results come in as
            # context of the new session (live_delegering, gen check).
            self._ateransluten = True
            logger.info(f"{self._spar} återansluter efter bortfall (seed av historik)")
            asyncio.get_running_loop().create_task(self._ateranslut_nu())
            return
        asyncio.get_running_loop().create_task(self.sov_begaran("OpenAI Live 1.1 socket closed"))

    async def _ateranslut_nu(self) -> None:
        await self._close_socket()
        await self._connect()
        if not await self._ar_uppkopplad():
            await self.sov_begaran("OpenAI Live 1.1: reconnect failed")
        else:
            self._ateransluten = False

    async def _disconnect(self):
        if self._fyll_task is not None:
            self._fyll_task.cancel()
            self._fyll_task = None
        self._fore_start = []
        logger.info(f"{self._spar} sammanfattning {self.matning.rad()} ops={len(self.delegering.uppdrag)}")
        await super()._disconnect()


def build(options, tools):
    """Tools are not given to Live: in client delegation the backend owns them (Core, through Comms)."""
    service = OpenAILive11Service(options=options, api_key=options.api_key, config={})
    service._fas_vad = LocalTurns.create(int(options.gemini_turn_silence_ms) or 800)
    logger.info(f"🎚️ openai_live_11[{getattr(options, 'rum', '') or '-'}]: {options.model or DEFAULT_MODEL} voice={options.voice or 'marin'} full duplex, "
                f"Live's own turn-taking, client delegation to Comms/Core (local tools unused: {len(tools)})")
    return service
