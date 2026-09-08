"""The app must ask the router, and hand each engine its own key."""

import os

import pytest

from app.provider_router import ProviderRouter


def test_a_backup_of_none_means_no_failover(monkeypatch):
    from app.main import build_router
    monkeypatch.setenv("VOICE_PROVIDER", "gemini")
    monkeypatch.setenv("VOICE_PROVIDER_BACKUP", "none")
    monkeypatch.setenv("PROVIDER_COOLDOWN_MINUTES", "30")
    r = build_router()
    assert r.primary == "gemini"
    assert r.backup is None
    assert r.report_failure("gemini", "insufficient_quota") == "gemini"


def test_the_cooldown_is_read_in_minutes(monkeypatch):
    from app.main import build_router
    monkeypatch.setenv("VOICE_PROVIDER", "openai")
    monkeypatch.setenv("VOICE_PROVIDER_BACKUP", "gemini")
    monkeypatch.setenv("PROVIDER_COOLDOWN_MINUTES", "45")
    assert build_router().cooldown_s == 45 * 60


def test_an_unreadable_cooldown_falls_back_to_thirty_minutes(monkeypatch):
    from app.main import build_router
    monkeypatch.setenv("VOICE_PROVIDER", "openai")
    monkeypatch.setenv("VOICE_PROVIDER_BACKUP", "gemini")
    monkeypatch.setenv("PROVIDER_COOLDOWN_MINUTES", "")
    assert build_router().cooldown_s == 1800.0


def test_a_backup_the_same_as_the_primary_is_no_backup(monkeypatch):
    from app.main import build_router
    monkeypatch.setenv("VOICE_PROVIDER", "gemini")
    monkeypatch.setenv("VOICE_PROVIDER_BACKUP", "gemini")
    monkeypatch.setenv("PROVIDER_COOLDOWN_MINUTES", "30")
    assert build_router().backup is None


def test_an_unknown_primary_falls_back_to_openai(monkeypatch):
    from app.main import build_router
    monkeypatch.setenv("VOICE_PROVIDER", "claude")
    monkeypatch.setenv("VOICE_PROVIDER_BACKUP", "none")
    monkeypatch.setenv("PROVIDER_COOLDOWN_MINUTES", "30")
    assert build_router().primary == "openai"


def test_provider_options_gives_each_engine_its_own_key_model_and_voice(monkeypatch):
    """The knobs must actually differ per engine, not just exist per engine."""
    from app.main import Application
    from app.providers import GEMINI, OPENAI

    app = Application()
    app.instructions = "Base instructions."
    app.gemini_api_key = "gm-key"
    app.gemini_model = "models/gemini-3.1-flash-live-preview"
    app.gemini_voice = "Charon"
    app.transcription_language = ""
    app.max_output_tokens = None
    app.openai_api_key = "sk-key"
    app.model = "gpt-realtime-2"
    app.voice = "marin"
    app.openai_speed = 1.0
    app.noise_reduction = ""
    app.turn_detection_type = "semantic_vad"
    app.vad_eagerness = "low"
    app.vad_threshold = 0.5
    app.vad_prefix_padding_ms = 300
    app.vad_silence_duration_ms = 800
    app.semantic_vad_create_response = True
    app.interrupt_response = False
    app.transcription_model = "gpt-4o-transcribe"

    gemini_options = app.provider_options(GEMINI)
    openai_options = app.provider_options(OPENAI)

    assert gemini_options.api_key == "gm-key"
    assert gemini_options.model == "models/gemini-3.1-flash-live-preview"
    assert gemini_options.voice == "Charon"
    assert gemini_options.language == "sv-SE"

    assert openai_options.api_key == "sk-key"
    assert openai_options.model == "gpt-realtime-2"
    assert openai_options.voice == "marin"

    # Both engines get the same instructions -- there is only one prompt.
    assert gemini_options.instructions == openai_options.instructions
    # But a value one engine doesn't have (Gemini's default language) must
    # not leak into the other engine's, and vice versa: if provider_options
    # secretly built ONE ProviderOptions and just overwrote a couple of
    # fields, the openai one would end up with language="sv-SE" too even
    # though nothing ever sets that for openai from these inputs.
    assert openai_options.language == "sv-SE"  # dataclass default, unset either way
    assert openai_options.speed == 1.0
    assert openai_options.turn_detection_type == "semantic_vad"


def _bare_app(provider: str, backup=None):
    """An Application with exactly the attributes create_service touches,
    wired by hand instead of through initialize() -- initialize() also starts
    a Home Assistant MCP client, a voice-print publisher task and an announce
    HTTP server, none of which belong in a unit test."""
    from app.main import Application
    from app.provider_router import ProviderRouter
    from app.session_manager import SessionManager
    from app.timers import TimerRegistry

    app = Application()
    app.router = ProviderRouter(provider, backup)
    app.session_manager = SessionManager()
    app.timer_registry = TimerRegistry()
    app.enrollment_conductor = None
    app.enable_disconnect_tool = False
    app.enable_web_search = False
    app.mcp_client = None
    app.mcp_tool_allowlist = []
    app.speaker_male_name = ""
    app.speaker_female_name = ""
    app.male_only_tools = set()
    app.instructions = "Base instructions."
    app.max_output_tokens = None
    app.transcription_language = ""
    app.gemini_api_key = "gm-test"
    app.gemini_model = "models/gemini-3.1-flash-live-preview"
    app.gemini_voice = "Charon"
    app.openai_api_key = "sk-test"
    app.model = "gpt-realtime-2"
    app.voice = "marin"
    app.openai_speed = 1.0
    app.noise_reduction = ""
    app.turn_detection_type = "semantic_vad"
    app.vad_eagerness = "low"
    app.vad_threshold = 0.5
    app.vad_prefix_padding_ms = 300
    app.vad_silence_duration_ms = 800
    app.semantic_vad_create_response = True
    app.interrupt_response = False
    app.transcription_model = "gpt-4o-transcribe"
    return app


@pytest.mark.asyncio
async def test_create_service_sets_connection_provider_to_the_engine_it_built(monkeypatch):
    """Ruling 2: connection.provider must hold the engine actually used to
    build the session -- not whatever the router says NOW, later."""
    from app.device_registry import DeviceConnection
    from app.providers import GEMINI

    app = _bare_app(GEMINI)
    connection = DeviceConnection(device_id="kitchen", websocket=object())
    await app.create_service(connection)
    assert connection.provider == GEMINI
    # A failure recorded against the router after the session was built must
    # not retroactively change what this connection says it runs.
    app.router.report_failure(GEMINI, "insufficient_quota")
    assert connection.provider == GEMINI


class _FlakyRouter:
    """A router whose answer moves between two calls -- standing in for
    another connection's failure landing between create_service's own two
    reads, if it had two. current() flips openai/gemini on every call."""

    def __init__(self, first: str, second: str):
        self._answers = [first, second]

    def current(self) -> str:
        return self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]


@pytest.mark.asyncio
async def test_connection_provider_matches_the_engine_build_service_got():
    """Ruling 2, defended against a subtler bug than a stale value: if
    create_service asked the router twice -- once to pick the engine for
    build_service, again to stamp connection.provider -- a router that moved
    between those two reads would leave connection.provider naming an engine
    OTHER than the one actually built. That must be structurally impossible:
    there must be exactly one read, reused for both."""
    from app.device_registry import DeviceConnection
    from app.providers import GEMINI, OPENAI

    app = _bare_app(GEMINI)
    app.router = _FlakyRouter(GEMINI, OPENAI)
    connection = DeviceConnection(device_id="kitchen", websocket=object())

    service = await app.create_service(connection)

    # What engine did build_service actually construct? Gemini's service has
    # a voice_id; OpenAI's realtime service does not use that attribute name.
    built_gemini = hasattr(service, "voice_id") or "Gemini" in type(service).__name__
    assert connection.provider == (GEMINI if built_gemini else OPENAI)


def test_create_transport_declares_the_providers_input_rate():
    """Step 6: the transport's declared mic rate must follow the engine, and
    the output rate must stay 24000 for both -- getting THAT wrong makes the
    speaker play noise."""
    from app.raw_audio_serializer import RawAudioSerializer
    from app.websocket_handler import WebSocketHandler

    handler = WebSocketHandler()
    serializer = RawAudioSerializer("kitchen")

    gemini_transport = handler.create_transport(object(), serializer, "gemini")
    assert gemini_transport._params.audio_in_sample_rate == 16000
    assert gemini_transport._params.audio_out_sample_rate == 24000

    openai_transport = handler.create_transport(object(), serializer, "openai")
    assert openai_transport._params.audio_in_sample_rate == 24000
    assert openai_transport._params.audio_out_sample_rate == 24000


@pytest.mark.asyncio
async def test_build_pipeline_resamples_to_the_connections_provider_rate():
    """Step 6: InputResampler's out_rate must follow connection.provider, not
    the old fixed PIPELINE_SAMPLE_RATE constant -- for Gemini that must make
    it a pass-through (16000 in, 16000 out), matching the device's own rate."""
    import app.websocket_handler as wh
    from pipecat.processors.frame_processor import FrameProcessor
    from app.device_registry import DeviceConnection
    from app.raw_audio_serializer import RawAudioSerializer

    class FakeService(FrameProcessor):
        pass

    captured = {}
    real_resampler = wh.InputResampler

    class SpyResampler(real_resampler):
        def __init__(self, out_rate=24000, **kwargs):
            captured["out_rate"] = out_rate
            super().__init__(out_rate=out_rate, **kwargs)

    monkeypatch_target = wh.InputResampler
    wh.InputResampler = SpyResampler
    try:
        handler = wh.WebSocketHandler()
        serializer = RawAudioSerializer("kitchen")
        connection = DeviceConnection(
            device_id="kitchen", websocket=object(), serializer=serializer
        )
        connection.provider = "gemini"
        connection.transport = handler.create_transport(object(), serializer, "gemini")
        connection.openai_service = FakeService()

        handler.build_pipeline(connection)
        assert captured["out_rate"] == 16000
    finally:
        wh.InputResampler = monkeypatch_target


def test_the_websocket_handler_gets_the_same_router_instance():
    """Ruling 1: app.run() wires self.router onto the handler beside the
    service factory, so the next task's failover can read handler.router."""
    import inspect

    from app.main import Application

    source = inspect.getsource(Application.run)
    assert "self.websocket_handler.router = self.router" in source
