"""The microphone stays open because the model asked, not because a timer said so.

Before this, `follow_up_ms` was sent once at connect and the device applied it
after EVERY reply. Say "that was all", get "Bra. Hörs." back, and the mic still
opened for eight seconds, with a chime, listening to an empty room. The device
has accepted {"type":"request_follow_up"} all along -- va_client.cpp's own
comment names the tool that was supposed to send it, and never existed here.
"""
import asyncio
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    LLMFullResponseEndFrame,
    UserStartedSpeakingFrame,
)

from app.follow_up_tool import (
    create_follow_up_tool_handler,
    get_follow_up_tool_definition,
)
from app.phase_emitter import PhaseEmitter


async def _noop(_value):
    return None


class _Params:
    def __init__(self):
        self.results = []
        self.arguments = {}

    async def result_callback(self, value):
        self.results.append(value)


def _emitter_with_sender():
    sent = []
    pe = PhaseEmitter(_noop, idle_debounce_s=0)

    async def sender():
        sent.append({"type": "request_follow_up"})

    pe.set_follow_up_sender(sender)
    return pe, sent


def test_the_tool_takes_no_arguments_and_says_when_not_to_call_it():
    definition = get_follow_up_tool_definition()
    assert definition["name"] == "request_follow_up"
    assert definition["parameters"] == {"type": "object", "properties": {}}
    # The whole point is that it is NOT called after a finished answer.
    assert "Do NOT call it" in definition["description"]


@pytest.mark.asyncio
async def test_calling_the_tool_does_not_open_the_mic_yet():
    """The model calls this BEFORE it speaks. The device opens the mic as soon
    as its speaker has drained -- which at tool-call time it usually has -- so
    sending immediately would open the mic before the question was asked."""
    pe, sent = _emitter_with_sender()
    params = _Params()

    await create_follow_up_tool_handler(pe.note_follow_up_requested)(params)

    assert params.results  # the model got an answer
    assert sent == []      # but the device has been told nothing


@pytest.mark.asyncio
async def test_the_request_goes_out_when_the_engine_finishes_the_turn():
    pe, sent = _emitter_with_sender()
    await create_follow_up_tool_handler(pe.note_follow_up_requested)(_Params())

    await pe._flush_follow_up()

    assert sent == [{"type": "request_follow_up"}]


@pytest.mark.asyncio
async def test_a_reply_that_asked_for_nothing_leaves_the_mic_shut():
    """The case the whole change exists for: a finished answer, a goodbye."""
    pe, sent = _emitter_with_sender()

    await pe._flush_follow_up()

    assert sent == []


@pytest.mark.asyncio
async def test_the_request_is_sent_once_not_once_per_end_frame():
    pe, sent = _emitter_with_sender()
    await create_follow_up_tool_handler(pe.note_follow_up_requested)(_Params())

    await pe._flush_follow_up()
    await pe._flush_follow_up()

    assert sent == [{"type": "request_follow_up"}]


@pytest.mark.asyncio
async def test_a_new_user_turn_drops_a_request_that_never_went_out():
    """The user answered without waiting, or interrupted. A request left over
    from the abandoned turn must not open the mic after the NEXT reply."""
    pe, sent = _emitter_with_sender()
    await create_follow_up_tool_handler(pe.note_follow_up_requested)(_Params())

    pe.push_frame = _noop_two
    await pe.process_frame(UserStartedSpeakingFrame(), _DOWN)
    await pe._flush_follow_up()

    assert sent == []


@pytest.mark.asyncio
async def test_a_failing_send_never_takes_the_turn_down():
    """Losing the window costs the user a wake word. Losing the turn costs
    them the answer."""
    pe = PhaseEmitter(_noop, idle_debounce_s=0)

    async def broken_sender():
        raise RuntimeError("websocket gone")

    pe.set_follow_up_sender(broken_sender)
    pe.note_follow_up_requested()

    await pe._flush_follow_up()  # must not raise


async def _noop_two(_frame, _direction):
    return None


from pipecat.processors.frame_processor import FrameDirection  # noqa: E402

_DOWN = FrameDirection.DOWNSTREAM
