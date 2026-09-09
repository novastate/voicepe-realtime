"""A false wake is countable even when no audio was kept.

Live 2026-09-09, twice in one conversation, on a default install:

    ❌ mark_false_wake failed: [Errno 2] No such file or directory:
       '/share/voice-probes'
    ⚠️ button false-wake flag failed: FileNotFoundError(2, ...)

Both call sites listed a directory that only exists while ENABLE_RECORDING is
on. So the tool the model is given could never succeed in the configuration
the house actually runs, and the counter it feeds never moved either.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.speaker_context as speaker_context
from app.speaker_context import mark_latest_probe_as_false_wake


def test_a_missing_directory_is_nothing_to_mark_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        speaker_context, "PROBE_DUMP_DIR", str(tmp_path / "never-created")
    )
    assert mark_latest_probe_as_false_wake() is None


def test_an_empty_directory_is_nothing_to_mark(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker_context, "PROBE_DUMP_DIR", str(tmp_path))
    assert mark_latest_probe_as_false_wake() is None


def test_the_newest_capture_is_the_one_renamed(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker_context, "PROBE_DUMP_DIR", str(tmp_path))
    for name in ("probe_20260909_120000.wav", "probe_20260909_130000.wav"):
        (tmp_path / name).write_bytes(b"")

    marked = mark_latest_probe_as_false_wake()

    assert marked == "falsewake_20260909_130000.wav"
    assert (tmp_path / marked).exists()
    # The older one is left alone -- only the wake being complained about is
    # relabelled.
    assert (tmp_path / "probe_20260909_120000.wav").exists()


def test_a_capture_already_marked_is_not_marked_again(tmp_path, monkeypatch):
    """Only probe_* files are candidates. Without the prefix filter, a second
    complaint would rename falsewake_ files into falsewake_falsewake_ and lose
    track of which wake was which."""
    monkeypatch.setattr(speaker_context, "PROBE_DUMP_DIR", str(tmp_path))
    (tmp_path / "falsewake_20260909_130000.wav").write_bytes(b"")

    assert mark_latest_probe_as_false_wake() is None
