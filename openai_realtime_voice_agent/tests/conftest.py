"""Fail fast with a clear message when the test environment is wrong.

The suite needs the dependencies pinned in pyproject.toml (pipecat-ai 0.0.97
and sherpa-onnx). Without them xai/gemini/speaker_injection tests fail with
confusing import errors. See "Running the tests" in README.md.
"""
import importlib.metadata as md

import pytest

PIPECAT_PIN = "0.0.97"


def pytest_configure(config):
    problems = []
    try:
        v = md.version("pipecat-ai")
        if v != PIPECAT_PIN:
            problems.append(f"pipecat-ai is {v}, pyproject.toml pins {PIPECAT_PIN}")
    except md.PackageNotFoundError:
        problems.append("pipecat-ai is not installed")
    try:
        md.version("sherpa-onnx")
    except md.PackageNotFoundError:
        problems.append("sherpa-onnx is not installed")
    if problems:
        pytest.exit(
            "Wrong test environment: " + "; ".join(problems)
            + ". Create the venv as described in README.md, 'Running the tests'.",
            returncode=3,
        )
