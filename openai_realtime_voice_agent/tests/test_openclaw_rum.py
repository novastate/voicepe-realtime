"""A question asked in the kitchen must carry the kitchen as its room, so the late answer is announced there."""
import json
from types import SimpleNamespace

import httpx
import pytest

from app import openclaw_tool
from app.device_registry import rum_ur_enhet


class Llm:
    def __init__(self):
        self.handlers = {}

    def register_function(self, name, fn):
        self.handlers[name] = fn


async def _fraga(monkeypatch, device_id):
    monkeypatch.setenv("OPENCLAW_URL", "http://adapter:3400")
    monkeypatch.setenv("INSTANCE_NAME", "kontor")  # the process-wide value that used to win
    sedda = []

    def handler(request):
        sedda.append(request.read().decode())
        return httpx.Response(200, json={"answer": "ok"})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)}))
    llm = Llm()
    openclaw_tool.register_openclaw_tool(llm, device_id)

    async def cb(r):
        pass

    await llm.handlers["ask_openclaw"](SimpleNamespace(arguments={"question": "hur gick det?"}, result_callback=cb))
    return sedda[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("device,rum", [("koket", "koket"), ("kontoret", "kontoret"),
                                        ("10.0.3.9", "kontor"), ("unknown", "kontor"), ("", "kontor")])
async def test_sena_svar_far_enhetens_rum(monkeypatch, device, rum):
    assert json.loads(await _fraga(monkeypatch, device))["room"] == rum


def test_rum_ur_enhet():
    assert rum_ur_enhet("koket") == "koket" and rum_ur_enhet("fe80::1") == "" and rum_ur_enhet(None) == ""
