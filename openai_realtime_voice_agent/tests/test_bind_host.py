import logging
import pytest

@pytest.mark.parametrize("name", ["WEBSOCKET_HOST", "ANNOUNCE_HOST"])
def test_default_loopback(monkeypatch, name):
    from app.bind_host import resolve_bind_host
    monkeypatch.delenv(name, raising=False)
    assert resolve_bind_host(name) == "127.0.0.1"


def test_warning_once_per_variable(monkeypatch, caplog):
    from app import bind_host
    monkeypatch.setattr(bind_host, "_warned_variables", set())
    with caplog.at_level(logging.WARNING):
        for name in ("WEBSOCKET_HOST", "ANNOUNCE_HOST"):
            monkeypatch.setenv(name, "0.0.0.0")
            for _ in range(3):
                assert bind_host.resolve_bind_host(name) == "0.0.0.0"
    assert len(caplog.records) == 2
    for name, record in zip(("WEBSOCKET_HOST", "ANNOUNCE_HOST"), caplog.records):
        assert name in record.message
        assert "Comms is no longer the only door" in record.message

@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "::ffff:127.0.0.1"])
def test_loopback_quiet(monkeypatch, caplog, host):
    from app.bind_host import resolve_bind_host
    monkeypatch.setenv("WEBSOCKET_HOST", host)
    with caplog.at_level(logging.WARNING):
        assert resolve_bind_host("WEBSOCKET_HOST") == host
    assert not caplog.records
