import sys
import types

import pytest
from fastapi.testclient import TestClient

# undetected_chromedriver may be unavailable in CI; provide a safe stub.
if "undetected_chromedriver" not in sys.modules:
    sys.modules["undetected_chromedriver"] = types.SimpleNamespace(
        Chrome=lambda *args, **kwargs: None
    )

from app import app  # noqa: E402
from core import debug_mode  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_debug_state(monkeypatch):
    """Isolate each test: reset override, buffer and env var."""
    monkeypatch.delenv("SELENIUM_DEBUG", raising=False)
    debug_mode.reset_override()
    debug_mode.clear_events()
    yield
    debug_mode.reset_override()
    debug_mode.clear_events()


# ---------------------------------------------------------------------------
# Unit tests for the debug_mode module
# ---------------------------------------------------------------------------


def test_disabled_by_default():
    assert debug_mode.debug_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "On"])
def test_env_enables_debug(monkeypatch, value):
    monkeypatch.setenv("SELENIUM_DEBUG", value)
    debug_mode.reset_override()
    assert debug_mode.debug_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
def test_env_keeps_debug_disabled(monkeypatch, value):
    monkeypatch.setenv("SELENIUM_DEBUG", value)
    debug_mode.reset_override()
    assert debug_mode.debug_enabled() is False


def test_runtime_override_beats_env(monkeypatch):
    monkeypatch.setenv("SELENIUM_DEBUG", "true")
    debug_mode.reset_override()
    assert debug_mode.debug_enabled() is True
    debug_mode.set_debug(False)
    assert debug_mode.debug_enabled() is False
    debug_mode.reset_override()
    assert debug_mode.debug_enabled() is True


def test_record_event_noop_when_disabled():
    debug_mode.record_event("input", "my-engine", prompt="hi")
    assert debug_mode.get_events() == []


def test_record_and_read_events():
    debug_mode.set_debug(True)
    debug_mode.record_event("input", "my-engine", prompt="hello", model="default")
    debug_mode.record_event("chunk", "my-engine", index=1, total=2, size=5)
    debug_mode.record_event("output", "my-engine", reply="hi there", elapsed_ms=42)

    events = debug_mode.get_events()
    assert len(events) == 3
    kinds = [e["kind"] for e in events]
    assert kinds == ["input", "chunk", "output"]
    assert events[0]["engine"] == "my-engine"
    assert events[0]["prompt"] == "hello"
    assert events[2]["elapsed_ms"] == 42
    # Sequence numbers are strictly increasing.
    assert events[0]["seq"] < events[1]["seq"] < events[2]["seq"]


def test_get_events_since_filter():
    debug_mode.set_debug(True)
    debug_mode.record_event("input", "my-engine", prompt="a")
    debug_mode.record_event("input", "my-engine", prompt="b")
    first_seq = debug_mode.get_events()[0]["seq"]
    later = debug_mode.get_events(since=first_seq)
    assert len(later) == 1
    assert later[0]["prompt"] == "b"


def test_clear_events():
    debug_mode.set_debug(True)
    debug_mode.record_event("input", "my-engine", prompt="a")
    assert debug_mode.get_events()
    debug_mode.clear_events()
    assert debug_mode.get_events() == []


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


def test_api_get_debug_default():
    res = client.get("/api/debug")
    assert res.status_code == 200
    body = res.json()
    assert body["enabled"] is False
    assert body["env_default"] is False


def test_api_toggle_debug():
    res = client.post("/api/debug", json={"enabled": True})
    assert res.status_code == 200
    assert res.json()["enabled"] is True
    assert debug_mode.debug_enabled() is True

    res = client.post("/api/debug", json={"enabled": False})
    assert res.status_code == 200
    assert res.json()["enabled"] is False


def test_api_toggle_debug_invalid_payload():
    res = client.post("/api/debug", json={"enabled": "yes"})
    assert res.status_code == 400


def test_api_debug_log_incremental():
    client.post("/api/debug", json={"enabled": True})
    debug_mode.record_event("output", "my-engine", reply="ok", elapsed_ms=10)
    res = client.get("/api/debug/log?since=0")
    assert res.status_code == 200
    body = res.json()
    assert body["enabled"] is True
    assert len(body["events"]) == 1
    assert body["events"][0]["reply"] == "ok"


def test_api_debug_clear():
    client.post("/api/debug", json={"enabled": True})
    debug_mode.record_event("input", "my-engine", prompt="x")
    assert debug_mode.get_events()
    res = client.post("/api/debug/clear")
    assert res.status_code == 200
    assert debug_mode.get_events() == []
