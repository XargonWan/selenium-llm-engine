"""Debug mode — engine-agnostic runtime tracing for the LLM engine.

Debug mode is **disabled by default**.  It can be enabled in two ways:

* At startup via the ``SELENIUM_DEBUG`` environment variable
  (``1``/``true``/``yes``/``on`` — case-insensitive).
* At runtime via the Web UI / API (``POST /api/debug``), which overrides the
  environment default until the process restarts.

When enabled, every prompt lifecycle is traced into an in-memory ring buffer:

* the input prompt,
* the chunks generated when an oversized prompt is split,
* the output reply,
* which engine served the request,
* the total time from input to output.

The buffer is capped so it can never grow without bound.  All operations are
thread-safe because chunk events are recorded from Selenium worker threads
while HTTP handlers read the buffer from the event loop thread.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

# Ring buffer of debug events (survives for the process lifetime).
_MAX_EVENTS = 500
_EVENTS: Deque[Dict[str, Any]] = deque(maxlen=_MAX_EVENTS)
_SEQ = 0
_LOCK = threading.Lock()

# Runtime override: None means "follow the environment variable", while True or
# False is an explicit toggle set through the API/UI.
_runtime_override: Optional[bool] = None


def _env_default() -> bool:
    raw = os.getenv("SELENIUM_DEBUG", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def debug_enabled() -> bool:
    """Return True when debug tracing is active."""
    if _runtime_override is not None:
        return _runtime_override
    return _env_default()


def set_debug(enabled: bool) -> bool:
    """Enable or disable debug tracing at runtime.

    Returns the new effective state.
    """
    global _runtime_override
    _runtime_override = bool(enabled)
    return _runtime_override


def reset_override() -> None:
    """Drop the runtime override so the environment default applies again."""
    global _runtime_override
    _runtime_override = None


def status() -> Dict[str, Any]:
    """Return the current debug state for the UI/API."""
    return {
        "enabled": debug_enabled(),
        "env_default": _env_default(),
        "override": _runtime_override,
        "buffered_events": len(_EVENTS),
    }


def record_event(kind: str, engine: str, **fields: Any) -> None:
    """Append a debug event to the ring buffer.

    No-op when debug mode is disabled, so callers can invoke it
    unconditionally without paying any cost in the common (disabled) case.

    Parameters
    ----------
    kind:
        Event category (e.g. ``"input"``, ``"chunk"``, ``"output"``,
        ``"error"``).
    engine:
        The engine name the event refers to.
    **fields:
        Arbitrary extra data (prompt text, chunk index, elapsed_ms, ...).
    """
    if not debug_enabled():
        return
    global _SEQ
    with _LOCK:
        _SEQ += 1
        event: Dict[str, Any] = {
            "seq": _SEQ,
            "ts": time.strftime("%H:%M:%S", time.localtime()),
            "kind": kind,
            "engine": engine,
        }
        event.update(fields)
        _EVENTS.append(event)


def get_events(since: int = 0) -> List[Dict[str, Any]]:
    """Return buffered events with ``seq`` greater than *since*."""
    with _LOCK:
        return [e for e in _EVENTS if e["seq"] > since]


def clear_events() -> None:
    """Empty the debug event buffer."""
    with _LOCK:
        _EVENTS.clear()
