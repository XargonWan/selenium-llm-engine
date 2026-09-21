"""MCP server exposing the selenium-llm-engine REST API as tools.

Thin wrapper: every tool below forwards to an already-running
selenium-llm-engine instance (see ``docker-compose.yml``) over HTTP. It adds
no logic of its own beyond request/response shaping, so it stays correct as
long as the underlying REST API (documented in DEVELOPERS.md) does.

Configure the target instance with SELENIUM_LLM_ENGINE_URL (default:
http://localhost:14848).

Run standalone:  python mcp_server/server.py
Register in Claude Code:  claude mcp add selenium-llm-engine -- python mcp_server/server.py
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer

BASE_URL = os.environ.get("SELENIUM_LLM_ENGINE_URL", "http://localhost:14848").rstrip("/")
HTTP_TIMEOUT = float(os.environ.get("SELENIUM_LLM_ENGINE_MCP_TIMEOUT", "30"))

mcp = MCPServer(
    "selenium-llm-engine",
    instructions=(
        "Tools to inspect and manage a running selenium-llm-engine instance: "
        "check which browser-driven LLM engines are logged in, trigger a login "
        "flow, read the current page HTML/app logs for debugging, and reset or "
        "kill a stuck browser session. Does not start or stop the container "
        "itself -- the engine must already be running at SELENIUM_LLM_ENGINE_URL."
    ),
)


async def _request(method: str, path: str, **kwargs: Any) -> Any:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=HTTP_TIMEOUT) as client:
        resp = await client.request(method, path, **kwargs)
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        if resp.is_error:
            return {"error": True, "status_code": resp.status_code, "body": body}
        return body


@mcp.tool()
async def list_engines() -> Any:
    """List every discovered engine (name, aliases, capabilities) without starting a browser."""
    return await _request("GET", "/api/engines")


@mcp.tool()
async def get_login_state(engine: str) -> Any:
    """Check whether *engine* (e.g. 'gemini') currently has a logged-in browser session."""
    return await _request("GET", f"/login/{engine}/state")


@mcp.tool()
async def start_login(engine: str) -> Any:
    """Open *engine*'s service URL in its browser session so a human can log in via the noVNC UI.

    Does not perform the login itself -- it only navigates there and reports
    the resulting login state. Watch the container's noVNC port (see
    docker-compose.yml) to complete the login manually.
    """
    return await _request("POST", f"/login/{engine}")


@mcp.tool()
async def get_default_engine() -> Any:
    """Return the engine currently used by default when no engine is specified."""
    return await _request("GET", "/api/engines/default")


@mcp.tool()
async def set_default_engine(engine: str) -> Any:
    """Set the default engine by name or alias."""
    return await _request("POST", "/api/engines/default", json={"engine": engine})


@mcp.tool()
async def get_page_html(engine: str | None = None) -> Any:
    """Return the current page HTML from an engine's live browser session, for debugging selectors.

    Uses the active engine when *engine* is omitted. Returns an error if no
    browser session is running yet for that engine.
    """
    params = {"engine_name": engine} if engine else {}
    return await _request("GET", "/api/debug/page-html", params=params)


@mcp.tool()
async def get_app_logs(since: int = 0) -> Any:
    """Return application log entries with sequence number greater than *since* (incremental polling)."""
    return await _request("GET", "/api/logs/app", params={"since": since})


@mcp.tool()
async def get_selector_hints() -> Any:
    """Return the best-matching selectors runtime-discovered for each active engine instance.

    Useful when a site changes its DOM and the configured selectors in
    engines/*.json need updating.
    """
    return await _request("GET", "/api/engines/selector-hints")


@mcp.tool()
async def reset_engines() -> Any:
    """Soft-reset: cancel in-flight jobs, drain queues and gracefully quit all browser sessions.

    Engine instances are cleared and will be recreated (with a fresh browser)
    on the next request. Prefer this over kill_session when the browser is
    still responsive, since it lets Chrome flush cookies/session state to
    disk before exiting.
    """
    return await _request("POST", "/api/reset")


@mcp.tool()
async def kill_session() -> Any:
    """Hard-kill: SIGKILL the shared browser process immediately.

    Only use this when the browser is completely frozen and reset_engines
    doesn't help -- an unclean kill can lose any session/cookie state Chrome
    had not yet flushed to disk, forcing a fresh login on some engines.
    """
    return await _request("POST", "/api/session/kill")


@mcp.tool()
async def ping() -> Any:
    """Health check for the selenium-llm-engine instance at SELENIUM_LLM_ENGINE_URL."""
    return await _request("GET", "/api/ping")


if __name__ == "__main__":
    mcp.run()
