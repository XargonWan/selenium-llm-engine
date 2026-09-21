# selenium-llm-engine MCP server

A thin [MCP](https://modelcontextprotocol.io) server that exposes the
already-running selenium-llm-engine's REST API (see `DEVELOPERS.md`) as
tools, so an agent (or Claude Code itself) can check login state, trigger a
login flow, read debug page HTML / app logs, and reset or kill a stuck
browser session — without shelling out to `curl`.

It does not start, stop, or otherwise manage the container. The engine must
already be running (`docker compose up`) and reachable at
`SELENIUM_LLM_ENGINE_URL` (default `http://localhost:14848`).

## Setup

```sh
pip install -r mcp_server/requirements.txt   # installs into whatever interpreter runs this
```

The project's existing `.venv` already has `httpx`, so installing into it
only adds the `mcp` package.

## Run standalone

```sh
python mcp_server/server.py
```

## Register with Claude Code

Add to `.mcp.json` at the repo root (already present):

```json
{
  "mcpServers": {
    "selenium-llm-engine": {
      "command": ".venv/bin/python",
      "args": ["mcp_server/server.py"],
      "env": {
        "SELENIUM_LLM_ENGINE_URL": "http://localhost:14848"
      }
    }
  }
}
```

## Tools

| Tool | Wraps | Notes |
|---|---|---|
| `list_engines` | `GET /api/engines` | No browser started. |
| `get_login_state` | `GET /login/{engine}/state` | |
| `start_login` | `POST /login/{engine}` | Navigates only; finish login via the container's noVNC UI. |
| `get_default_engine` / `set_default_engine` | `GET`/`POST /api/engines/default` | |
| `get_page_html` | `GET /api/debug/page-html` | For debugging selectors against the live DOM. |
| `get_app_logs` | `GET /api/logs/app` | Incremental polling via `since`. |
| `get_selector_hints` | `GET /api/engines/selector-hints` | Runtime-discovered selectors per engine. |
| `reset_engines` | `POST /api/reset` | Graceful — prefer this over `kill_session`. |
| `kill_session` | `POST /api/session/kill` | SIGKILL; can lose unflushed session/cookie state. Last resort. |
| `ping` | `GET /api/ping` | Health check. |
