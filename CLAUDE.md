# CLAUDE.md

Orientation for an agent working in this repo. Contribution rules (what to
check, when to add tests, engine-agnosticism) live in `AGENTS.md` — read that
too. This file covers how the system actually behaves at runtime, which is
easy to get wrong from reading the code in isolation.

## What this is

A FastAPI service that drives real LLM web UIs (Gemini, ChatGPT, Claude,
Copilot, Grok, Perplexity, StepFun) through a single shared, persistent
Chrome instance controlled by Selenium/`undetected_chromedriver`, exposing an
OpenAI-compatible `/v1/chat/completions` API on top. `core/` is engine-agnostic
infrastructure; every engine-specific detail (selectors, URLs, login
detection) lives in `engines/*.json`, loaded by `core/json_engine.py`. See
`DEVELOPERS.md` for the full schema when adding or fixing an engine, and
`README.md` for the user-facing API surface.

## Local dev loop

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/ruff check .
.venv/bin/python -m py_compile core/*.py app.py
.venv/bin/python -m pytest -q
```

These three are the mandatory checks from `AGENTS.md` §5 — run them after
every change, before calling anything done. `ruff` may need a separate
install (`pip install ruff`) if it's not already on the venv.

Two pre-existing test failures (`test_upload_via_file_input_rejects_missing_value`,
`test_sync_generate_response_once_retries_on_stale_element`) only occur in
environments without a real Chromium/chromedriver binary (e.g. a bare dev
sandbox) — they are not a signal of a regression there.

## Local Docker workflow

This is a local/test deployment — build, up, logs, restart, stop, whatever's
needed. `docker-compose.yml` ships with the `build: .` line **commented out**
and the registry `image:` line active, because the repo pulls from the
published image by default. If you uncomment `build: .` to iterate locally,
**never commit that change** — it's local-only; keep the registry image line
as the committed default.

## Browser session & login persistence — read this before touching timeouts

All engine instances share **one** Chrome process (`_shared_driver` in
`core/selenium_llm_base.py`) bound to a single on-disk profile directory
(`CHROMIUM_PROFILE_DIR`, default `/config/.config/chromium-synth`, normally a
mounted volume). Persistence relies **entirely** on that native profile —
exactly like a normal desktop browser — and nothing else.
`_save_cookies`/`_restore_cookies`/`_maybe_save_cookies` are intentionally
dead no-ops; do not re-enable them without reading the paragraph below first.

A prior version of this project added a second layer on top: a background
thread that snapshotted every cookie via CDP `Network.getAllCookies` every
60s and reapplied it (`Network.setCookie`) on the next driver creation, to
survive a crash between Chrome's own (batched, not synchronous) writes to
its cookie store. **It was removed after being observed live to cause the
exact problem it was meant to prevent**: replaying an old value of a
short-lived, rotating anti-replay cookie (Google uses one — e.g.
`__Secure-1PSIDTS` — specifically to detect session/cookie replay) looks
like session hijacking to Google's own security systems, which then forces
a fresh sign-in. A single, well-spaced test (90s+ after a manual login, one
clean kill) reproduced the logout — not just the earlier rapid-repeated-kill
case, which had first looked like the more likely cause. If a similar safety
net is ever reconsidered, it would need to exclude that class of cookie
rather than round-tripping the whole cookie jar blindly.

**The one rule that matters**: Chrome must be allowed to `quit()` normally —
via `_reset_driver()` — before its process dies. `_force_reset_driver()` /
`force_kill_session()` send SIGKILL straight to the process tree with no
`quit()` call at all, which can lose whatever session/auth cookie Chrome had
just written but not yet flushed. Reserve those nuclear paths for when the
driving thread is itself stuck (e.g. the `asyncio.TimeoutError` handler in
`generate_response`) — anywhere the driver is still responsive, use
`_reset_driver()` instead, even for routine recycling. (This is exactly the
bug fixed in the session-hard-timeout watchdog in
`_sync_generate_response_once` — see its comment for the full reasoning.)

`SELENIUM_SESSION_HARD_TIMEOUT` (env, seconds, default 21600 = 6h) bounds how
long a single driver instance may live before being recycled, to stop a
renderer from running for days. Don't drop this default back down without
re-checking the graceful-quit rule above — a short interval combined with a
nuclear reset is what silently logged users out roughly every 5 minutes
before this was fixed.

## Known limitation: anti-bot / Cloudflare inside the container

Observed: the Cloudflare challenge on ChatGPT fails even when completed
manually by a human through the container's noVNC desktop. Suspected cause —
**not yet confirmed or fixed**: the container has no GPU, so Chromium falls
back to software rendering (e.g. SwiftShader/llvmpipe) for WebGL, which is a
strong bot-detection signal on its own, independent of any Selenium
automation flag. Chromium itself is pinned to 148 specifically so
`undetected_chromedriver`'s stealth patching (`uc.Chrome`) stays available —
check `_prefer_webdriver_fallback` in `selenium_llm_base.py` isn't silently
in effect (it skips stealth patching entirely) before assuming that patching
is active. Worth investigating if/when ChatGPT support is prioritized: GPU
passthrough (`--device=/dev/dri`), or at least not forcing `--disable-gpu` in
`_build_options()`. Out of scope while work is focused on Gemini.

## MCP server (`mcp_server/`)

A thin MCP server wrapping this service's own REST API (login state, start
login, debug page HTML, app logs, graceful reset, hard kill) so an agent can
inspect/manage a running instance without shelling out to `curl`. See
`mcp_server/README.md` for setup and the full tool list. It requires the
engine to already be running and reachable — it does not manage the
container itself.

## Debugging a stuck or logged-out engine

- `GET /login/{engine}/state` — current login state.
- `GET /api/debug/page-html?engine_name={engine}` — live DOM of that engine's
  browser session (or use the MCP server's `get_page_html`).
- `GET /api/logs/app?since=N` — incremental app log tail (in-memory ring
  buffer) — also persisted to `/app/logs/selenium-llm-engine.log` inside the
  container when writable.
- The container's noVNC port (see `docker-compose.yml`) shows the actual
  browser live — useful to complete a login manually or see what a selector
  is missing.
