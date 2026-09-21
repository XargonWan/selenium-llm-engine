# CLAUDE.md

Orientation for an agent working in this repo. Contribution rules (what to
check, when to add tests, engine-agnosticism) live in `AGENTS.md` — read that
too. This file covers how the system actually behaves at runtime, which is
easy to get wrong from reading the code in isolation.

## What this is

A FastAPI service that drives real LLM web UIs (Gemini, ChatGPT, Claude,
Copilot, Grok, Perplexity, StepFun) through a single shared, persistent
Chrome instance controlled by [zendriver](https://github.com/stephanlensky/zendriver)
(a native-asyncio Chrome DevTools Protocol driver — no separate chromedriver
binary), exposing an OpenAI-compatible `/v1/chat/completions` API on top.
`core/` is engine-agnostic infrastructure; every engine-specific detail
(selectors, URLs, login detection) lives in `engines/*.json`, loaded by
`core/json_engine.py`. See `DEVELOPERS.md` for the full schema when adding or
fixing an engine, and `README.md` for the user-facing API surface.

Previously this ran on Selenium + `undetected_chromedriver` (uc). It was
migrated to zendriver because uc re-patches the chromedriver binary on every
launch to hide automation, and that per-launch patching was the leading
hypothesis for why a killed-and-relaunched browser lost its Gemini login even
with a valid cookie/profile restored — a stock, unpatched Chrome never does
that. zendriver talks to a real, unmodified Chromium directly over CDP, so
there is no binary to re-patch. See the git history around the
`refactor/zendriver` branch for the full investigation. zendriver is
AGPL-3.0-licensed (uc was MIT) — this project is already open source, so this
is a note, not a blocker.

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

One pre-existing test failure, `test_upload_via_file_input_rejects_missing_value`,
is not a real-browser-availability issue (it is fully mock-based) — it
reflects a genuine mismatch between the test's expectation and
`_upload_via_file_input`'s own "trust a successful send even if the
post-send poll never confirms it" fallback, which predates this file's
zendriver rewrite. Left as-is; see the docstring on that test.

## Local Docker workflow

This is a local/test deployment — build, up, logs, restart, stop, whatever's
needed. `docker-compose.yml` ships with the `build: .` line **commented out**
and the registry `image:` line active, because the repo pulls from the
published image by default. If you uncomment `build: .` to iterate locally,
**never commit that change** — it's local-only; keep the registry image line
as the committed default.

## Browser session & login persistence — read this before touching timeouts

All engine instances share **one** Chrome process wrapped by a single
zendriver `Browser` (`_shared_browser` in `core/zendriver_llm_base.py`) bound
to a single on-disk profile directory (`CHROMIUM_PROFILE_DIR`, default
`/config/.config/chromium-synth`, normally a mounted volume). Persistence
relies **entirely** on that native profile — exactly like a normal desktop
browser — and nothing else. `_save_cookies`/`_restore_cookies`/
`_maybe_save_cookies` are intentionally dead no-ops; do not re-enable them
without reading the paragraph below first.

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
rather than round-tripping the whole cookie jar blindly. zendriver exposes
`browser.cookies.get_all()/set_all()/save()/load()`, which would make
rebuilding this trivial — do NOT use them for this purpose; that's exactly
what was tried and reverted.

This history is *why* the zendriver migration happened at all: even with
native-profile-only persistence and a graceful shutdown (below), a from-
scratch test still lost the Gemini session on a real kill+relaunch under
Selenium/`undetected_chromedriver` — behavior a stock desktop Chrome does
not exhibit. The leading hypothesis was `undetected_chromedriver` re-patching
its chromedriver binary on every launch, introducing fingerprint drift Google
treats as suspicious. zendriver has no such binary (it drives Chrome directly
over CDP), which directly removes that mechanism. This is a hypothesis test,
not a confirmed fix — watch actual session longevity after this change ships.

**The one rule that matters**: Chrome must be allowed to shut down
gracefully — via `_reset_driver()` (which calls the module-level
`_stop_browser_with_timeout()`, itself `browser.stop()` with a bounded
timeout) — before its process dies. `_force_reset_driver()` /
`force_kill_session()` send SIGKILL straight to the known Chrome process with
no graceful `stop()` call at all, which can lose whatever session/auth cookie
Chrome had just written but not yet flushed. Reserve those nuclear paths for
when the coroutine driving the browser is itself wedged (e.g. the outer
`asyncio.TimeoutError` handler in `generate_response`) — anywhere the browser
is still responsive, use `_reset_driver()` instead, even for routine
recycling. (This is exactly the fix carried over into the session-hard-
timeout watchdog in `_generate_response_once` — see its comment for the full
reasoning.)

Because zendriver owns the exact Chrome subprocess handle
(`Browser._process`), shutdown/kill no longer has to pattern-match process
names across the whole system the way the old `pkill -f chromium` dance did
— `shutdown_shared_driver()`/`force_kill_session()` in
`core/zendriver_llm_base.py` target that one known PID directly. A narrow
profile-dir-pattern pkill is still used, but only once, before a browser
exists, to clean up a stale lock left by a crashed previous run.

`SELENIUM_SESSION_HARD_TIMEOUT` (env, seconds, default 21600 = 6h) bounds how
long a single driver instance may live before being recycled, to stop a
renderer from running for days. Don't drop this default back down without
re-checking the graceful-shutdown rule above — a short interval combined with
a nuclear reset is what silently logged users out roughly every 5 minutes
before this was fixed.

## Cross-engine browser access — one shared Tab, many engines

All engines share one `Browser`/`Tab`, but `EngineManager` gives each engine
its *own* FIFO queue and worker task — that only serialises requests to the
*same* engine. Nothing previously stopped two different engines' workers
from driving the one shared tab at the same time. Observed live: switching
from a still-running Copilot request to ChatGPT mid-flight let both
coroutines type/click/navigate the same tab concurrently, corrupting both
requests and eventually forcing a nuclear browser reset (the ChatGPT
request's own `asyncio.TimeoutError` handler). Fixed with
`_BROWSER_ACCESS_LOCK` (`core/zendriver_llm_base.py`, module-level
`asyncio.Lock`), acquired by the small number of true external entry points
— `generate_response`, `start_login_flow`, `check_login_state` — and
nowhere else (it is **not reentrant**: an internal helper one of those calls
must never also acquire it, or it deadlocks). Acquired *before* the
timeout-guarded work starts, not inside it, so waiting for another engine's
turn never eats into this request's own `total_timeout` and never lets this
request's timeout handler force-reset a browser another engine is still
legitimately using.

## Orphan cleanup vs. the active browser

`EngineManager._cleanup_orphans()` (a periodic sweep, default every 300s)
kills any `chromium`/`chrome`-matching process older than
`SELENIUM_ORPHAN_AGE_THRESHOLD` (default 600s) — it used to have **no
concept of "in use"** at all, so once the shared browser had been alive for
over 10 minutes it looked identical to a genuine orphan and got SIGKILLed
out from under whatever was using it, every single sweep. This was masked
for a long time: the session-hard-timeout watchdog used to recycle the
browser every ~300s by default, always safely under the 600s orphan
threshold. Once that default was raised to hours (see above — a long-lived
shared browser is the *intended* normal state now), the orphan sweep started
killing the live browser every 5 minutes, surfacing as an immediate
CDP-transport `"no close frame received or sent"` error on whatever request
happened to be using it. Fixed with `EngineManager._protected_pids()`
(`core/engine_manager.py`), which walks the active browser's whole process
tree (via `core.zendriver_llm_base.get_shared_browser_pid()` plus recursive
`pgrep -P`) and excludes it from the sweep — protecting only the root PID
isn't enough, since a long-open tab's renderer process has its own,
independently old start time.

Relatedly, `no close frame received or sent` (zendriver's CDP transport,
via the `websockets` library, when the browser process died and the TCP
connection dropped without a proper WS close handshake) is now in
`_is_dead_session`'s marker list — it wasn't, so every retry kept hitting
the same dead connection and failing in ~1ms instead of the browser ever
getting recreated.

## Resetting a stuck session

Three different levers, in order of how much they cost:

- `POST /api/session/reset` (web UI: "Reset Session") — cancels in-flight
  requests, drains every engine's queue, then gracefully restarts the shared
  browser (`EngineManager.stop_all()` → graceful `browser.stop()`, login
  preserved via the native profile). Does **not** touch stats or prompt
  history. This is the right button for "something's stuck" (e.g. the
  cross-engine interference above) without losing login or history.
- `POST /api/reset` / `POST /reset` (web UI: "Clear engine state") — the
  same graceful reset, but also wipes stats and prompt history. Use when you
  actually want a clean slate, not just to unblock.
- `POST /api/session/kill` (web UI: "Kill Session") — SIGKILL, no graceful
  `stop()` at all. Only for a browser that's completely frozen and
  unresponsive to the graceful paths above; can lose unflushed session/login
  state.

## Known limitation: anti-bot / Cloudflare inside the container

Observed: the Cloudflare challenge on ChatGPT fails even when completed
manually by a human through the container's noVNC desktop. Suspected cause —
**not yet confirmed or fixed**: the container has no GPU, so Chromium falls
back to software rendering (e.g. SwiftShader/llvmpipe) for WebGL, which is a
strong bot-detection signal on its own, independent of any automation flag.
Chromium itself is pinned to 148 — that pin was originally load-bearing for
`undetected_chromedriver` (uc 3.5.5 doesn't support Chromium ≥149) but
zendriver has no such coupling (it speaks plain CDP to whatever Chromium
version is installed, with no driver-binary version negotiation), so the pin
is no longer functionally required. It was deliberately left in place by the
zendriver migration anyway (bumping it is an independent decision the user
hasn't signed off on) — do not bump it without checking with the user first.
Worth investigating if/when ChatGPT support is prioritized: GPU passthrough
(`--device=/dev/dri`), or at least not forcing `--disable-gpu` in
`_build_config()` (`core/zendriver_llm_base.py`). zendriver also ships a
built-in Cloudflare Turnstile solver (`zendriver.core.cloudflare`) and
benchmarks well on Turnstile bypass rates generally — worth evaluating later,
but the engine currently only *detects* a captcha and asks the human to solve
it via noVNC (`_is_captcha_present` → a user-facing message); that detect-only
behavior was deliberately left unchanged by the migration. Out of scope while
work is focused on Gemini.

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
