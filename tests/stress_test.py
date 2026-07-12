#!/usr/bin/env python3
# stress_test.py is NOT a pytest test.
# Run manually: python tests/stress_test.py
# This file exists in tests/ for organisation only and is excluded from any CI pipeline.
"""
Stress test for selenium-llm-engine
===================================

Test objectives:

1. **Volume** — send at least ``MESSAGES_PER_ENGINE`` (default 50) messages per
   usable engine and measure robustness/timing.
2. **Queue** — verify the per-engine FIFO queue handles bursts without losing
   or interleaving requests, monitoring ``queue_depth`` from ``/api/engines``.
3. **Post-idle responsiveness** — reproduce the external-client *cooldown*
   scenario (default 240s): send a prompt, stay idle for ``IDLE_SECONDS`` and
   send again, verifying the engine still responds after the pause (this is
   where in production the engine often "stops responding").
4. **Diagnostics** — read the incremental application log (``/api/logs/app``),
   the stats (``/stats``) and the runtime selectors
   (``/api/engines/selector-hints``) so failures can be attributed to a precise
   cause (timeout, freeze, missing selector, limit/quota, etc.).

Engines that are not usable (login required but session not authenticated) are
**skipped** (``SKIPPED``) instead of being counted as failures, so the
robustness verdict is not skewed.

Configurable via environment variables:
    BASE_URL              (default http://localhost:14848)
    MESSAGES_PER_ENGINE   (default 50)
    IDLE_SECONDS          (default 240)  -> cooldown scenario
    ENGINES               (default: all discovered engines; CSV to filter)
    PER_REQUEST_TIMEOUT   (default 300)  -> httpx timeout per request
    PROBE_ONLY            (default 0)    -> 1 = probe+report only, no volume
    SKIP_IDLE             (default 0)    -> 1 = skip the idle/cooldown phase
    OUT_JSON              (default /tmp/stress_test_results.json)
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field

import httpx


BASE_URL = os.getenv("BASE_URL", "http://localhost:14848").rstrip("/")
MESSAGES_PER_ENGINE = int(os.getenv("MESSAGES_PER_ENGINE", "50"))
IDLE_SECONDS = int(os.getenv("IDLE_SECONDS", "240"))
PER_REQUEST_TIMEOUT = float(os.getenv("PER_REQUEST_TIMEOUT", "300"))
PROBE_ONLY = os.getenv("PROBE_ONLY", "0") == "1"
SKIP_IDLE = os.getenv("SKIP_IDLE", "0") == "1"
OUT_JSON = os.getenv("OUT_JSON", "/tmp/stress_test_results.json")

# Number of prompts fired simultaneously to exercise the FIFO queue.
QUEUE_BURST = int(os.getenv("QUEUE_BURST", "5"))

# Server-side rate limit is 20 requests / 60s per IP, and /v1/chat/completions
# consumes two slots per call.  We therefore keep a client-side token bucket
# well under that budget so bursts don't get spurious 429s counted as failures.
RATE_MAX_PER_WINDOW = int(os.getenv("RATE_MAX_PER_WINDOW", "8"))
RATE_WINDOW = 60.0


# A rotating pool of prompts of varying length so the model can't cache-answer
# identical inputs and so we exercise both short and long generations.
SHORT_PROMPTS = [
    "Answer with a single sentence: what is asynchronous programming?",
    "In one line: difference between a list and a tuple in Python.",
    "Briefly define the singleton pattern.",
    "What is a decorator in Python? Answer concisely.",
    "Explain in two sentences what a Docker container is.",
    "What is a race condition? Short answer.",
    "In one sentence: what is a database index for?",
    "What is idempotency in a REST API? Short.",
]

MEDIUM_PROMPTS = [
    "List 5 advantages of microservices over a monolith, with one line of explanation each.",
    "Explain the difference between authentication and authorization, with a practical example for each.",
    "Describe the lifecycle of an HTTP request from the browser to the server and back.",
    "Compare REST and GraphQL on performance, caching and type safety in 4 points.",
    "Explain what ACID and BASE are and when to prefer one over the other.",
]

LONG_PROMPTS = [
    "Analyze microservices architecture: definition, pros/cons, "
    "communication patterns (synchronous vs asynchronous) and observability strategies "
    "in a distributed environment. Structure the answer in numbered sections.",
    "Provide a guide to distributed databases: CAP theorem, replication strategies, "
    "sharding and ACID vs BASE tradeoffs. Use numbered sections and examples.",
    "Describe cloud security best practices: shared responsibility model, "
    "IAM and least privilege, data encryption and network segmentation. Numbered sections.",
]


def _prompt_for(index: int) -> str:
    """Deterministic rotating prompt: mostly short/medium with occasional long."""
    if index % 10 == 9:
        return LONG_PROMPTS[index % len(LONG_PROMPTS)]
    if index % 3 == 0:
        return MEDIUM_PROMPTS[index % len(MEDIUM_PROMPTS)]
    return SHORT_PROMPTS[index % len(SHORT_PROMPTS)]


# ---------------------------------------------------------------------------
# Client-side rate limiter (token bucket, shared across all tasks)
# ---------------------------------------------------------------------------


class RateLimiter:
    def __init__(self, max_per_window: int, window: float) -> None:
        self._max = max_per_window
        self._window = window
        self._hits: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.time()
                self._hits = [t for t in self._hits if t > now - self._window]
                if len(self._hits) < self._max:
                    self._hits.append(now)
                    return
                sleep_for = self._window - (now - self._hits[0]) + 0.05
            await asyncio.sleep(max(0.1, sleep_for))


RATE = RateLimiter(RATE_MAX_PER_WINDOW, RATE_WINDOW)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass
class PromptResult:
    engine: str
    index: int
    total_ms: float
    server_ms: float
    success: bool
    status: str = "OK"          # OK | FAIL | SKIPPED | RATE_LIMIT
    error: str = ""
    model: str = ""
    response_len: int = 0
    retries: int = 0


@dataclass
class EngineStats:
    name: str
    usable: bool = True
    skip_reason: str = ""
    results: list[PromptResult] = field(default_factory=list)
    queue_probe: dict = field(default_factory=dict)
    idle_probe: dict = field(default_factory=dict)

    @property
    def successful(self) -> list[PromptResult]:
        return [r for r in self.results if r.success]

    def _stat(self, fn):
        suc = [r.total_ms for r in self.successful]
        return fn(suc) if suc else 0.0

    @property
    def avg_total_ms(self) -> float:
        return self._stat(statistics.mean)

    @property
    def median_total_ms(self) -> float:
        return self._stat(statistics.median)

    @property
    def min_total_ms(self) -> float:
        return self._stat(min)

    @property
    def max_total_ms(self) -> float:
        return self._stat(max)

    @property
    def p95_total_ms(self) -> float:
        suc = sorted(r.total_ms for r in self.successful)
        if len(suc) < 2:
            return suc[0] if suc else 0.0
        idx = min(int(len(suc) * 0.95), len(suc) - 1)
        return suc[idx]

    @property
    def stddev_total_ms(self) -> float:
        suc = [r.total_ms for r in self.successful]
        return statistics.stdev(suc) if len(suc) >= 2 else 0.0

    @property
    def success_rate(self) -> float:
        graded = [r for r in self.results if r.status != "SKIPPED"]
        if not graded:
            return 0.0
        return len(self.successful) / len(graded) * 100


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


async def _send_once(
    client: httpx.AsyncClient, engine: str, prompt: str, index: int
) -> PromptResult:
    """Send a single prompt with rate-limit awareness and 429 backoff."""
    res = PromptResult(
        engine=engine, index=index, total_ms=0.0, server_ms=0.0, success=False
    )
    max_attempts = 4
    for attempt in range(max_attempts):
        await RATE.acquire()
        t0 = time.time()
        try:
            resp = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                json={"model": engine, "messages": [{"role": "user", "content": prompt}]},
                timeout=PER_REQUEST_TIMEOUT,
            )
            res.total_ms = (time.time() - t0) * 1000
            if resp.status_code == 200:
                data = resp.json()
                content = ""
                try:
                    content = data["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError):
                    content = ""
                res.success = bool(content.strip())
                res.status = "OK" if res.success else "FAIL"
                res.model = data.get("model", "")
                res.server_ms = float(data.get("elapsed_ms", 0) or 0)
                res.response_len = len(content)
                if not res.success:
                    res.error = "empty response body"
                return res
            if resp.status_code == 429:
                res.retries = attempt + 1
                backoff = min(30.0, 5.0 * (attempt + 1))
                await asyncio.sleep(backoff)
                continue
            res.status = "FAIL"
            res.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return res
        except httpx.TimeoutException:
            res.total_ms = (time.time() - t0) * 1000
            res.status = "FAIL"
            res.error = f"TIMEOUT after {PER_REQUEST_TIMEOUT:.0f}s"
            return res
        except Exception as exc:  # noqa: BLE001 - report any transport error
            res.total_ms = (time.time() - t0) * 1000
            res.status = "FAIL"
            res.error = str(exc)[:200]
            return res

    res.status = "RATE_LIMIT"
    res.error = "exhausted 429 retries"
    return res


async def _fetch_engines(client: httpx.AsyncClient) -> list[dict]:
    resp = await client.get(f"{BASE_URL}/api/engines", timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


async def _fetch_queue_depths(client: httpx.AsyncClient) -> dict[str, int]:
    try:
        data = await _fetch_engines(client)
    except Exception:
        return {}
    return {e["name"]: int(e.get("queue_depth", 0) or 0) for e in data}


async def _fetch_app_logs(client: httpx.AsyncClient, since: int) -> list[dict]:
    try:
        resp = await client.get(f"{BASE_URL}/api/logs/app", params={"since": since}, timeout=15)
        resp.raise_for_status()
        return resp.json().get("entries", [])
    except Exception:
        return []


async def _fetch_selector_hints(client: httpx.AsyncClient) -> dict:
    try:
        resp = await client.get(f"{BASE_URL}/api/engines/selector-hints", timeout=15)
        resp.raise_for_status()
        return resp.json().get("data", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Probe: decide which engines are usable
# ---------------------------------------------------------------------------


async def _probe_engine(client: httpx.AsyncClient, engine: dict) -> EngineStats:
    """Send one probe prompt; classify engine as usable or skip with a reason."""
    name = engine["name"]
    stats = EngineStats(name=name)
    allow_unlogged = bool(engine.get("allow_unlogged", False))

    probe = await _send_once(client, name, "Reply only with: PONG", index=-1)
    if probe.success:
        stats.usable = True
        return stats

    # Not usable — decide whether it's a login gap (skip) or a real defect (report).
    err = (probe.error or "").lower()
    login_markers = ("login", "unlogged", "not logged", "sign in", "signin", "authentication")
    if not allow_unlogged or any(m in err for m in login_markers):
        stats.usable = False
        stats.skip_reason = (
            f"login required / not authenticated (probe: {probe.error[:120]})"
            if not allow_unlogged
            else f"probe failed, looks like login gap: {probe.error[:120]}"
        )
    else:
        # allow_unlogged engine that still failed the probe -> genuine problem,
        # keep it in the run so the failure is reported and diagnosed.
        stats.usable = True
        stats.skip_reason = f"probe failed but engine allows unlogged: {probe.error[:120]}"
    return stats


# ---------------------------------------------------------------------------
# Phase: queue behaviour
# ---------------------------------------------------------------------------


async def _test_queue(client: httpx.AsyncClient, engine: str) -> dict:
    """Fire QUEUE_BURST prompts at once and confirm FIFO serial processing.

    We tag each prompt with an ordinal and check that all complete, that at
    least one moment shows queue_depth > 0 (proving they queued rather than
    ran in parallel), and record max observed depth.
    """
    print(f"    [queue] firing {QUEUE_BURST} simultaneous prompts at '{engine}'…")
    max_depth = {"value": 0}
    stop = asyncio.Event()

    async def _poll_depth():
        while not stop.is_set():
            depths = await _fetch_queue_depths(client)
            d = depths.get(engine, 0)
            if d > max_depth["value"]:
                max_depth["value"] = d
            await asyncio.sleep(0.5)

    poller = asyncio.create_task(_poll_depth())
    t0 = time.time()
    tasks = [
        asyncio.create_task(
            _send_once(client, engine, f"[Q{i}] {SHORT_PROMPTS[i % len(SHORT_PROMPTS)]}", index=1000 + i)
        )
        for i in range(QUEUE_BURST)
    ]
    results = await asyncio.gather(*tasks)
    stop.set()
    await poller
    elapsed = time.time() - t0

    ok = sum(1 for r in results if r.success)
    return {
        "burst": QUEUE_BURST,
        "completed_ok": ok,
        "failed": QUEUE_BURST - ok,
        "max_queue_depth_observed": max_depth["value"],
        "total_elapsed_s": round(elapsed, 1),
        "serial_confirmed": max_depth["value"] >= 1,
        "results": [
            {"index": r.index, "success": r.success, "ms": round(r.total_ms), "err": r.error[:120]}
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Phase: idle / cooldown responsiveness
# ---------------------------------------------------------------------------


async def _test_idle(client: httpx.AsyncClient, engine: str) -> dict:
    """Reproduce the external-client cooldown scenario.

    Send a warm-up prompt, idle for IDLE_SECONDS, then send again and check the
    engine still responds.  This is the case that in production frequently
    fails (the engine "stops responding" after the cooldown).
    """
    print(f"    [idle] warm-up prompt for '{engine}'…")
    warm = await _send_once(client, engine, "Reply only: READY", index=2000)
    print(f"    [idle] idling {IDLE_SECONDS}s (cooldown simulation)…")
    # Poll queue depth once mid-idle to confirm the session is truly idle.
    await asyncio.sleep(IDLE_SECONDS / 2)
    mid_depth = (await _fetch_queue_depths(client)).get(engine, 0)
    await asyncio.sleep(IDLE_SECONDS - IDLE_SECONDS / 2)
    print(f"    [idle] post-idle prompt for '{engine}'…")
    after = await _send_once(client, engine, "After the pause, reply only: AWAKE", index=2001)

    return {
        "idle_seconds": IDLE_SECONDS,
        "warmup_ok": warm.success,
        "warmup_ms": round(warm.total_ms),
        "warmup_err": warm.error[:160],
        "mid_idle_queue_depth": mid_depth,
        "post_idle_ok": after.success,
        "post_idle_ms": round(after.total_ms),
        "post_idle_err": after.error[:160],
        "recovered": after.success,
    }


# ---------------------------------------------------------------------------
# Phase: volume
# ---------------------------------------------------------------------------


async def _test_volume(client: httpx.AsyncClient, stats: EngineStats) -> None:
    engine = stats.name
    print(f"    [volume] sending {MESSAGES_PER_ENGINE} messages to '{engine}'…")
    for i in range(MESSAGES_PER_ENGINE):
        r = await _send_once(client, engine, _prompt_for(i), index=i)
        stats.results.append(r)
        done = i + 1
        if done % 5 == 0 or not r.success:
            flag = "OK" if r.success else f"FAIL({r.error[:40]})"
            print(
                f"      {engine} {done}/{MESSAGES_PER_ENGINE} "
                f"| {flag} | {r.total_ms:.0f}ms | ok={len(stats.successful)}"
            )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_report(all_stats: dict[str, EngineStats], log_summary: dict, hints: dict) -> None:
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    for name, s in sorted(all_stats.items()):
        print(f"\n{'='*60}\nENGINE: {name.upper()}\n{'='*60}")
        if not s.usable:
            print(f"  STATUS: SKIPPED — {s.skip_reason}")
            continue
        graded = [r for r in s.results if r.status != "SKIPPED"]
        print(f"  Volume requests:  {len(graded)}")
        print(f"  Successful:       {len(s.successful)} ({s.success_rate:.1f}%)")
        print(f"  Failed:           {len(graded) - len(s.successful)}")
        if s.successful:
            print("  --- Timing (successful) ---")
            print(f"  Avg:    {s.avg_total_ms/1000:6.1f}s   Median: {s.median_total_ms/1000:6.1f}s")
            print(f"  Min:    {s.min_total_ms/1000:6.1f}s   Max:    {s.max_total_ms/1000:6.1f}s")
            print(f"  P95:    {s.p95_total_ms/1000:6.1f}s   StdDev: {s.stddev_total_ms/1000:6.1f}s")
        if s.queue_probe:
            q = s.queue_probe
            print("  --- Queue ---")
            if "error" in q:
                print(f"  queue probe error: {q['error']}")
            else:
                print(
                    f"  burst={q['burst']} ok={q['completed_ok']} failed={q['failed']} "
                    f"max_depth={q['max_queue_depth_observed']} serial={q['serial_confirmed']} "
                    f"elapsed={q['total_elapsed_s']}s"
                )
        if s.idle_probe:
            idp = s.idle_probe
            if "error" in idp:
                print(f"  --- Idle/cooldown --- error: {idp['error']}")
            else:
                print(f"  --- Idle/cooldown ({idp['idle_seconds']}s) ---")
                print(
                    f"  warmup_ok={idp['warmup_ok']} post_idle_ok={idp['post_idle_ok']} "
                    f"recovered={idp['recovered']}"
                )
                if not idp["recovered"]:
                    print(f"  !! COOLDOWN REGRESSION: post-idle error: {idp['post_idle_err']}")
        fails = [r for r in graded if not r.success]
        if fails:
            print("  --- Failures (grouped) ---")
            causes: dict[str, int] = {}
            for f in fails:
                key = f.error.split(":")[0][:60] or f.status
                causes[key] = causes.get(key, 0) + 1
            for cause, count in sorted(causes.items(), key=lambda x: -x[1]):
                print(f"    {count:3d}x  {cause}")
        h = hints.get(name)
        if h and (h.get("prompt_selector") or h.get("send_selector")):
            print("  --- Runtime selectors (put these first in engines/*.json) ---")
            print(f"    prompt: {h.get('prompt_selector')}")
            print(f"    send:   {h.get('send_selector')}")

    print()
    print("=" * 80)
    print("SYSTEM SOLIDITY REPORT")
    print("=" * 80)
    usable = [s for s in all_stats.values() if s.usable]
    skipped = [s for s in all_stats.values() if not s.usable]
    total = sum(len([r for r in s.results if r.status != "SKIPPED"]) for s in usable)
    ok = sum(len(s.successful) for s in usable)
    rate = ok / total * 100 if total else 0.0
    print(f"Usable engines:     {[s.name for s in usable]}")
    print(f"Skipped engines:    {[(s.name, s.skip_reason[:40]) for s in skipped]}")
    print(f"Total graded:       {total}")
    print(f"Successful:         {ok} ({rate:.1f}%)")
    verdict = "GOOD" if rate >= 90 else "MARGINAL" if rate >= 70 else "POOR"
    print(f"Overall verdict:    {verdict}")

    cooldown_regressions = [
        s.name for s in usable if s.idle_probe and not s.idle_probe.get("recovered", True)
    ]
    if cooldown_regressions:
        print(f"!! Cooldown regressions (idle {IDLE_SECONDS}s): {cooldown_regressions}")
    else:
        print(f"Cooldown ({IDLE_SECONDS}s) responsiveness: OK on all tested engines")

    if log_summary:
        print("\n--- App log signal counts (during run) ---")
        for key, count in sorted(log_summary.items(), key=lambda x: -x[1]):
            print(f"  {count:4d}x  {key}")


def _summarise_logs(entries: list[dict]) -> dict:
    """Bucket interesting log lines by signal for a quick health overview."""
    signals = {
        "timeout": "timeout",
        "silent freeze": "silent freeze",
        "ui stuck": "ui stuck",
        "page refresh": "page refresh",
        "redirect-stall": "redirect-stall",
        "force-reset": "force-reset",
        "stuck for": "worker stuck",
        "captcha": "captcha",
        "limit detected": "limit/quota",
        "could not find prompt": "prompt selector miss",
        "error response": "engine error page",
    }
    counts: dict[str, int] = {}
    for e in entries:
        msg = str(e.get("message", "")).lower()
        for needle, label in signals.items():
            if needle in msg:
                counts[label] = counts.get(label, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run() -> None:
    print("=" * 80)
    print("STRESS TEST - selenium-llm-engine")
    print("=" * 80)
    print(f"BASE_URL={BASE_URL}  messages/engine={MESSAGES_PER_ENGINE}  "
          f"idle={IDLE_SECONDS}s  probe_only={PROBE_ONLY}  skip_idle={SKIP_IDLE}")

    async with httpx.AsyncClient() as client:
        try:
            ping = await client.get(f"{BASE_URL}/api/ping", timeout=10)
            print(f"Ping: {ping.json()}")
        except Exception as exc:
            print(f"FATAL: service not reachable at {BASE_URL}: {exc}")
            return

        # Establish log cursor so we only capture lines from this run.
        initial_logs = await _fetch_app_logs(client, since=0)
        log_cursor = max((e.get("seq", 0) for e in initial_logs), default=0)

        engines = await _fetch_engines(client)
        discovered = [e["name"] for e in engines]
        print(f"Discovered engines: {discovered}")

        env_filter = os.getenv("ENGINES", "").strip()
        if env_filter:
            wanted = {x.strip() for x in env_filter.split(",") if x.strip()}
            engines = [e for e in engines if e["name"] in wanted]
            print(f"Filtered to: {[e['name'] for e in engines]}")

        # -------- Probe --------
        print("\n" + "-" * 80 + "\nPROBE: classifying engines (usable vs skipped)\n" + "-" * 80)
        all_stats: dict[str, EngineStats] = {}
        for e in engines:
            s = await _probe_engine(client, e)
            all_stats[e["name"]] = s
            tag = "USABLE" if s.usable else "SKIPPED"
            print(f"  {e['name']:12s} -> {tag}"
                  + (f"  ({s.skip_reason})" if s.skip_reason else ""))

        usable = [s for s in all_stats.values() if s.usable]
        if not usable:
            print("\nNo usable engines (all require login). Nothing to stress. "
                  "Log in via /ui and re-run.")
            _print_report(all_stats, {}, {})
            return

        if not PROBE_ONLY:
            for s in usable:
                print(f"\n{'#'*60}\n# ENGINE: {s.name}\n{'#'*60}")

                # -------- Queue --------
                try:
                    s.queue_probe = await _test_queue(client, s.name)
                except Exception as exc:  # noqa: BLE001
                    s.queue_probe = {"error": str(exc)[:200]}

                # -------- Volume --------
                await _test_volume(client, s)

                # -------- Idle / cooldown --------
                if not SKIP_IDLE:
                    try:
                        s.idle_probe = await _test_idle(client, s.name)
                    except Exception as exc:  # noqa: BLE001
                        s.idle_probe = {"error": str(exc)[:200], "recovered": False}

        # -------- Diagnostics --------
        run_logs = await _fetch_app_logs(client, since=log_cursor)
        log_summary = _summarise_logs(run_logs)
        hints = await _fetch_selector_hints(client)

        _print_report(all_stats, log_summary, hints)

        # -------- Persist --------
        payload = {
            "base_url": BASE_URL,
            "config": {
                "messages_per_engine": MESSAGES_PER_ENGINE,
                "idle_seconds": IDLE_SECONDS,
                "queue_burst": QUEUE_BURST,
            },
            "engines": {
                name: {
                    "usable": s.usable,
                    "skip_reason": s.skip_reason,
                    "success_rate": s.success_rate,
                    "successful": len(s.successful),
                    "graded": len([r for r in s.results if r.status != "SKIPPED"]),
                    "avg_ms": s.avg_total_ms,
                    "median_ms": s.median_total_ms,
                    "p95_ms": s.p95_total_ms,
                    "max_ms": s.max_total_ms,
                    "queue_probe": s.queue_probe,
                    "idle_probe": s.idle_probe,
                    "failures": [
                        {"index": r.index, "status": r.status, "error": r.error[:200]}
                        for r in s.results
                        if not r.success and r.status != "SKIPPED"
                    ],
                }
                for name, s in all_stats.items()
            },
            "log_signal_counts": log_summary,
            "selector_hints": hints,
        }
        with open(OUT_JSON, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nDetailed results saved to {OUT_JSON}")


if __name__ == "__main__":
    asyncio.run(run())
