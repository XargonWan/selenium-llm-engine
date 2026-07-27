#!/usr/bin/env python3
"""Heavy prompting smoke test against a running selenium-llm-engine instance.

Sends a long prompt (to exercise the chunking path) plus a few short prompts to
a target engine via the OpenAI-compatible ``/v1/chat/completions`` endpoint and
reports latency / success for each. Intended as a manual, real end-to-end check
after a rebuild — it needs an *authenticated* engine session on the target
instance (the browser profile must already be logged in).

Usage:
    python scripts/heavy_prompt_test.py \
        --base-url http://localhost:14848 \
        --model gemini \
        --rounds 3

The script is engine-agnostic: the engine/model name is supplied on the command
line, never hard-coded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def _post_chat(base_url: str, model: str, content: str, timeout: float) -> dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _extract_text(payload: dict) -> str:
    try:
        return payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _run_case(base_url: str, model: str, label: str, content: str, timeout: float) -> bool:
    print(f"\n=== {label} (prompt chars={len(content)}) ===", flush=True)
    started = time.time()
    try:
        payload = _post_chat(base_url, model, content, timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:800]
        print(f"  HTTP {exc.code} after {time.time() - started:.1f}s: {detail}", flush=True)
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR after {time.time() - started:.1f}s: {exc!r}", flush=True)
        return False
    elapsed = time.time() - started
    text = _extract_text(payload)
    preview = text.strip().replace("\n", " ")[:200]
    ok = bool(text.strip())
    print(f"  {'OK' if ok else 'EMPTY'} in {elapsed:.1f}s, reply chars={len(text)}", flush=True)
    print(f"  reply preview: {preview}", flush=True)
    return ok


def _build_long_prompt() -> str:
    # A long, self-contained instruction that forces a large single message so
    # the engine's chunking path is exercised, while asking for a short answer
    # so response detection is quick.
    filler_lines = [
        f"Fact #{i}: item {i} has value {i * 7 % 101} and tag T{i % 13}."
        for i in range(1, 401)
    ]
    body = "\n".join(filler_lines)
    return (
        "You will receive a long list of facts. Read them all, then answer with "
        "a SINGLE short line only.\n\n"
        f"{body}\n\n"
        "Question: reply with exactly the word DONE followed by the number of "
        "facts you received."
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:14848")
    parser.add_argument("--model", required=True, help="Engine/model name, e.g. my-engine")
    parser.add_argument("--rounds", type=int, default=3, help="Repeat count for the heavy prompt")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    print(f"Target: {args.base_url}  model={args.model}  rounds={args.rounds}", flush=True)

    results: list[bool] = []

    # 1) Short warm-up prompt.
    results.append(
        _run_case(
            args.base_url,
            args.model,
            "warm-up short prompt",
            "Reply with exactly: PONG",
            args.timeout,
        )
    )

    # 2) Repeated heavy prompts (chunking + stress to surface transient errors
    #    such as the engine "something went wrong" toast).
    long_prompt = _build_long_prompt()
    for r in range(1, args.rounds + 1):
        results.append(
            _run_case(
                args.base_url,
                args.model,
                f"heavy chunked prompt round {r}/{args.rounds}",
                long_prompt,
                args.timeout,
            )
        )

    ok_count = sum(1 for x in results if x)
    total = len(results)
    print(f"\n==== SUMMARY: {ok_count}/{total} cases returned a non-empty reply ====", flush=True)
    return 0 if ok_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
