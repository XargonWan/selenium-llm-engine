# Agent Guidance for selenium-llm-engine

## General purpose
This document defines the expected behavior of agents working on this repository.
For runtime behavior, gotchas and operational context (session/login
persistence, local Docker workflow, the MCP server), see `CLAUDE.md`.

1. Check and verify documentation
   - Read and align with the instructions and features defined in:
     - `README.md`
     - `DEVELOPERS.md`
     - `CLAUDE.md`
   - Ensure implementations and APIs follow documented usage flows.

2. Diagnose LOGs
   - read container logs if the container is running

3. Fix behavior
   - If a bug is reported (user request), apply the minimal required fix + regression test.
   - Update existing tests or add new tests in `tests/test_api_endpoints.py` to cover behavior.

4. Formatting and style
   - Use idiomatic Python; specifically, avoid JSON-style constructs (`true`/`false`) in Python code.

5. Always cehck your code
   - After any change, run quality checks: `ruff check .`, `python -m py_compile ...`, `pytest -q`.
   - Do not deliver any broken code

6. Never `git commit` or `git add` if not explicitly requested.

7. Engine agnosticism — strictly enforced
   - The `core/` package **must not** contain any reference to a specific engine
     (e.g. `chatgpt`, `gemini`, `copilot`, `grok`, `perplexity`, or any other
     named LLM service).
   - All engine-specific data (selectors, URLs, models, aliases) belongs
     **exclusively** in the `engines/` directory as `.json` or `.py` files.
   - There is **no built-in engine fallback**: if `engines/` is empty the
     manager starts with no engines and logs a warning. Do not re-introduce a
     fallback list.
   - Docstring examples in `core/` must use placeholder names (e.g.
     `"my-engine"`) rather than real engine names.
   - Any violation of this rule is a bug and must be fixed immediately.

## Mandatory checks during review
- `ruff check .` must pass.
- If the change involves APIs, add request tests with valid and invalid payloads.
- Document the solution
