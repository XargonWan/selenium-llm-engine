# DEVELOPERS.md — Adding New Engines to selenium-llm-engine

This guide explains how to add a new LLM web interface as an engine.
Engines are discovered automatically from the `engines/` directory — **no
code changes to the core are required.**

---

## Quick start

1. Copy the template or write a new JSON file:
   ```
   engines/my_service.json
   ```
2. Start (or reload) the service. The new engine will appear automatically in:
   - `GET /api/engines`
   - `GET /models`
   - The **Engines** panel in the web UI
   - The engine dropdown in the **Send Prompt** panel

3. To reload without restarting, click **⟳ Reload** in the UI or call:
   ```
   POST /api/engines/reload
   ```

---

## Choose: JSON or Python?

| Scenario | Use |
|----------|-----|
| Standard chat interface (type → click send → read response) | **JSON** |
| Multi-step auth (CAPTCHA, OTP, OAuth redirect) | **Python** |
| Custom model switching via DOM interaction | **Python** |
| Non-standard send flow (e.g. file upload, voice) | **Python** |
| Everything else | **JSON** (always try JSON first) |

---

## Option A — JSON engine (recommended)

Create a file `engines/<name>.json` with the schema below.
Only `name` and `service_url` are required; everything else is optional.

```json
{
  "name": "my_service",
  "display_name": "My LLM Service",
  "aliases": ["my_service", "mls"],
  "service_url": "https://chat.example.com",
  "default_model": "ultra",
  "models": {
    "ultra":   100000,
    "default":  50000,
    "unlogged": 10000
  },
  "selectors": {
    "prompt_area":   ["div[contenteditable='true']", "textarea"],
    "send_button":   ["button[type='submit']", "button[aria-label*='Send']"],
    "response_area": [".assistant-message", "div.markdown"],
    "stop":          ["button[aria-label*='Stop']"]
  },
  "login_detection": {
    "url_prefix":                 "https://chat.example.com",
    "url_deny_keywords":          ["login", "signin", "auth"],
    "login_button_xpath":         "//button[contains(normalize-space(.), 'Sign in')]",
    "authenticated_css_selectors": ["#chat-container", ".user-avatar"]
  }
}
```

### Field reference

#### Top level

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **yes** | Unique lowercase identifier (used in API paths). |
| `service_url` | string | **yes** | URL of the chat interface. |
| `display_name` | string | no | Human-readable name shown in the UI. |
| `aliases` | string[] | no | Alternative names for `GET /engine/<alias>/prompt`. |
| `default_model` | string | no | Model slug used when logged in. Should also appear in `models`. |
| `allow_unlogged` | boolean | no | If true and `models` includes `unlogged`, `get_current_model()` returns `unlogged` when not logged in. |

#### `models`

A JSON object mapping model slugs to **max prompt character limits**.

```json
"models": {
  "gpt-4o":  60000,
  "unlogged": 20000,
  "default":  51000
}
```

The special slug `"unlogged"` is returned by `get_current_model()` when the
browser is not authenticated. The special slug `"default"` is used as a
fallback when the active model name is not in the map.

#### `selectors`

CSS selector lists, tried in order. The engine stops at the first one that
matches a clickable element.

| Key | What it targets |
|-----|----------------|
| `prompt_area` | The chat input field (textarea or contenteditable div). |
| `send_button` | The button that submits the prompt. |
| `response_area` | The container(s) of the assistant's replies. |
| `stop` | The button that cancels an in-progress generation. |

**Tips for finding selectors:**
- Open the site in Chrome DevTools → Inspect element → right-click → Copy → Copy selector.
- Prefer `data-testid` or `aria-label` attributes — they are more stable than class names.
- Add multiple fallback selectors from most specific to least specific.

#### `media_support`

Optional media support rules for engines that can upload or paste images, audio, or documents.
Each supported media type is a key mapping to an object with optional limits and model filters.

```json
"media_support": {
  "image": {
    "limits": {"unlogged": 0, "base": 3, "paid": -1},
    "supported_models": ["default", "all"]
  },
  "audio": {
    "limits": {"unlogged": 0, "base": -1, "paid": -1},
    "supported_models": ["not-unlogged"]
  }
}
```

Supported keys in each media entry:

- `limits` — per-tier upload limits. Use `-1` for unlimited; a tier limit of `0` means the engine should still attempt the upload flow before failing.
- `supported_models` — optional model whitelist. Special values:
  - `"all"` — all models are considered supported.
  - `"not-unlogged"` — any logged-in model is considered supported, but `unlogged` is not.
  - normal model slugs like `"default"`, `"gpt-4o"`, or `"unlogged"`.

If `supported_models` is present, the engine will still attempt the upload flow even when
the current model is not listed; this is a hint to developers, not a hard reject.
Only explicit limits or missing media type entries are treated as hard failures.

#### `login_detection`

Controls how the engine decides whether the browser session is logged in.
Detection runs in this order and stops at the first match:

| Step | Key | Effect |
|------|-----|--------|
| 1 | `url_prefix` | Navigate to this URL if `driver.current_url` does not start with it. |
| 2 | `url_deny_keywords` | If any keyword is found in the current URL → **not logged in**. |
| 3 | `login_button_xpath` | If any visible button matches the XPath → **not logged in**. |
| 4 | `authenticated_css_selectors` | If any listed element is visible → **logged in**. |
| 5 | *(fallback)* | Assume logged in when none of the above triggered. |

All four keys are optional. Omit the whole `login_detection` block for
public / no-auth services.

---

## Option B — Python engine

Use this when the site needs custom logic.

1. Copy `engines/_example_custom.py.dist` to `engines/my_engine.py`.
2. Fill in the required class attributes and implement `_ensure_logged_in()`.
3. Override additional base-class methods only if needed.

### Required class attributes

```python
class MyEngine(SeleniumLLMBase):
    ENGINE_NAME         = "my_engine"        # unique ID
    ENGINE_ALIASES      = ["my_engine"]      # alternate names
    ENGINE_DISPLAY_NAME = "My Engine"        # UI label
    ENGINE_SERVICE_URL  = "https://…"        # home URL
    ENGINE_MODELS       = {"default": 10000} # name → char limit
    ENGINE_DEFAULT_MODEL = "default"
```

The engine manager discovers your class automatically via `ENGINE_NAME`.

### Files starting with `_` are ignored

Files named `_something.py` or `_something.json` are **never** loaded.
Use this convention for:
- Template files (like `_example_custom.py.dist`)
- Disabled engines
- Work-in-progress files

---

## Testing your engine

1. Drop your file into `engines/`.
2. Call `POST /api/engines/reload` (or restart the service).
3. Verify it appears in `GET /api/engines`.
4. Trigger a login: `POST /login/<name>`
5. Send a prompt: `POST /engine/<name>/prompt` with `{"prompt": "Hello"}`

### Minimal curl smoke test

```bash
# 1. Discover engines
curl -s http://localhost:8000/api/engines | python3 -m json.tool

# 2. Check login state
curl -s http://localhost:8000/login/my_service/state

# 3. Send a prompt
curl -s -X POST http://localhost:8000/engine/my_service/prompt \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello, world!"}'
```

---

## JSON vs Python feature comparison

| Feature | JSON | Python |
|---------|------|--------|
| URL / selectors / models | ✅ | ✅ |
| Standard login detection | ✅ | ✅ |
| Multi-step auth / CAPTCHA | ❌ | ✅ |
| Custom model switching | ❌ | ✅ |
| Custom response parsing | ❌ | ✅ |
| Requires code changes to core | No | No |
| Hot-reload via API | ✅ | ✅ |

---

## API reference (discovery)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/engines` | GET | List all discovered engines (metadata only, no browser instance created). |
| `/api/engines/reload` | POST | Re-scan `engines/` and update the registry. Returns new engine list. |
| `/engine/{name}/prompt` | POST | Send a prompt to any engine by name or alias. Body: `{"prompt": "..."}`. |
| `/models` | GET | Legacy endpoint — same data as `/api/engines` in a different shape. |

---

## Contribution checklist

- [ ] File is named `engines/<slug>.json` or `engines/<slug>.py` (no spaces, lowercase).
- [ ] `name` / `ENGINE_NAME` matches the filename slug.
- [ ] At least one selector in each `selectors.*` list.
- [ ] `"unlogged"` model entry present in `models` and `"allow_unlogged": true` if the site allows unauthenticated use.
- [ ] Tested with `POST /api/engines/reload` and a real prompt (or mock test).
- [ ] PR description mentions which site was tested and Chromium version used.
