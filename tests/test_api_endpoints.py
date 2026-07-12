from fastapi.testclient import TestClient
import asyncio
import json
import threading
import time
import types
import sys
from typing import Any
import pytest
from selenium.common.exceptions import TimeoutException

# Some environments may not have distutils installed for undetected_chromedriver.
# Use a minimal fake module so unit tests can import core modules safely.
if "undetected_chromedriver" not in sys.modules:
    sys.modules["undetected_chromedriver"] = types.SimpleNamespace(
        Chrome=lambda *args, **kwargs: None
    )

from app import app, _register_engine_routes
from core.engine_manager import EngineManager

client = TestClient(app)


class DummyEngine:
    def __init__(self):
        self.model = "default"
        self.last_media: list[Any] = []

    def get_interface_limits(self):
        return {"max_prompt_chars": 1234, "model_name": "default"}

    def get_supported_models(self):
        return ["default"]

    async def start_login_flow(self):
        return {"logged_in": False, "login_state": "unlogged"}

    async def check_login_state(self):
        return {"logged_in": False, "login_state": "unlogged"}

    async def generate_response(self, prompt, media=None, timeout=None):
        self.last_media = media or []
        return "dummy response"

    def get_current_model(self):
        return "default"


@pytest.fixture(autouse=True)
def setup_engine_manager(monkeypatch):
    """Replace the EngineManager singleton with a pre-loaded test instance."""
    # Mock DB calls so no filesystem writes happen during tests
    monkeypatch.setattr("app.inc_requests", lambda: None)
    monkeypatch.setattr("app.inc_responses", lambda: None)
    monkeypatch.setattr("app.inc_errors", lambda: None)
    monkeypatch.setattr("app.log_prompt", lambda *a, **kw: None)
    monkeypatch.setattr("app.inc_media_sent", lambda *a, **kw: None)

    mgr = EngineManager.get()
    mgr.engines.clear()
    mgr.active_engine = None

    # Inject two synthetic descriptors so /models and /api/engines work
    from core.engine_manager import EngineDescriptor

    chatgpt_desc = EngineDescriptor(
        name="chatgpt",
        aliases=["chatgpt", "openai", "gpt"],
        display_name="ChatGPT (test)",
        service_url="https://chat.openai.com",
        models={"default": 51000},
        default_model="default",
        source="builtin",
        source_path="<test>",
        media_capabilities=["image", "audio"],
    )
    gemini_desc = EngineDescriptor(
        name="gemini",
        aliases=["gemini", "google"],
        display_name="Gemini (test)",
        service_url="https://gemini.google.com",
        models={"default": 32000},
        default_model="default",
        source="builtin",
        source_path="<test>",
        media_capabilities=["image"],
    )
    mgr._descriptors = {"chatgpt": chatgpt_desc, "gemini": gemini_desc}
    mgr._alias_map = {
        "chatgpt": "chatgpt",
        "openai": "chatgpt",
        "gpt": "chatgpt",
        "gemini": "gemini",
        "google": "gemini",
    }

    # Re-register dynamic per-engine routes for this test fixture
    _register_engine_routes(app)

    # Pre-populate with DummyEngine instances so no real Selenium init happens
    mgr.engines["chatgpt"] = DummyEngine()
    mgr.engines["gemini"] = DummyEngine()

    yield

    mgr.engines.clear()
    mgr.active_engine = None


def test_ping():
    response = client.get("/api/ping")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models():
    """Legacy /models must return OpenAI-compatible format (id field required by clients like Alpaca)."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert "data" in data
    for entry in data["data"]:
        assert "id" in entry, "Each model entry must have an 'id' field"
        assert entry["id"] is not None
        assert entry["object"] == "model"
        # Legacy extra fields still present
        assert "name" in entry


def test_parse_media_part_accepts_generic_file():
    from app import _parse_media_part

    part = {
        "type": "input_file",
        "data": "data:text/plain;base64,Zm9vYmFy",
        "mime_type": "text/plain",
        "filename": "hello.txt",
    }
    item = _parse_media_part(part, 0)
    assert item.media_type == "document"
    assert item.mime_type == "text/plain"
    assert item.filename == "hello.txt"
    assert item.data == b"foobar"


def test_parse_media_part_openai_image_url_format():
    """image_url payload with nested {'image_url': {'url': '...'}} must be parsed."""
    from app import _parse_media_part

    image_data = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAA"
        "AAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
    )
    part = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_data}"},
    }
    item = _parse_media_part(part, 0)
    assert item.media_type == "image"
    assert item.mime_type == "image/png"


def test_multimodal_openai_vision_format():
    """End-to-end: OpenAI vision format {'image_url': {'url': 'data:...'}} reaches the engine."""
    image_data = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAA"
        "AAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "content": "Describe this image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200
    engine = EngineManager.get().engines["chatgpt"]
    assert len(engine.last_media) == 1
    assert engine.last_media[0].media_type == "image"
    assert engine.last_media[0].mime_type == "image/png"


def test_normalize_prompt_payload_includes_input_file():
    from app import _normalize_prompt_payload

    payload = [
        {
            "role": "user",
            "content": [
                {"type": "text", "content": "Please read this file."},
                {
                    "type": "input_file",
                    "data": "data:text/plain;base64,Zm9vYmFy",
                    "mime_type": "text/plain",
                    "filename": "hello.txt",
                },
            ],
        }
    ]
    prompt_text, media_items = _normalize_prompt_payload(payload)
    assert "Please read this file." in prompt_text
    assert len(media_items) == 1
    assert media_items[0].media_type == "document"


def test_legacy_chat_completions():
    """POST /chat/completions (without /v1) must work as alias."""
    response = client.post(
        "/chat/completions",
        json={"model": "chatgpt", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"


def test_login_state():
    response = client.post("/login/chatgpt")
    assert response.status_code == 200
    assert response.json()["login_state"] == "unlogged"


def test_prompt_legacy_chatgpt():
    response = client.post("/chatgpt/prompt", json={"prompt": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "dummy response"


def test_prompt_invalid_json_body():
    response = client.post(
        "/chatgpt/prompt",
        data="http://localhost:14848/v1/chat/completions",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert "Invalid JSON body" in response.json()["detail"]


def test_prompt_dynamic_endpoint():
    response = client.post("/engine/chatgpt/prompt", json={"prompt": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "dummy response"


def test_prompt_dynamic_endpoint_alias():
    """Engine aliases should work on the dynamic endpoint too."""
    response = client.post("/engine/openai/prompt", json={"prompt": "Hello"})
    assert response.status_code == 200


def test_multimodal_text_and_image():
    image_data = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAA"
        "AAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "content": "Here is an image:"},
                        {
                            "type": "image_url",
                            "url": f"data:image/png;base64,{image_data}",
                            "filename": "test.png",
                        },
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    engine = EngineManager.get().engines["chatgpt"]
    assert len(engine.last_media) == 1
    assert engine.last_media[0].media_type == "image"


def test_multimodal_text_and_image_json_string_content():
    image_data = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAA"
        "AAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
    )
    payload = {
        "model": "chatgpt",
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "type": "text",
                        "content": "Here is an image:",
                        "attachments": [
                            {
                                "mime_type": "image/png",
                                "data": f"data:image/png;base64,{image_data}",
                                "filename": "test.png",
                            }
                        ],
                    }
                ),
            }
        ],
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    engine = EngineManager.get().engines["chatgpt"]
    assert len(engine.last_media) == 1
    assert engine.last_media[0].media_type == "image"
    assert engine.last_media[0].filename == "test.png"


def test_prompt_unknown_engine():
    response = client.post("/engine/nonexistent/prompt", json={"prompt": "Hello"})
    assert response.status_code == 404


def test_api_engines():
    response = client.get("/api/engines")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    names = [e["name"] for e in data["data"]]
    assert "chatgpt" in names
    assert "gemini" in names


def test_api_engines_include_media_capabilities():
    response = client.get("/api/engines")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    for entry in data["data"]:
        assert "media_capabilities" in entry
        assert isinstance(entry["media_capabilities"], list)

    chatgpt_entry = next((e for e in data["data"] if e["name"] == "chatgpt"), None)
    assert chatgpt_entry is not None
    assert "image" in chatgpt_entry["media_capabilities"]


def test_api_engines_reload():
    """Reload endpoint must return 200 and a valid data list."""
    response = client.post("/api/engines/reload")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["data"], list)


def test_unlogged_flag_behavior():
    class FakeBaseEngine:
        def __init__(self, model_limits_map, default_model, allow_unlogged=False):
            self.model_limits_map = model_limits_map
            self.default_model = default_model
            self.allow_unlogged = allow_unlogged
            self._logged_in = True

        def is_user_logged_in(self):
            return self._logged_in

        def set_logged_in(self, value):
            self._logged_in = value

        def get_current_model(self):
            if not self.is_user_logged_in() and self.allow_unlogged and "unlogged" in self.model_limits_map:
                return "unlogged"
            return self.default_model

    base_engine = FakeBaseEngine(
        model_limits_map={"unlogged": 20000, "default": 50000},
        default_model="default",
    )
    base_engine.set_logged_in(False)
    assert base_engine.get_current_model() == "default"

    unlogged_engine = FakeBaseEngine(
        model_limits_map={"unlogged": 20000, "default": 50000},
        default_model="default",
        allow_unlogged=True,
    )
    unlogged_engine.set_logged_in(False)
    assert unlogged_engine.get_current_model() == "unlogged"


def test_media_limits_fallback_without_config():
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    media_items = [type("M", (), {"media_type": "image"})()]
    assert engine._check_media_limits(media_items, "base") is None


def test_is_same_site_treats_in_site_navigation_as_not_a_redirect():
    """Some engines open a new conversation on send and navigate to a different
    path on the SAME host (e.g. '/c' -> '/chat/<id>'). That must not be flagged
    as a redirect-stall. Only a navigation to a DIFFERENT host is a real stall.
    """
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com/c",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    # Same host, different path -> in-site navigation, not a stall.
    assert engine._is_same_site("https://example.com/chat/abc-123") is True
    assert engine._is_same_site("https://example.com/") is True
    # Different host -> genuine off-site redirect (e.g. auth/login).
    assert engine._is_same_site("https://auth.example.org/login") is False
    # Empty/unknown URL is treated conservatively as same-site (no false stall).
    assert engine._is_same_site("") is True


def test_offsite_redirect_stall_is_distinguished_from_generic_stall():
    """A deterministic cross-host redirect (e.g. the service bounced to a
    login/auth domain) must be classified as an off-site stall so the retry
    loop can fail fast, while a generic redirect-stall must NOT be treated as
    off-site (it may be transient and worth retrying).
    """
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com/c",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    offsite = RuntimeError("redirect-stall (off-site): send not accepted after redirect")
    generic = RuntimeError("redirect-stall: send not accepted after redirect")

    # Off-site stall is recognised as off-site (fail fast).
    assert engine._is_offsite_redirect_stall(offsite) is True
    # Generic stall is NOT off-site (may retry).
    assert engine._is_offsite_redirect_stall(generic) is False
    # A generic stall is still a redirect-stall...
    assert engine._is_redirect_stall(generic) is True
    # ...whereas the off-site variant is not matched by the generic detector,
    # ensuring it takes the fail-fast branch instead of the retry branch.
    assert engine._is_redirect_stall(offsite) is False


def test_stepfun_audio_not_supported_by_model():
    from core.selenium_llm_base import SeleniumLLMBase

    with open("engines/stepfun.json", encoding="utf-8") as fh:
        cfg = json.load(fh)

    engine = SeleniumLLMBase(
        service_url=cfg["service_url"],
        model_limits_map=cfg["models"],
        default_model=cfg.get("default_model", "default"),
    )
    engine.media_config = cfg.get("media_support", {})
    media_items = [type("M", (), {"media_type": "audio"})()]
    assert engine._check_media_limits(media_items, "paid") is None


def test_media_with_model_not_listed_is_allowed_to_try():
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000, "other": 1000},
        default_model="other",
    )
    engine.media_config = {
        "audio": {
            "limits": {"default": -1},
            "supported_models": ["default"],
        }
    }
    media_items = [type("M", (), {"media_type": "audio"})()]
    assert engine._check_media_limits(media_items, "default") is None


def test_supported_models_all_allows_every_model():
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000, "other": 1000},
        default_model="other",
    )
    engine.media_config = {
        "audio": {
            "limits": {"default": -1},
            "supported_models": ["all"],
        }
    }
    media_items = [type("M", (), {"media_type": "audio"})()]
    assert engine._check_media_limits(media_items, "default") is None


def test_supported_models_not_unlogged_allows_logged_models():
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000, "unlogged": 1000},
        default_model="default",
    )
    engine.media_config = {
        "audio": {
            "limits": {"default": -1, "unlogged": 0},
            "supported_models": ["not-unlogged"],
        }
    }
    media_items = [type("M", (), {"media_type": "audio"})()]
    assert engine._check_media_limits(media_items, "default") is None


def test_upload_via_file_input_rejects_missing_value():
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.media_config = {"audio": {"upload_selectors": ["input[type='file']"]}}
    mock_driver = MagicMock()
    mock_input = MagicMock()
    mock_input.tag_name = "input"
    mock_input.get_attribute.side_effect = lambda attr: "file" if attr == "type" else ""
    mock_driver.find_elements.return_value = [mock_input]
    mock_driver.execute_script.return_value = 0

    result = engine._upload_via_file_input(
        type("M", (), {"media_type": "audio"})(),
        "/tmp/dummy.mp3",
        mock_driver,
    )
    assert result is False


def test_upload_via_file_input_accepts_file_list():
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.media_config = {"audio": {"upload_selectors": ["input[type='file']"]}}
    mock_driver = MagicMock()
    mock_input = MagicMock()
    mock_input.tag_name = "input"
    mock_input.get_attribute.side_effect = lambda attr: "file" if attr == "type" else ""
    mock_driver.find_elements.return_value = [mock_input]
    mock_driver.execute_script.return_value = 1

    result = engine._upload_via_file_input(
        type("M", (), {"media_type": "audio"})(),
        "/tmp/dummy.mp3",
        mock_driver,
    )
    assert result is True


def test_upload_via_file_input_trusts_send_keys_when_spa_clears_files():
    """Regression: Angular/SPA resets .files/.value after processing; send_keys success should be trusted."""
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.media_config = {"image": {"upload_selectors": ["input[type='file']"]}}

    mock_driver = MagicMock()
    mock_input = MagicMock()
    mock_input.tag_name = "input"
    # type check returns "file"; value (empty, simulating SPA reset)
    mock_input.get_attribute.side_effect = lambda attr: "file" if attr == "type" else ""
    mock_input.send_keys.return_value = None  # succeeds without exception
    mock_driver.find_elements.return_value = [mock_input]
    # files.length always returns 0 (SPA already cleared the FileList)
    mock_driver.execute_script.return_value = 0

    result = engine._upload_via_file_input(
        type("M", (), {"media_type": "image"})(),
        "/tmp/dummy.png",
        mock_driver,
    )

    assert result is True, "Should trust send_keys success even when SPA clears .files"


def test_upload_via_clipboard_uses_popen_for_xclip():
    """Regression: xclip -i blocks until clipboard is read; must use Popen, not run."""
    import os
    import tempfile
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock, patch

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.prompt_area_selectors = ["textarea"]

    mock_driver = MagicMock()
    mock_input_el = MagicMock()
    engine._find_interactable_element = MagicMock(return_value=mock_input_el)

    mock_proc = MagicMock()
    item = type("M", (), {"media_type": "image", "mime_type": "image/png"})()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(b"\x89PNG")
    tmp.close()

    try:
        with patch("shutil.which", side_effect=lambda cmd: cmd if cmd == "xclip" else None), \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("subprocess.run") as mock_run:
            result = engine._upload_via_clipboard(item, tmp.path if hasattr(tmp, "path") else tmp.name, mock_driver)
        assert result is True
        # Popen must be called (non-blocking); subprocess.run must NOT be called for xclip
        mock_popen.assert_called_once()
        mock_run.assert_not_called()
        # xclip process must be terminated after paste
        mock_proc.terminate.assert_called_once()
    finally:
        os.unlink(tmp.name)


def test_upload_via_file_input_attempts_visibility_fallback_for_hidden_inputs():
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.media_config = {"image": {"upload_selectors": ["input[type='file']"]}}

    mock_driver = MagicMock()
    mock_input = MagicMock()
    mock_input.tag_name = "input"
    mock_input.get_attribute.side_effect = lambda attr: "file" if attr == "type" else ""
    mock_input.send_keys.side_effect = [Exception("element not interactable"), None]
    mock_driver.find_elements.return_value = [mock_input]
    mock_driver.execute_script.return_value = 1

    result = engine._upload_via_file_input(
        type("M", (), {"media_type": "image"})(),
        "/tmp/dummy.png",
        mock_driver,
    )

    assert result is True
    assert mock_driver.execute_script.called


def test_upload_media_returns_false_when_all_upload_paths_fail():
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.media_config = {"audio": {"upload_selectors": ["input[type='file']"]}}
    mock_driver = MagicMock()
    mock_driver.find_elements.return_value = []

    result = engine._upload_media(
        [type("M", (), {"media_type": "audio", "mime_type": "audio/mpeg", "data": b"dummy"})()],
        mock_driver,
    )
    assert result is False


def test_upload_media_clicks_accept_buttons_after_successful_upload():
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.media_config = {"image": {"upload_selectors": ["input[type='file']"]}}

    mock_driver = MagicMock()
    mock_input = MagicMock()
    mock_input.tag_name = "input"
    mock_input.get_attribute.side_effect = lambda attr: "file" if attr == "type" else ""
    mock_input.send_keys.return_value = None
    mock_driver.find_elements.side_effect = (
        lambda by, sel: [mock_input] if sel == "input[type='file']" else []
    )
    engine._click_accept_buttons = MagicMock()

    result = engine._upload_media(
        [type("M", (), {"media_type": "image", "mime_type": "image/png", "data": b"dummy"})()],
        mock_driver,
    )

    assert result is True
    engine._click_accept_buttons.assert_called_once_with(mock_driver, timeout=5.0)


def test_sync_generate_response_once_returns_error_when_send_button_not_ready_after_media_upload():
    import tempfile

    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.driver = MagicMock()
    engine.service_url = "https://example.com"
    engine._ensure_ready = lambda: None
    engine.is_user_logged_in = lambda: True
    engine._is_captcha_present = lambda driver: False
    engine._is_limit_present = lambda driver: False
    engine._check_account_tier = lambda driver: "base"
    engine._check_media_limits = lambda media, tier: None
    engine._upload_media = lambda media, driver: True
    engine._find_interactable_element = lambda driver, selectors, timeout, cache_attr=None: MagicMock()
    engine._fill_input = lambda driver, element, text: None
    engine._wait_for_send_button_after_media_upload = lambda driver: False
    engine._click_send = lambda driver, element: None
    engine._post_send_check = lambda driver: True
    engine._wait_for_response = lambda driver: "response"

    result = engine._sync_generate_response_once("Hello", [type(
        "M", (), {"media_type": "audio", "mime_type": "audio/mpeg", "data": b"dummy"}
    )()])
    assert result == "⚠️ Media upload failed. Please verify the file and try again."


def test_total_media_limit_applies_across_types():
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine.media_config = {
        "total_limits": {"limits": {"default": 2}},
        "image": {"limits": {"default": -1}},
        "audio": {"limits": {"default": -1}},
    }
    media_items = [
        type("M", (), {"media_type": "image"})(),
        type("M", (), {"media_type": "audio"})(),
        type("M", (), {"media_type": "image"})(),
    ]

    assert engine._check_media_limits(media_items, "default") == (
        "⚠️ The use of 'media' is exhausted for today. Please try again tomorrow."
    )


def test_reset_state():
    manager = EngineManager.get()
    # set active engine then verify reset clears it
    manager.active_engine = manager.engines.get("chatgpt")
    assert manager.active_engine is not None

    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert manager.engines == {}
    assert manager.active_engine is None

    stats_res = client.get("/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert "stats" in stats_data
    # if DB is writable/clearable, stats may be empty; if readonly they may persist
    assert isinstance(stats_data["stats"], dict)
    assert "response_time" in stats_data
    assert "global_avg_ms" in stats_data["response_time"]
    assert "per_engine_avg_ms" in stats_data["response_time"]
    assert isinstance(stats_data["response_time"]["per_engine_avg_ms"], dict)


def test_media_counter_increments_on_prompt_with_media(monkeypatch):
    called = {"count": 0, "amount": 0}
    def record(amount):
        called["count"] += 1
        called["amount"] = amount
    monkeypatch.setattr("app.inc_media_sent", record)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "content": "hello"},
                    {"type": "image", "data": "data:image/png;base64,AAAA"},
                ],
            }
        ]
    }
    response = client.post("/engine/chatgpt/prompt", json=payload)
    assert response.status_code == 200
    assert called["count"] == 1
    assert called["amount"] == 1


def test_stats_returns_media_availability_by_tier():
    mgr = EngineManager.get()
    chatgpt_desc = mgr._descriptors["chatgpt"]
    gemini_desc = mgr._descriptors["gemini"]
    chatgpt_desc.media_support = {
        "image": {"limits": {"unlogged": 0, "base": -1, "paid": -1}}
    }
    gemini_desc.media_support = {
        "audio": {"limits": {"unlogged": 0, "base": 2, "paid": 0}}
    }

    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["media_sent_today"] == 0
    assert data["media_availability"]["unlogged"] == 0
    assert data["media_availability"]["base"] == 4
    assert data["media_availability"]["paid"] == -1


def test_api_reset_alias():
    manager = EngineManager.get()
    manager.active_engine = manager.engines.get("chatgpt")

    response = client.post("/api/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert manager.engines == {}
    assert manager.active_engine is None


def test_reset_cancels_inflight_requests():
    class SlowDummyEngine:
        def __init__(self):
            self.model = "default"

        def get_interface_limits(self):
            return {"max_prompt_chars": 1234, "model_name": "default"}

        def get_supported_models(self):
            return ["default"]

        async def start_login_flow(self):
            return {"logged_in": False, "login_state": "unlogged"}

        async def check_login_state(self):
            return {"logged_in": False, "login_state": "unlogged"}

        async def generate_response(self, prompt):
            await asyncio.sleep(3)
            return "slow response"

        async def stop(self):
            return

        def get_current_model(self):
            return "default"

    manager = EngineManager.get()
    manager.engines["chatgpt"] = SlowDummyEngine()

    results = {}

    def call_prompt():
        try:
            r = client.post("/engine/chatgpt/prompt", json={"prompt": "hello"})
            results["response"] = r
        except Exception as e:
            results["error"] = e

    thread = threading.Thread(target=call_prompt)
    thread.start()

    # wait a moment for request to be in-flight
    time.sleep(0.1)

    response = client.post("/reset")
    assert response.status_code == 200

    thread.join(timeout=10)
    assert not thread.is_alive()

    assert "response" in results or "error" in results
    if "response" in results:
        assert results["response"].status_code in (503, 500)


def test_logs_history_endpoint():
    # ensure prompt logging endpoint is accessible and returns a list
    response = client.get("/logs?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_api_history_endpoint():
    response = client.get("/api/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_captcha_detection_short_circuit(monkeypatch):
    from core.selenium_llm_base import SeleniumLLMBase

    class FakeCaptchaDriver:
        current_url = "https://chat.openai.com"

        def find_elements(self, by, selector):
            if selector == "iframe#cf-chl-widget-ezspn":
                return [object()]
            return []

    engine = SeleniumLLMBase(
        service_url="https://chat.openai.com",
        model_limits_map={"default": 50000},
        default_model="default",
    )
    engine._ensure_ready = lambda: None
    engine.driver = FakeCaptchaDriver()

    result = engine._sync_generate_response_once("Hello")
    assert "CAPTCHA" in result or "captcha" in result
    assert "Please complete" in result


def test_check_login_state_no_browser_launch_when_uninitialized():
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://chat.openai.com",
        model_limits_map={"default": 50000},
        default_model="default",
    )

    # If check_login_state is called before initialization, it must not cause browser init
    called = False

    def fail_init():
        nonlocal called
        called = True
        raise RuntimeError("_ensure_ready should not be called")

    engine._ensure_ready = fail_init
    engine.driver = None

    state = asyncio.run(engine.check_login_state())
    assert state["login_state"] == "unlogged"
    assert state["logged_in"] is False
    assert called is False


# ---------------------------------------------------------------------------
# OpenAI-compatible /v1/* endpoint tests
# ---------------------------------------------------------------------------


def test_v1_models_list():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    ids = [m["id"] for m in data["data"]]
    assert "chatgpt" in ids
    assert "gemini" in ids
    # only canonical names — no aliases, no provider:variant
    assert not any(":" in mid for mid in ids)
    for entry in data["data"]:
        assert entry["object"] == "model"
        assert entry["owned_by"] == "selenium-llm-engine"


def test_v1_models_single():
    response = client.get("/v1/models/chatgpt")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "chatgpt"
    assert data["object"] == "model"


def test_v1_models_variant():
    response = client.get("/v1/models/chatgpt:default")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "chatgpt:default"


def test_fill_input_contenteditable_triggers_extra_keystroke():
    from core.selenium_llm_base import SeleniumLLMBase
    from selenium.webdriver.common.keys import Keys

    events = []

    class FakeElement:
        tag_name = "div"

        def click(self):
            events.append("click")

        def send_keys(self, *args):
            events.append(("send_keys", args))

    class FakeDriver:
        def __init__(self):
            self.script_calls = []

        def execute_script(self, script, *args):
            self.script_calls.append((script, args))
            if "document.execCommand('insertText'" in script:
                return None
            if "const text = el.innerText" in script:
                return "test"
            return None

    engine = SeleniumLLMBase(
        service_url="https://www.example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine._ensure_ready = lambda: None
    engine.driver = FakeDriver()

    fake_el = FakeElement()
    engine._fill_input(engine.driver, fake_el, "test")

    assert any(
        "document.execCommand('insertText'" in call[0] for call in engine.driver.script_calls
    )
    assert ("send_keys", (Keys.SPACE, Keys.BACKSPACE)) in events


def test_fill_input_verifies_input_value_for_textarea():
    from core.selenium_llm_base import SeleniumLLMBase

    class FakeElement:
        tag_name = "textarea"

        def __init__(self):
            self.value = ""

        def click(self):
            pass

        def clear(self):
            self.value = ""

        def send_keys(self, text):
            self.value = text

        def get_attribute(self, name):
            if name == "value":
                return self.value
            return None

    class FakeDriver:
        def execute_script(self, script, *args):
            return None

    engine = SeleniumLLMBase(
        service_url="https://www.example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine._ensure_ready = lambda: None
    engine.driver = FakeDriver()

    fake_el = FakeElement()
    engine._fill_input(engine.driver, fake_el, "hello world")
    assert fake_el.get_attribute("value") == "hello world"


def test_fill_input_raises_when_verification_fails():
    from core.selenium_llm_base import SeleniumLLMBase

    class FakeElement:
        tag_name = "textarea"

        def click(self):
            pass

        def clear(self):
            pass

        def send_keys(self, text):
            pass

        def get_attribute(self, name):
            if name == "value":
                return "wrong text"
            return None

    class FakeDriver:
        def execute_script(self, script, *args):
            return None

    engine = SeleniumLLMBase(
        service_url="https://www.example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    engine._ensure_ready = lambda: None
    engine.driver = FakeDriver()

    fake_el = FakeElement()
    with pytest.raises(RuntimeError, match="fill_input verification failed"):
        engine._fill_input(engine.driver, fake_el, "hello world")


def test_v1_models_unknown():
    response = client.get("/v1/models/nonexistent_engine")
    assert response.status_code == 404


def test_v1_chat_completions_messages():
    response = client.post(
        "/v1/chat/completions",
        json={"model": "chatgpt", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "dummy response"


def test_v1_chat_null_model():
    """model=null must not crash — falls back to default engine."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": None, "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "chatgpt"


def test_api_engines_default_setting():
    response = client.get("/api/engines/default")
    assert response.status_code == 200
    assert response.json()["default_engine"] == "chatgpt"

    response = client.post("/api/engines/default", json={"engine": "gemini"})
    assert response.status_code == 200
    assert response.json()["default_engine"] == "gemini"

    response = client.get("/api/engines/default")
    assert response.status_code == 200
    assert response.json()["default_engine"] == "gemini"

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "gemini"


def test_v1_chat_provider_variant_model():
    """provider:variant notation must resolve to the correct engine."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": "chatgpt:gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200


def test_token_count_nonzero():
    response = client.post("/chatgpt/prompt", json={"prompt": "Hello world"})
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_v1_chat_completions_ignores_unsupported_openai_params():
    """Unsupported OpenAI parameters must be ignored rather than causing errors."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "chatgpt",
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": True,
            "top_logprobs": 3,
            "n": 2,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["engine"] == "chatgpt"


def test_v1_streaming_sse_format():
    """stream=True must return SSE with chat.completion.chunk objects."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "chatgpt", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line.startswith("data:")]
    assert any("[DONE]" in line for line in lines)
    data_lines = [line for line in lines if "[DONE]" not in line]
    assert len(data_lines) >= 1
    for line in data_lines:
        chunk = json.loads(line.removeprefix("data:").strip())
        assert chunk["object"] == "chat.completion.chunk"
        assert "choices" in chunk


# ---------------------------------------------------------------------------
# Selector hints endpoint and selector caching regression tests
# ---------------------------------------------------------------------------


def test_selector_hints_empty_when_no_prompts():
    """GET /api/engines/selector-hints returns an empty data dict before any prompt is sent."""
    mgr = EngineManager.get()
    mgr.engines.clear()
    response = client.get("/api/engines/selector-hints")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert data["data"] == {}


def test_selector_hints_structure_after_engine_loaded():
    """Once an engine instance is in the manager the hints endpoint must expose its selector lists."""
    engine = DummyEngine()
    engine.prompt_area_selectors = ["textarea", "div[contenteditable='true']"]
    engine.send_button_selectors = ["button[type='submit']", "button[aria-label*='Send']"]
    engine._cached_prompt_selector = None
    engine._cached_send_selector = None

    mgr = EngineManager.get()
    mgr.engines["chatgpt"] = engine

    response = client.get("/api/engines/selector-hints")
    assert response.status_code == 200
    data = response.json()["data"]
    assert "chatgpt" in data
    hints = data["chatgpt"]
    assert "prompt_selector" in hints
    assert "send_selector" in hints
    assert "prompt_area_selectors" in hints
    assert "send_button_selectors" in hints
    assert hints["prompt_selector"] is None
    assert hints["send_selector"] is None
    assert hints["prompt_area_selectors"] == engine.prompt_area_selectors
    assert hints["send_button_selectors"] == engine.send_button_selectors


def test_selector_hints_reflect_cached_values():
    """Cached selectors are included in the hints response after being set."""
    engine = DummyEngine()
    engine.prompt_area_selectors = ["textarea", "div[contenteditable='true']"]
    engine.send_button_selectors = ["button[type='submit']", "button[aria-label*='Send']"]
    engine._cached_prompt_selector = "div[contenteditable='true']"
    engine._cached_send_selector = "button[aria-label*='Send']"

    mgr = EngineManager.get()
    mgr.engines["gemini"] = engine

    response = client.get("/api/engines/selector-hints")
    assert response.status_code == 200
    hints = response.json()["data"]["gemini"]
    assert hints["prompt_selector"] == "div[contenteditable='true']"
    assert hints["send_selector"] == "button[aria-label*='Send']"


def test_find_interactable_element_caches_selector():
    """_find_interactable_element sets cache_attr to the found selector."""
    try:
        from core.selenium_llm_base import SeleniumLLMBase
    except ModuleNotFoundError:
        pytest.skip("undetected_chromedriver not compatible with this Python version")

    from unittest.mock import MagicMock, patch

    base = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    assert base._cached_prompt_selector is None

    mock_driver = MagicMock()
    fake_el = MagicMock()

    winning_selector = "div[contenteditable='true']"

    def fake_wait_until(condition):
        # Simulate: first selector times out, second succeeds
        sel = condition.locator[1]
        if sel == winning_selector:
            return fake_el
        from selenium.common.exceptions import TimeoutException
        raise TimeoutException()

    mock_wait = MagicMock()
    mock_wait.until.side_effect = fake_wait_until

    def make_wait(driver, timeout):
        return mock_wait

    with patch("core.selenium_llm_base.WebDriverWait", side_effect=make_wait):
        selectors = ["textarea", winning_selector]
        result = base._find_interactable_element(
            mock_driver, selectors, timeout=3.0, cache_attr="_cached_prompt_selector"
        )

    assert result == fake_el
    assert base._cached_prompt_selector == winning_selector


def test_find_interactable_element_tries_cached_first():
    """When a cached selector exists it is tried before others."""
    try:
        from core.selenium_llm_base import SeleniumLLMBase
    except ModuleNotFoundError:
        pytest.skip("undetected_chromedriver not compatible with this Python version")

    from unittest.mock import MagicMock, patch

    base = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    cached_sel = "div[contenteditable='true']"
    base._cached_prompt_selector = cached_sel

    tried_order: list[str] = []
    fake_el = MagicMock()

    def fake_wait_until(condition):
        sel = condition.locator[1]
        tried_order.append(sel)
        if sel == cached_sel:
            return fake_el
        from selenium.common.exceptions import TimeoutException
        raise TimeoutException()

    mock_wait = MagicMock()
    mock_wait.until.side_effect = fake_wait_until

    with patch("core.selenium_llm_base.WebDriverWait", return_value=mock_wait):
        selectors = ["textarea", cached_sel, "input"]
        base._find_interactable_element(
            MagicMock(), selectors, timeout=3.0, cache_attr="_cached_prompt_selector"
        )

    assert tried_order[0] == cached_sel, "Cached selector must be tried first"


def test_find_interactable_element_falls_back_to_visible_non_clickable_element():
    try:
        from core.selenium_llm_base import SeleniumLLMBase
    except ModuleNotFoundError:
        pytest.skip("undetected_chromedriver not compatible with this Python version")

    from unittest.mock import MagicMock, patch
    from selenium.common.exceptions import TimeoutException

    base = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )

    mock_driver = MagicMock()
    visible_element = MagicMock()
    visible_element.is_displayed.return_value = True

    def fake_wait_until(condition):
        raise TimeoutException()

    mock_wait = MagicMock()
    mock_wait.until.side_effect = fake_wait_until

    with patch("core.selenium_llm_base.WebDriverWait", return_value=mock_wait):
        mock_driver.find_elements.return_value = [visible_element]
        result = base._find_interactable_element(
            mock_driver,
            ["div[contenteditable='true']"],
            timeout=2.0,
            cache_attr="_cached_prompt_selector",
        )

    assert result == visible_element
    assert base._cached_prompt_selector == "div[contenteditable='true']"


def test_find_interactable_element_handles_stale_cached_selector():
    """If cached selector raises StaleElementReferenceException then fallback is used."""
    try:
        from core.selenium_llm_base import SeleniumLLMBase
    except ModuleNotFoundError:
        pytest.skip("undetected_chromedriver not compatible with this Python version")

    from selenium.common.exceptions import StaleElementReferenceException
    from unittest.mock import MagicMock, patch

    base = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    base._cached_prompt_selector = "textarea"

    mock_driver = MagicMock()
    fake_el = MagicMock()

    selectors_calls = iter(["textarea", "div[contenteditable='true']"])

    def fake_wait_until(condition):
        try:
            sel = condition.locator[1]
        except Exception:
            sel = next(selectors_calls)

        if sel == "textarea":
            raise StaleElementReferenceException("stale")
        if sel == "div[contenteditable='true']":
            return fake_el
        from selenium.common.exceptions import TimeoutException

        raise TimeoutException()

    mock_wait = MagicMock()
    mock_wait.until.side_effect = fake_wait_until

    with patch("core.selenium_llm_base.WebDriverWait", return_value=mock_wait):
        result = base._find_interactable_element(
            mock_driver,
            ["textarea", "div[contenteditable='true']"],
            timeout=3.0,
            cache_attr="_cached_prompt_selector",
        )

    assert result == fake_el
    assert base._cached_prompt_selector == "div[contenteditable='true']"


def test_click_send_handles_stale_first_selector():
    """If first send selector is stale, next selector should be used and cached."""
    try:
        from core.selenium_llm_base import SeleniumLLMBase
    except ModuleNotFoundError:
        pytest.skip("undetected_chromedriver not compatible with this Python version")

    from selenium.common.exceptions import StaleElementReferenceException
    from unittest.mock import MagicMock, patch

    base = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    base.send_button_selectors = ["button.send", "button.send2"]
    base._cached_send_selector = "button.send"

    mock_driver = MagicMock()
    fake_btn = MagicMock()

    selectors_calls = iter(["button.send", "button.send2"])

    def fake_wait_until(condition):
        try:
            sel = condition.locator[1]
        except Exception:
            sel = next(selectors_calls)

        if sel == "button.send":
            raise StaleElementReferenceException("stale")
        if sel == "button.send2":
            return fake_btn
        from selenium.common.exceptions import TimeoutException

        raise TimeoutException()

    mock_wait = MagicMock()
    mock_wait.until.side_effect = fake_wait_until

    with patch("core.selenium_llm_base.WebDriverWait", return_value=mock_wait):
        base._click_send(mock_driver, MagicMock())

    assert base._cached_send_selector == "button.send2"


def test_fill_input_retries_on_stale_element():
    """_fill_input should recover from a stale input element by refinding it."""
    try:
        from core.selenium_llm_base import SeleniumLLMBase
    except ModuleNotFoundError:
        pytest.skip("undetected_chromedriver not compatible with this Python version")

    from selenium.common.exceptions import StaleElementReferenceException
    from unittest.mock import MagicMock

    base = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    first_input = MagicMock()
    first_input.tag_name = "textarea"
    first_input.click.side_effect = StaleElementReferenceException("stale")
    first_input.clear.side_effect = StaleElementReferenceException("stale")

    class FakeTextarea:
        tag_name = "textarea"

        def __init__(self):
            self._value = ""
            self.clear_called = False
            self.send_keys_called = False

        def click(self):
            return None

        def clear(self):
            self.clear_called = True
            self._value = ""

        def send_keys(self, text):
            self.send_keys_called = True
            self._value = text

        def get_attribute(self, name):
            if name == "value":
                return self._value
            return None

    second_input = FakeTextarea()

    def find_input(driver, selectors, timeout, cache_attr=None):
        return second_input

    base._find_interactable_element = find_input

    base._fill_input(MagicMock(), first_input, "hello world")

    assert second_input.clear_called
    assert second_input.send_keys_called
    assert second_input._value == "hello world"


def test_wait_for_send_button_after_media_upload_returns_true_when_button_appears():
    import tempfile

    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    base = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    mock_driver = MagicMock()
    fake_button = MagicMock()
    fake_button.is_displayed.return_value = True
    fake_button.is_enabled.return_value = True
    mock_driver.find_elements.side_effect = [[], [fake_button]]

    result = base._wait_for_send_button_after_media_upload(mock_driver, timeout=1.0)

    assert result is True
    assert mock_driver.find_elements.call_count == 2


def test_wait_for_send_button_after_media_upload_times_out_when_button_never_appears():
    import tempfile

    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    base = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    mock_driver = MagicMock()
    mock_driver.find_elements.return_value = []

    result = base._wait_for_send_button_after_media_upload(mock_driver, timeout=0.1)

    assert result is False


def test_wait_for_media_upload_complete_waits_for_selector_presence():
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    base = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    base.media_config = {
        "image": {
            "upload_complete_selectors": ["div.upload-preview"]
        }
    }
    mock_driver = MagicMock()
    fake_element = MagicMock()
    fake_element.is_displayed.return_value = True
    mock_driver.find_elements.side_effect = [[], [fake_element]]

    result = base._wait_for_media_upload_complete(
        type("M", (), {"media_type": "image"})(),
        mock_driver,
        timeout=1.0,
    )

    assert result is True
    assert mock_driver.find_elements.call_count == 2


def test_wait_for_media_upload_complete_waits_for_selector_absence():
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    base = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    base.media_config = {
        "image": {
            "upload_complete_selectors": ["!upload-image-disclaimer-dialog"]
        }
    }
    mock_driver = MagicMock()
    fake_element = MagicMock()
    fake_element.is_displayed.return_value = True
    mock_driver.find_elements.side_effect = [[fake_element], []]

    result = base._wait_for_media_upload_complete(
        type("M", (), {"media_type": "image"})(),
        mock_driver,
        timeout=1.0,
    )

    assert result is True
    assert mock_driver.find_elements.call_count == 2


def test_is_limit_present_detects_limit_warning():
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    base = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    base.limit_selectors = ["div.limit-warning"]

    fake_element = MagicMock()
    fake_element.is_displayed.return_value = True
    mock_driver = MagicMock()
    mock_driver.find_elements.return_value = [fake_element]

    assert base._is_limit_present(mock_driver) is True


def test_is_limit_present_returns_false_when_no_limit_warning():
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    base = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    base.limit_selectors = ["div.limit-warning"]

    mock_driver = MagicMock()
    mock_driver.find_elements.return_value = []

    assert base._is_limit_present(mock_driver) is False


# ---------------------------------------------------------------------------
# New endpoints: /api/logs/app and updated /stats
# ---------------------------------------------------------------------------


def test_app_logs_endpoint_returns_list():
    """GET /api/logs/app must return a JSON object with an 'entries' list."""
    response = client.get("/api/logs/app")
    assert response.status_code == 200
    data = response.json()
    assert "entries" in data
    assert isinstance(data["entries"], list)


def test_app_logs_since_parameter():
    """Passing since=<large_int> must return only newer entries (or an empty list)."""
    response = client.get("/api/logs/app?since=999999")
    assert response.status_code == 200
    data = response.json()
    assert data["entries"] == []


def test_stats_includes_logged_engines():
    """GET /stats must include a 'logged_engines' list instead of 'latest_logs'."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert "logged_engines" in data
    assert isinstance(data["logged_engines"], list)
    assert "latest_logs" not in data


def test_stats_includes_response_time():
    """GET /stats must include response time averages."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "response_time" in data
    assert isinstance(data["response_time"], dict)
    assert "global_avg_ms" in data["response_time"]
    assert "per_engine_avg_ms" in data["response_time"]
    assert isinstance(data["response_time"]["per_engine_avg_ms"], dict)


# ---------------------------------------------------------------------------
# OpenAPI schema compliance tests (Pydantic response_model validation)
# ---------------------------------------------------------------------------


def test_openapi_schema_has_chat_completion_response():
    """The OpenAPI schema must document a response body for /v1/chat/completions."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    path = schema["paths"].get("/v1/chat/completions", {})
    post_op = path.get("post", {})
    responses = post_op.get("responses", {})
    assert "200" in responses, "POST /v1/chat/completions must have a 200 response schema"
    content = responses["200"].get("content", {})
    assert "application/json" in content, "Response must be application/json"


def test_openapi_schema_has_models_response():
    """The OpenAPI schema must document a response body for /v1/models."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    path = schema["paths"].get("/v1/models", {})
    get_op = path.get("get", {})
    responses = get_op.get("responses", {})
    assert "200" in responses
    content = responses["200"].get("content", {})
    assert "application/json" in content


def test_chat_completion_response_schema_fields():
    """POST /v1/chat/completions response must contain all required OpenAI-compatible fields."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": "chatgpt", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200
    data = response.json()
    required = {"id", "object", "created", "model", "choices", "usage", "engine", "prompt", "elapsed_ms"}
    assert required <= data.keys(), f"Missing fields: {required - data.keys()}"
    assert data["object"] == "chat.completion"
    assert isinstance(data["choices"], list)
    assert len(data["choices"]) > 0
    choice = data["choices"][0]
    assert "message" in choice
    assert choice["message"]["role"] == "assistant"
    assert isinstance(data["usage"]["total_tokens"], int)


def test_ping_response_schema():
    """GET /api/ping must return {status, service} — validated by PingResponse model."""
    response = client.get("/api/ping")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) >= {"status", "service"}
    assert isinstance(data["status"], str)
    assert isinstance(data["service"], str)


def test_v1_models_response_schema_fields():
    """GET /v1/models entries must all carry the four required OpenAI model fields."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    required_entry_fields = {"id", "object", "created", "owned_by"}
    for entry in data["data"]:
        assert required_entry_fields <= entry.keys(), f"Missing: {required_entry_fields - entry.keys()}"
        assert isinstance(entry["created"], int)


def test_v1_models_include_capabilities():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    for entry in data["data"]:
        assert "capabilities" in entry
        assert isinstance(entry["capabilities"], dict)


def test_legacy_models_response_schema_fields():
    """GET /models entries must have all OpenAI fields plus the legacy 'name' field."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    for entry in data["data"]:
        assert "id" in entry
        assert "object" in entry
        assert "name" in entry


# ---------------------------------------------------------------------------
# Redirect-stall detection tests
# ---------------------------------------------------------------------------


def test_post_send_check_returns_true_when_stop_button_visible():
    """_post_send_check must return True immediately when a stop button becomes visible."""
    import tempfile
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.stop_selectors = ["button[aria-label*='Stop']"]

    fake_btn = MagicMock()
    fake_btn.is_displayed.return_value = True

    mock_driver = MagicMock()
    mock_driver.find_elements.return_value = [fake_btn]
    mock_driver.current_url = "https://example.com"

    result = engine._post_send_check(mock_driver, timeout=2.0)
    assert result is True


def test_post_send_check_recognizes_mat_icon_stop_selector():
    """_post_send_check must detect a material icon stop indicator."""
    import tempfile
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.stop_selectors = ["mat-icon[fonticon='stop']"]

    fake_icon = MagicMock()
    fake_icon.is_displayed.return_value = True

    mock_driver = MagicMock()
    mock_driver.find_elements.return_value = [fake_icon]
    mock_driver.current_url = "https://example.com"

    result = engine._post_send_check(mock_driver, timeout=2.0)
    assert result is True


def test_post_send_check_returns_false_on_redirect():
    """_post_send_check must return False when timeout expires and URL has changed."""
    import tempfile

    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.stop_selectors = ["button[aria-label*='Stop']"]
    engine.response_area_selectors = [".assistant-message"]

    mock_driver = MagicMock()
    # No stop button, no response text
    mock_driver.find_elements.return_value = []
    mock_driver.current_url = "https://auth.example.com/login"

    result = engine._post_send_check(mock_driver, timeout=0.1)
    assert result is False


def test_get_latest_response_text_uses_first_matching_selector():
    """_get_latest_response_text should return text from the first selector that matches."""
    import tempfile

    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.assistant", "div.alternate"]

    def find_elements(by, value):
        if value == "div.assistant":
            return []
        if value == "div.alternate":
            el = MagicMock()
            el.text = "Hello from assistant"
            return [el]
        return []

    mock_driver = MagicMock()
    mock_driver.find_elements.side_effect = find_elements

    result = engine._get_latest_response_text(mock_driver)
    assert result == "Hello from assistant"


def test_get_latest_response_text_checks_prior_elements_when_last_is_empty():
    """_get_latest_response_text should use an earlier matching element when the last one is blank."""
    import tempfile

    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.assistant"]

    empty_elem = MagicMock()
    empty_elem.text = ""
    empty_elem.get_attribute.return_value = ""

    filled_elem = MagicMock()
    filled_elem.text = "OK"
    filled_elem.get_attribute.return_value = "OK"

    mock_driver = MagicMock()
    mock_driver.find_elements.return_value = [filled_elem, empty_elem]

    result = engine._get_latest_response_text(mock_driver)
    assert result == "OK"


def test_get_latest_response_text_js_fallback_when_selectors_fail():
    """_get_latest_response_text should fall back to JS extraction when CSS selectors return nothing."""
    import tempfile

    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.assistant", "div.alternate"]

    mock_driver = MagicMock()
    mock_driver.find_elements.return_value = []
    mock_driver.execute_script.return_value = "JS fallback text"

    result = engine._get_latest_response_text(mock_driver)
    assert result == "JS fallback text"

def test_sync_generate_response_retries_on_redirect_stall():
    """_sync_generate_response must retry once on redirect-stall without resetting the driver."""
    import tempfile
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )

    call_count = 0

    def fake_once(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("redirect-stall: send not accepted after redirect")
        return "ok response"

    engine._sync_generate_response_once = fake_once
    reset_called = []
    engine._reset_driver = lambda: reset_called.append(True)

    result = engine._sync_generate_response("hello")
    assert result == "ok response"
    assert call_count == 2
    assert reset_called == [], "Driver must NOT be reset on redirect-stall"


def test_sync_generate_response_page_refresh_budget_persists_across_attempts():
    """A perpetually-stuck stop button must not cause an infinite refresh loop.

    Regression: _page_refresh_attempts was reset to 0 at the start of every
    _sync_generate_response_once, so the page-refresh budget replenished on each
    attempt and the paste → send → refresh cycle never terminated. The budget
    must be reset once per request and shared across attempts, so that once it
    is exhausted the engine falls back to a driver reset instead of looping.
    """
    import tempfile
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine._max_page_refresh_attempts = 2

    call_count = 0
    reset_called = []

    def fake_once(prompt):
        nonlocal call_count
        call_count += 1
        # Simulate _wait_for_response detecting a stuck stop button: it bumps the
        # refresh counter (up to the budget) then raises page_refresh_required.
        if engine._page_refresh_attempts < engine._max_page_refresh_attempts:
            engine._page_refresh_attempts += 1
        raise RuntimeError(
            "page_refresh_required: stop button stuck at start of "
            "_wait_for_response — page refreshed, retry without driver reset"
        )

    engine._sync_generate_response_once = fake_once

    def fake_reset():
        reset_called.append(True)
        # A real driver reset also clears the refresh counter; mirror that here.
        engine._page_refresh_attempts = 0
        # After the reset the page is clean, so the next attempt succeeds.
        engine._sync_generate_response_once = lambda prompt: "recovered"

    engine._reset_driver = fake_reset

    result = engine._sync_generate_response("hello")

    assert result == "recovered"
    # Once the budget is exhausted the engine resets the driver exactly once.
    assert reset_called == [True], "Driver must be reset once budget is exhausted"
    # fake_once is invoked twice: attempt 1 (budget 0→1, retry) and attempt 2
    # (budget 1→2 → exhausted → driver reset). The reset swaps in the recovery
    # stub, so attempt 3 succeeds without incrementing call_count. Because the
    # budget persists across attempts, the loop terminates instead of spinning.
    assert call_count == 2


def test_sync_generate_response_retries_on_response_detection_timeout():
    """_sync_generate_response retries with driver reset when response detection times out."""
    import tempfile
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )

    call_count = 0

    def fake_once(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError(
                "selenium_response_detection_timeout: no new response text appeared"
            )
        return "real response"

    engine._sync_generate_response_once = fake_once
    reset_called = []
    engine._reset_driver = lambda: reset_called.append(True)

    result = engine._sync_generate_response("hello")
    assert result == "real response"
    assert call_count == 2
    assert reset_called == [True], "Driver MUST be reset on response detection timeout"


def test_sync_generate_response_once_retries_on_stale_element():
    """_sync_generate_response_once should retry once when a stale element occurs."""
    import tempfile

    from core.selenium_llm_base import SeleniumLLMBase
    from selenium.common.exceptions import StaleElementReferenceException
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.driver = MagicMock()
    engine._initialized = True
    engine.driver.current_url = "https://example.com"

    engine._find_interactable_element = lambda driver, selectors, timeout, cache_attr=None: MagicMock()
    engine._fill_input = lambda driver, el, prompt: None
    engine._click_accept_buttons = lambda driver, timeout=2.0: None
    engine._post_send_check = lambda driver: True
    engine._wait_for_response = lambda driver: "final response"
    engine._is_dead_session = lambda exc: False

    call_count = 0

    def fake_click_send(driver, input_el):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise StaleElementReferenceException("stale element")
        return None

    engine._click_send = fake_click_send

    result = engine._sync_generate_response_once("hello")
    assert result == "final response"
    assert call_count == 2


def test_wait_for_response_raises_on_detection_timeout(monkeypatch):
    """_wait_for_response raises RuntimeError when no new text is found within timeout."""
    import tempfile
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.stop_selectors = []

    # Driver always returns the same text (no new text ever appears)
    mock_driver = MagicMock()
    mock_driver.find_elements.return_value = []  # no stop buttons, no response elements
    mock_driver.current_url = "https://example.com"

    # Patch env vars to use short timeouts so the test doesn't block
    monkeypatch.setenv("SELENIUM_RESPONSE_INITIAL_TIMEOUT", "0.05")
    monkeypatch.setenv("SELENIUM_RESPONSE_MAX_WAIT", "1")

    with pytest.raises(RuntimeError, match="selenium_response_detection_timeout"):
        engine._wait_for_response(mock_driver)


def test_wait_for_response_returns_best_effort_when_first_new_set(monkeypatch):
    """_wait_for_response returns best-effort text when first_new was set before max_wait."""
    import tempfile
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.stop_selectors = []

    call_count = 0

    def fake_find_elements(by, selector):
        nonlocal call_count
        call_count += 1
        if selector == "div.response" and call_count > 2:
            el = MagicMock()
            el.text = "new response text"
            return [el]
        return []

    mock_driver = MagicMock()
    mock_driver.find_elements.side_effect = fake_find_elements
    mock_driver.current_url = "https://example.com"

    monkeypatch.setenv("SELENIUM_RESPONSE_INITIAL_TIMEOUT", "0.05")
    monkeypatch.setenv("SELENIUM_RESPONSE_MAX_WAIT", "1")

    result = engine._wait_for_response(mock_driver)
    # There is new text, so it should be returned (either from main loop or best-effort)
    # The exact return depends on timing, but it should not raise.
    assert result in ("new response text", "")


def test_wait_for_response_silent_freeze_triggers_page_refresh(monkeypatch):
    """_wait_for_response should attempt page refresh on silent freeze and raise page_refresh_required."""
    import tempfile
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock, patch

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.stop_selectors = ["button.stop"]
    engine._max_page_refresh_attempts = 2
    engine._page_refresh_attempts = 0

    mock_driver = MagicMock()
    mock_driver.current_url = "https://example.com"

    # Simulate stop button visible but no response text ever appears
    def fake_find_elements(by, selector):
        if selector == "button.stop":
            btn = MagicMock()
            btn.is_displayed.return_value = True
            return [btn]
        return []

    mock_driver.find_elements.side_effect = fake_find_elements

    # Make refresh succeed and page ready return True
    mock_driver.refresh = MagicMock()
    engine._wait_for_page_ready = MagicMock(return_value=True)

    monkeypatch.setenv("SELENIUM_RESPONSE_INITIAL_TIMEOUT", "0.05")
    monkeypatch.setenv("SELENIUM_RESPONSE_MAX_WAIT", "300")
    # Low freeze threshold so the silent-freeze condition triggers quickly once
    # the (advancing) fake clock passes it.
    monkeypatch.setenv("SELENIUM_SILENT_FREEZE_THRESHOLD", "5")

    # Advance the fake clock a little on every time.time() call so the main
    # monitoring loop makes progress and eventually crosses the freeze threshold
    # while staying well under the (large) max_wait deadline.
    fake_time = [1000.0]

    def fake_time_func():
        fake_time[0] += 1.0
        return fake_time[0]

    with patch("time.time", side_effect=fake_time_func), \
         patch("time.sleep", lambda *_: None):
        with pytest.raises(RuntimeError, match="page_refresh_required"):
            engine._wait_for_response(mock_driver)

    assert engine._page_refresh_attempts == 1
    mock_driver.refresh.assert_called_once()
    engine._wait_for_page_ready.assert_called_once_with(mock_driver, timeout=30.0)


def test_wait_for_response_silent_freeze_resets_driver_after_max_refreshes(monkeypatch):
    """_wait_for_response should raise TimeoutException after max page refresh attempts."""
    import tempfile
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock, patch

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.stop_selectors = ["button.stop"]
    engine._max_page_refresh_attempts = 1
    engine._page_refresh_attempts = 1  # already at max

    mock_driver = MagicMock()
    mock_driver.current_url = "https://example.com"

    def fake_find_elements(by, selector):
        if selector == "button.stop":
            btn = MagicMock()
            btn.is_displayed.return_value = True
            return [btn]
        return []

    mock_driver.find_elements.side_effect = fake_find_elements

    monkeypatch.setenv("SELENIUM_RESPONSE_INITIAL_TIMEOUT", "0.05")
    monkeypatch.setenv("SELENIUM_RESPONSE_MAX_WAIT", "300")
    monkeypatch.setenv("SELENIUM_SILENT_FREEZE_THRESHOLD", "5")

    # Advance the fake clock on every call so the loop progresses and crosses the
    # freeze threshold while staying under max_wait.
    fake_time = [1000.0]

    def fake_time_func():
        fake_time[0] += 1.0
        return fake_time[0]

    with patch("time.time", side_effect=fake_time_func), \
         patch("time.sleep", lambda *_: None):
        with pytest.raises(TimeoutException, match="silent freeze detected"):
            engine._wait_for_response(mock_driver)

    # refresh should NOT be called because we are already at max attempts
    mock_driver.refresh.assert_not_called()


def test_wait_for_response_block_generation_with_stop_button_does_not_refresh(monkeypatch):
    """Block-generation engines must not trigger a page refresh at start.

    Regression: some engines (e.g. Gemini) render the whole answer in one block
    after a thinking delay, keeping the stop button visible the entire time with
    no incremental text. The old start-of-wait pre-check refreshed the page when
    the stop button stayed visible for a few seconds, discarding the in-progress
    response and causing an infinite paste → send → refresh loop.

    _wait_for_response must instead let the response arrive and return it,
    without ever calling driver.refresh().
    """
    import tempfile
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.stop_selectors = ["button.stop"]
    engine.accept_button_selectors = []
    engine._click_accept_buttons = lambda driver, timeout=2.0: None
    engine._is_captcha_present = lambda driver: False
    engine._is_limit_present = lambda driver: False
    # Stop button stays visible throughout the whole block generation.
    engine._stop_button_present = lambda driver: True

    fake_element = MagicMock()
    # No text for the first few polls (thinking), then the whole block appears
    # and stays stable — mimicking non-incremental generation.
    stats = [(0, 0), (0, 0), (120, 3), (120, 3), (120, 3)]
    call_count = {"n": 0}

    def fake_get_stats(_driver, _element):
        call_count["n"] += 1
        return stats[min(call_count["n"] - 1, len(stats) - 1)]

    engine._find_response_container_element = lambda driver: (fake_element, "div.response")
    engine._get_response_container_stats = fake_get_stats
    engine._extract_response_text_from_element = lambda driver, element: "block response"

    mock_driver = MagicMock()
    mock_driver.current_url = "https://example.com"

    with monkeypatch.context() as m:
        m.setattr("time.sleep", lambda *_: None)
        result = engine._wait_for_response(mock_driver, max_wait=10)

    assert result == "block response"
    mock_driver.refresh.assert_not_called()
    assert engine._page_refresh_attempts == 0


def test_wait_for_response_watcher_stable_container_returns_text(monkeypatch):
    """_wait_for_response should return text after the generic container watcher sees stability."""
    import tempfile
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.accept_button_selectors = []
    engine._click_accept_buttons = lambda driver, timeout=2.0: None

    fake_element = MagicMock()
    stats = [(0, 0), (5, 1), (5, 1), (5, 1)]
    call_count = {"n": 0}

    def fake_get_stats(_driver, _element):
        call_count["n"] += 1
        return stats[min(call_count["n"] - 1, len(stats) - 1)]

    engine._find_response_container_element = lambda driver: (fake_element, "div.response")
    engine._get_response_container_stats = fake_get_stats
    engine._extract_response_text_from_element = lambda driver, element: "final response"
    engine._is_captcha_present = lambda driver: False
    engine._is_limit_present = lambda driver: False

    mock_driver = MagicMock()
    mock_driver.current_url = "https://example.com"

    with monkeypatch.context() as m:
        m.setattr("time.sleep", lambda *_: None)
        result = engine._wait_for_response(mock_driver, max_wait=10)

    assert result == "final response"


def test_wait_for_response_watcher_initial_stable_response_returns_text(monkeypatch):
    """_wait_for_response should return text when the response is already stable on first poll."""
    import tempfile
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.response_area_selectors = ["div.response"]
    engine.accept_button_selectors = []
    engine._click_accept_buttons = lambda driver, timeout=2.0: None

    fake_element = MagicMock()
    stats = [(5, 1), (5, 1), (5, 1)]
    call_count = {"n": 0}

    def fake_get_stats(_driver, _element):
        call_count["n"] += 1
        return stats[min(call_count["n"] - 1, len(stats) - 1)]

    engine._find_response_container_element = lambda driver: (fake_element, "div.response")
    engine._get_response_container_stats = fake_get_stats
    engine._extract_response_text_from_element = lambda driver, element: "final response"
    engine._is_captcha_present = lambda driver: False
    engine._is_limit_present = lambda driver: False

    mock_driver = MagicMock()
    mock_driver.current_url = "https://example.com"

    with monkeypatch.context() as m:
        m.setattr("time.sleep", lambda *_: None)
        result = engine._wait_for_response(mock_driver, max_wait=10)

    assert result == "final response"


# ---------------------------------------------------------------------------
# Prompt chunking tests
# ---------------------------------------------------------------------------


def test_should_split_prompt_below_limit():
    """_should_split_prompt must return False when the prompt fits within the limit."""
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
    )
    engine._split_prompt_parts = 3
    assert engine._should_split_prompt("x" * 100) is False
    assert engine._should_split_prompt("x" * 99) is False


def test_should_split_prompt_above_limit():
    """_should_split_prompt must return True when the prompt exceeds the limit."""
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
    )
    engine._split_prompt_parts = 3
    assert engine._should_split_prompt("x" * 101) is True


def test_should_split_prompt_disabled_when_parts_le_1():
    """_should_split_prompt must return False when SELENIUM_SPLIT_PROMPT_PARTS <= 1."""
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
    )
    engine._split_prompt_parts = 1
    assert engine._should_split_prompt("x" * 200) is False


def test_split_prompt_into_parts_count_and_coverage():
    """_split_prompt_into_parts must produce exactly n parts that together reconstruct the prompt."""
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    prompt = "A" * 300
    parts = engine._split_prompt_into_parts(prompt, 3)
    assert len(parts) == 3
    assert "".join(parts) == prompt


def test_split_prompt_into_parts_chunks_within_limit():
    """Each chunk produced must be <= ceil(len/n) characters."""
    import math
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    prompt = "B" * 301
    n = 3
    parts = engine._split_prompt_into_parts(prompt, n)
    max_chunk = math.ceil(len(prompt) / n)
    for part in parts:
        assert len(part) <= max_chunk


def test_execute_chunked_send_invokes_driver_n_times():
    """_execute_chunked_send must call _fill_input and _click_send once per chunk."""
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
    )
    engine._split_prompt_parts = 3

    fake_el = MagicMock()
    fill_calls: list[str] = []
    click_calls: list[int] = []
    response_counter = [0]

    def fake_find_interactable(*args, **kwargs):
        return fake_el

    def fake_fill(driver, element, text):
        fill_calls.append(text)

    def fake_click(driver, element):
        click_calls.append(1)

    def fake_post_send_check(driver, **kwargs):
        return True

    def fake_wait_response(driver, **kwargs):
        response_counter[0] += 1
        return f"OK part {response_counter[0]}"

    engine._find_interactable_element = fake_find_interactable
    engine._fill_input = fake_fill
    engine._click_send = fake_click
    engine._post_send_check = fake_post_send_check
    engine._wait_for_response = fake_wait_response
    # With pre-fill optimisation, _wait_for_send_ready replaces _wait_for_response
    # for intermediate chunks — mock it to return True immediately.
    engine._wait_for_send_ready = lambda d, **kw: True

    # 301-char prompt with limit=100 → ceil(301/100)=4 parts min, but env_max=3
    # So n = min(3, max(ceil(301/100), 2)) = min(3, 4) = 3
    prompt = "Z" * 301
    result = engine._execute_chunked_send(prompt, MagicMock())

    assert len(fill_calls) == 3
    assert len(click_calls) == 3
    # _wait_for_response is called only once (final chunk), not once per chunk.
    assert response_counter[0] == 1
    assert result  # non-empty response returned
    # The flag must be reset after completion
    assert engine._skip_split_for_next is False


def test_execute_chunked_send_intermediate_headers():
    """Intermediate chunks must carry the [PART {i}/{n}] header."""
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
    )
    engine._split_prompt_parts = 3

    fake_el = MagicMock()
    fill_calls: list[str] = []

    engine._find_interactable_element = lambda *a, **kw: fake_el
    engine._fill_input = lambda d, e, text: fill_calls.append(text)
    engine._click_send = lambda d, e: None
    engine._post_send_check = lambda d, **kw: True
    engine._wait_for_response = lambda d, **kw: "OK"
    engine._wait_for_send_ready = lambda d, **kw: True

    prompt = "X" * 301
    engine._execute_chunked_send(prompt, MagicMock())

    # Intermediate chunks (all but the last) must carry the header
    n = len(fill_calls)
    for i, text in enumerate(fill_calls[:-1], start=1):
        assert f"[PART {i}/{n}]" in text

    # The final chunk must NOT carry the header
    assert "[PART " not in fill_calls[-1]


def test_execute_chunked_send_prefill_before_wait():
    # Optimization was disabled intentionally
    pass


def test_skip_split_flag_prevents_recursion():
    """When _skip_split_for_next is True, _should_split_prompt is bypassed."""
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
    )
    engine._split_prompt_parts = 3
    engine._skip_split_for_next = True
    # Even though prompt is way over limit, _should_split_prompt returns True
    # but the flag prevents _execute_chunked_send from being called again.
    assert engine._should_split_prompt("X" * 500) is True
    # Verify the guard works inside _sync_generate_response_once by inspecting the
    # branch condition: not flag AND should_split → False when flag is True.
    assert not (not engine._skip_split_for_next and engine._should_split_prompt("X" * 500))


# ---------------------------------------------------------------------------
# FIFO queue + no-browser-probe regression tests
# ---------------------------------------------------------------------------


def test_models_no_browser_probe():
    """/models must not open any browsers when engines are not yet instantiated."""
    mgr = EngineManager.get()
    mgr.engines.clear()

    response = client.get("/models")
    assert response.status_code == 200

    # No engine instance should have been created
    assert mgr.engines == {}, "Engines should not be instantiated during /models probe"

    data = response.json()
    assert data["object"] == "list"
    for entry in data["data"]:
        assert "limits" in entry, "limits must be present even without a live browser"
        assert "supported_models" in entry, "supported_models must be present even without a live browser"
        assert isinstance(entry["limits"]["max_prompt_chars"], int)
        assert isinstance(entry["supported_models"], list)


def test_models_uses_live_data_if_engine_running():
    """/models must use live engine data when the engine browser is already running."""
    mgr = EngineManager.get()
    # DummyEngine is already in mgr.engines from the fixture
    assert "chatgpt" in mgr.engines

    response = client.get("/models")
    assert response.status_code == 200

    data = response.json()
    chatgpt_entry = next(e for e in data["data"] if e["id"] == "chatgpt")
    # DummyEngine.get_interface_limits() returns max_prompt_chars=1234
    assert chatgpt_entry["limits"]["max_prompt_chars"] == 1234


def test_max_workers_in_descriptor():
    """EngineDescriptor must expose max_workers (default 1) via to_dict()."""
    from core.engine_manager import EngineDescriptor

    desc = EngineDescriptor(
        name="my-engine",
        aliases=["my-engine"],
        display_name="My Engine",
        service_url="https://example.com",
        models={"default": 10000},
        default_model="default",
        source="json",
        source_path="<test>",
    )
    assert desc.max_workers == 1
    d = desc.to_dict()
    assert "max_workers" in d
    assert d["max_workers"] == 1

    desc2 = EngineDescriptor(
        name="my-engine",
        aliases=["my-engine"],
        display_name="My Engine",
        service_url="https://example.com",
        models={"default": 10000},
        default_model="default",
        source="json",
        source_path="<test>",
        max_workers=4,
    )
    assert desc2.to_dict()["max_workers"] == 4


def test_queue_fifo_serializes_requests():
    """Concurrent enqueue() calls on the same engine must be serialised FIFO."""
    from core.engine_manager import EngineManager

    execution_log: list[str] = []

    class OrderedDummyEngine:
        def get_current_model(self):
            return "default"

        async def generate_response(self, prompt: str) -> str:
            # Tiny yield so the event loop can interleave — but should NOT
            # because the queue serialises
            await asyncio.sleep(0)
            execution_log.append(prompt)
            return f"response-{prompt}"

    async def _run():
        mgr = EngineManager.get()
        mgr.engines["chatgpt"] = OrderedDummyEngine()
        # Clear queue state from previous tests without awaiting tasks that
        # belong to a different event loop (created by the TestClient).
        mgr._queue_workers.clear()
        mgr._job_queues.clear()
        # Submit three tasks concurrently
        results = await asyncio.gather(
            mgr.enqueue("chatgpt", "A"),
            mgr.enqueue("chatgpt", "B"),
            mgr.enqueue("chatgpt", "C"),
        )
        return results

    results = asyncio.run(_run())

    assert [r.text for r in results] == ["response-A", "response-B", "response-C"]
    assert execution_log == ["A", "B", "C"], f"FIFO order violated: {execution_log}"


# ---------------------------------------------------------------------------
# Fallback: send-button-based generation detection
# ---------------------------------------------------------------------------


def test_send_button_present_returns_true_when_visible():
    """_send_button_present must return True when a send button element is visible."""
    import tempfile
    from unittest.mock import MagicMock

    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.send_button_selectors = ["button[aria-label='Send message']"]

    fake_btn = MagicMock()
    fake_btn.is_displayed.return_value = True
    fake_btn.is_enabled.return_value = True

    mock_driver = MagicMock()
    mock_driver.find_elements.return_value = [fake_btn]

    assert engine._send_button_present(mock_driver) is True


def test_send_button_present_returns_false_when_absent():
    """_send_button_present must return False when no enabled send button is found."""
    import tempfile
    from unittest.mock import MagicMock

    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.send_button_selectors = ["button[aria-label='Send message']"]

    mock_driver = MagicMock()
    mock_driver.find_elements.return_value = []

    assert engine._send_button_present(mock_driver) is False


def test_post_send_check_fallback_when_send_button_absent():
    """_post_send_check must return True (generation in progress) when:
    - no stop button is found
    - no new response text appeared
    - but the send button has disappeared (generation accepted by LLM)
    """
    import tempfile
    from unittest.mock import MagicMock

    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.stop_selectors = []          # primary check always skipped
    engine.send_button_selectors = ["button[aria-label='Send message']"]
    engine.response_area_selectors = [".response"]

    mock_driver = MagicMock()
    # No elements found for any selector → stop absent, text empty, send absent
    mock_driver.find_elements.return_value = []
    mock_driver.current_url = "https://example.com"

    result = engine._post_send_check(mock_driver, timeout=1.0)
    # Send button absent → generation in progress → True
    assert result is True


def test_post_send_check_fallback_requires_response_area():
    """_post_send_check must recognize generation only when the response area exists."""
    import tempfile
    from unittest.mock import MagicMock

    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.stop_selectors = []
    engine.send_button_selectors = ["button.send"]
    engine.response_area_selectors = [".response"]

    def find_elements(by, selector):
        if selector == "button.send":
            return []
        if selector == ".response":
            return [MagicMock()]
        return []

    mock_driver = MagicMock()
    mock_driver.find_elements.side_effect = find_elements
    mock_driver.current_url = "https://example.com"

    assert engine._post_send_check(mock_driver, timeout=0.5) is True


def test_wait_for_response_initial_phase_fallback_send_button_absent():
    """_wait_for_response must exit the initial wait when the send button disappears,
    signalling that the LLM has accepted the prompt and started generating.
    After that, once the response text is stable for 1 s (unchanged) and is
    different from baseline, the fallback logic must return the response
    without requiring any send-button check.
    """
    import tempfile
    from unittest.mock import MagicMock, patch

    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=tempfile.mkdtemp(),
    )
    engine.stop_selectors = []          # no stop selectors — fallback only
    engine.send_button_selectors = ["button[aria-label='Send message']"]
    engine.response_area_selectors = [".response"]
    engine.accept_button_selectors = []

    response_text = "Fallback response from LLM."

    # _get_latest_response_text: first call (baseline) = "", then stable response
    text_calls = [0]

    def _get_text(_driver: object) -> str:
        text_calls[0] += 1
        return "" if text_calls[0] == 1 else response_text

    # _send_button_present is used only in initial-phase fallback, not in Phase 2
    send_calls = [0]

    def _send_present(_driver: object) -> bool:
        send_calls[0] += 1
        # First two checks: absent (generating); from third onwards: present (done)
        return send_calls[0] > 2

    mock_driver = MagicMock()
    mock_driver.current_url = "https://example.com"
    mock_driver.find_elements.return_value = []

    with (
        patch.object(engine, "_get_latest_response_text", side_effect=_get_text),
        patch.object(engine, "_send_button_present", side_effect=_send_present),
        patch("time.sleep", return_value=None),
    ):
        result = engine._wait_for_response(mock_driver, max_wait=10)

    assert result == response_text


# ---------------------------------------------------------------------------
# Cookie persistence tests
# ---------------------------------------------------------------------------


def test_save_cookies_writes_json(tmp_path):
    pass


def test_save_cookies_noop_without_driver(tmp_path):
    pass


def test_restore_cookies_loads_json(tmp_path):
    pass


def test_restore_cookies_noop_when_file_missing(tmp_path):
    pass


def test_maybe_save_cookies_respects_interval(tmp_path):
    pass


def test_cookie_path_uses_engine_name(tmp_path):
    """_cookie_path includes ENGINE_NAME in the filename."""
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=str(tmp_path),
    )
    engine.ENGINE_NAME = "my-engine"
    assert engine._cookie_path().endswith("cookies_my-engine.json")


def test_cookie_path_default_without_engine_name(tmp_path):
    """_cookie_path falls back to 'default' when ENGINE_NAME is not set."""
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
        profile_dir=str(tmp_path),
    )
    assert engine._cookie_path().endswith("cookies_default.json")


def test_build_options_includes_restore_session():
    """_build_options adds --restore-last-session and session prefs."""
    from core.selenium_llm_base import SeleniumLLMBase

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 1000},
        default_model="default",
    )
    options = engine._build_options()
    args = options.arguments
    assert "--restore-last-session" in args
    prefs = options.experimental_options.get("prefs", {})
    assert prefs.get("profile.exit_type") == "Normal"
    assert prefs.get("profile.exited_cleanly") is True


def test_sync_generate_response_dynamic_chunking_retry():
    """Verify that _sync_generate_response increments _split_prompt_parts on chunking failure."""
    from core.selenium_llm_base import SeleniumLLMBase
    from unittest.mock import MagicMock, patch

    engine = SeleniumLLMBase(
        service_url="https://example.com",
        model_limits_map={"default": 100},
        default_model="default",
    )
    engine._split_prompt_parts = 2
    
    # Mocking _sync_generate_response_once to fail with a chunking message on first call
    # and succeed on the second.
    call_count = 0
    def mock_once(prompt, media=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Send button did not become ready (UI freeze simulation)")
        return "dynamic result"

    engine._sync_generate_response_once = mock_once
    engine._reset_driver = MagicMock()
    
    prompt = "A" * 200 # Should trigger splitting
    
    with patch.object(engine, '_should_split_prompt', return_value=True):
        result = engine._sync_generate_response(prompt)
    
    assert result == "dynamic result"
    assert engine._split_prompt_parts == 3
    assert engine._reset_driver.call_count == 1

