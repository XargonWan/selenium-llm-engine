"""JsonEngine — a SeleniumLLMBase subclass driven entirely by a JSON config dict.

Any LLM web interface that follows the common pattern:
  1. Navigate to a URL
  2. Type in a prompt area
  3. Click a send button
  4. Wait for the response area to stabilise

can be fully described with a JSON file.  No Python code required.

Schema (all fields except ``name`` and ``service_url`` are optional):

.. code-block:: json

    {
        "name": "my-engine",
        "display_name": "My Engine (Example)",
        "aliases": ["my-engine", "myengine"],
        "service_url": "https://www.example-llm.com",
        "default_model": "default",
        "models": {
            "default": 32000,
            "unlogged": 10000
        },
        "selectors": {
            "prompt_area": ["div[contenteditable='true']", "textarea"],
            "send_button": ["button[type='submit']"],
            "response_area": ["div.markdown"],
            "stop":         ["button[aria-label*='Stop']"]
        },
        "login_detection": {
            "url_prefix": "https://www.example-llm.com",
            "url_deny_keywords": ["login", "auth", "signin"],
            "login_button_xpath": "//button[contains(normalize-space(.), 'Log in')]",
            "authenticated_css_selectors": [
                "div[data-testid='conversation-panel']"
            ]
        }
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from selenium.webdriver.common.by import By

from core.selenium_llm_base import SeleniumLLMBase

logger = logging.getLogger("json_engine")

# Required top-level keys in every engine JSON file.
_REQUIRED_FIELDS: tuple[str, ...] = ("name", "service_url")


def _load_json(source: Path | dict[str, Any]) -> dict[str, Any]:
    """Accept either a file path or an already-parsed dict."""
    if isinstance(source, dict):
        return source
    with source.open(encoding="utf-8") as fh:
        return json.load(fh)


def _validate(config: dict[str, Any], source: str) -> None:
    missing = [k for k in _REQUIRED_FIELDS if not config.get(k)]
    if missing:
        raise ValueError(
            f"Engine JSON '{source}' is missing required field(s): {missing}"
        )


class JsonEngine(SeleniumLLMBase):
    """Selenium engine whose entire configuration is read from a JSON dict/file.

    Parameters
    ----------
    source:
        Either a ``Path`` pointing to a ``.json`` file or a pre-parsed ``dict``.
    **kwargs:
        Forwarded verbatim to :class:`~core.selenium_llm_base.SeleniumLLMBase`
        (e.g. ``headless``, ``profile_dir``).
    """

    def __init__(
        self,
        source: Path | dict[str, Any],
        **kwargs: Any,
    ) -> None:
        config = _load_json(source)
        source_label = str(source) if isinstance(source, Path) else "<dict>"
        _validate(config, source_label)

        self._config = config

        # ------------------------------------------------------------------ meta
        self.ENGINE_NAME: str = config["name"]
        self.ENGINE_ALIASES: list[str] = config.get("aliases", [config["name"]])
        self.display_name: str = config.get("display_name", config["name"])

        # ------------------------------------------------------------------ base
        super().__init__(
            service_url=config["service_url"],
            model_limits_map=dict(config.get("models", {"default": 10000})),
            default_model=config.get("default_model", "default"),
            allow_unlogged=bool(config.get("allow_unlogged", False)),
            **kwargs,
        )

        # ------------------------------------------------------------------ selectors
        sel = config.get("selectors", {})

        if "prompt_area" in sel:
            self.prompt_area_selectors = list(sel["prompt_area"])
        if "send_button" in sel:
            self.send_button_selectors = list(sel["send_button"])
        if "accept_button" in sel:
            self.accept_button_selectors = list(sel["accept_button"])
        if "response_area" in sel:
            self.response_area_selectors = list(sel["response_area"])
        if "stop" in sel:
            self.stop_selectors = list(sel["stop"])
        if "limit" in sel:
            self.limit_selectors = list(sel["limit"])
        if "send_button_blacklist" in sel:
            self.send_button_blacklist = list(sel["send_button_blacklist"])
        self.base_account_selector = sel.get("base_account_selector")

        # ------------------------------------------------------------------ login rules
        self._login_cfg: dict[str, Any] = config.get("login_detection", {})

        # ------------------------------------------------------------------ response timing
        # Optional JSON key "response_max_wait" (seconds) overrides the built-in
        # 120-second stability-loop timeout in _wait_for_response.
        # Useful for slow/thinking models (e.g. Gemini 2.5 Flash) that can take
        # several minutes to produce a full response.
        if "response_max_wait" in config:
            self._response_max_wait = int(config["response_max_wait"])
        if "use_baseline_comparison" in config:
            self._use_baseline_comparison = bool(config["use_baseline_comparison"])

        # ------------------------------------------------------------------ media support
        self.media_config: dict[str, Any] = config.get("media_support", {})
        self.paid_account_selector = self.media_config.get("paid_account_selector")
        if self.base_account_selector is None:
            self.base_account_selector = self.media_config.get("base_account_selector")

    # ---------------------------------------------------------------------- login

    def _ensure_logged_in(self, driver: Any) -> bool:  # type: ignore[override]
        """Generic login detection driven by ``login_detection`` config block.

        Detection steps (all optional — steps are skipped if the corresponding
        config key is absent):

        1. Navigate to ``url_prefix`` when the current URL doesn't start with it.
        2. Any of the ``url_deny_keywords`` in the current URL → **not logged in**.
        3. Any element matched by ``login_button_xpath`` is visible → **not logged in**.
        4. Any element matched by one of ``authenticated_css_selectors`` → **logged in**.
        5. Fallback → **logged in** (assume OK when no evidence of being locked out).
        """
        cfg = self._login_cfg
        try:
            current_url = (driver.current_url or "").lower()

            # Step 0 — authenticated element present on the current page
            auth_selectors: list[str] = cfg.get("authenticated_css_selectors", [])
            for css in auth_selectors:
                try:
                    els = driver.find_elements(By.CSS_SELECTOR, css)
                    if els and any(_safe_displayed(e) for e in els):
                        return True
                except Exception:
                    pass

            url_prefix: str = cfg.get("url_prefix", self.service_url)
            if url_prefix and not current_url.startswith(url_prefix):
                driver.get(url_prefix)
                current_url = (driver.current_url or "").lower()

            # Step 2 — deny keywords in URL
            deny_keywords: list[str] = cfg.get("url_deny_keywords", [])
            if any(kw in current_url for kw in deny_keywords):
                return False

            # Step 3 — visible login button
            login_xpath: str = cfg.get("login_button_xpath", "")
            if login_xpath:
                try:
                    buttons = driver.find_elements(By.XPATH, login_xpath)
                    if any(b.is_displayed() for b in buttons if _safe_displayed(b)):
                        return False
                except Exception:
                    pass

            # Step 4 — authenticated element present after navigation
            for css in auth_selectors:
                try:
                    els = driver.find_elements(By.CSS_SELECTOR, css)
                    if els and any(_safe_displayed(e) for e in els):
                        return True
                except Exception:
                    pass

            # Step 5 — fallback
            return True

        except Exception as exc:
            logger.warning(
                f"[json_engine:{self.ENGINE_NAME}] login check failed: {exc}"
            )
            return False


def _safe_displayed(element: Any) -> bool:
    """Return False instead of raising when ``is_displayed()`` fails."""
    try:
        return bool(element.is_displayed())
    except Exception:
        return False
