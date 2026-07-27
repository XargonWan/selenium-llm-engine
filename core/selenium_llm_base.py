import asyncio
import concurrent.futures
import glob
import logging
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, Optional, cast
from urllib.parse import urlparse

import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from core.agent_protocol import AGENT_TAIL_MARKER
from core import debug_mode

logger = logging.getLogger("selenium_llm_base")

# Matches a bare error line that carries a numeric failure code, e.g.
# "Something went wrong (1076)", "Error 1076" or a lone parenthesised
# "(1076)". Requires either an explicit error keyword next to the code or the
# code to be parenthesised on its own, so a normal short reply ending in a
# number (e.g. "The answer is 1200") is not misclassified.
_NUMERIC_ERROR_CODE_RE = re.compile(
    r"(?:\berror\b\s*#?\s*\d{3,5})|(?:^\(\d{3,5}\)$)",
    re.IGNORECASE,
)

# Global lock that serialises Chrome initialisation across all engine instances.
# Prevents concurrent _cleanup_chromium_remnants() calls from killing each
# other's browsers when multiple engines are started simultaneously.
_CHROMIUM_INIT_LOCK = threading.Lock()

# Shared Chrome driver instance — all engine instances reuse the same browser
# to avoid profile-dir lock conflicts and preserve login sessions across engines.
_shared_driver: Optional[Any] = None


def shutdown_shared_driver() -> None:
    """Quit the shared Chrome driver and clean up all Chromium processes.

    Called once by :meth:`EngineManager.stop_all` during application shutdown.
    """
    global _shared_driver
    with _CHROMIUM_INIT_LOCK:
        drv = _shared_driver
        _shared_driver = None
    if drv is not None:
        # Use a timeout so we don't hang on a dead ChromeDriver process.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(drv.quit)
            try:
                fut.result(timeout=5)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "[selenium] shared driver quit() timed out after 5s"
                )
            except Exception as exc:
                logger.warning("[selenium] shared driver quit error: %s", exc)
    # Kill any remaining Chromium processes and remove lock files.
    profile_dir = os.getenv(
        "CHROMIUM_PROFILE_DIR", "/config/.config/chromium-synth"
    )
    patterns = [profile_dir] if profile_dir else [
        "chromium", "chrome", "chromedriver", "undetected_chromedriver",
    ]
    for pattern in patterns:
        try:
            subprocess.run(
                ["pkill", "-15", "-f", pattern],
                check=False, capture_output=True, timeout=5,
            )
        except Exception:
            pass
    time.sleep(3)
    for pattern in patterns:
        try:
            subprocess.run(
                ["pkill", "-9", "-f", pattern],
                check=False, capture_output=True, timeout=5,
            )
        except Exception:
            pass
    logger.info("[selenium] Shared driver shut down")


def force_kill_session() -> None:
    """Immediately SIGKILL the shared browser session without trying ``quit()``.

    This is the "nuclear" option for when the browser is completely frozen
    and ``driver.quit()`` would hang.  It:

    1. Nullifies ``_shared_driver`` under the init lock.
    2. Sends SIGKILL to all Chromium / ChromeDriver processes.
    3. Cleans up lock files from the profile directory.

    Engine instances remain in memory but their ``driver`` reference is
    stale — the next request will automatically re-initialise the browser.
    """
    global _shared_driver
    logger.warning("[selenium] force_kill_session invoked — SIGKILL mode")
    with _CHROMIUM_INIT_LOCK:
        _shared_driver = None

    profile_dir = os.getenv(
        "CHROMIUM_PROFILE_DIR", "/config/.config/chromium-synth"
    )
    patterns = [profile_dir] if profile_dir else [
        "chromium", "chrome", "chromedriver", "undetected_chromedriver",
    ]
    for pattern in patterns:
        try:
            subprocess.run(
                ["pkill", "-9", "-f", pattern],
                check=False, capture_output=True, timeout=5,
            )
        except Exception:
            pass

    # Remove lock files that would prevent a clean restart
    if profile_dir:
        lock_patterns = [
            os.path.join(profile_dir, "SingletonLock"),
            os.path.join(profile_dir, "SingletonCookie"),
            os.path.join(profile_dir, "SingletonSocket"),
        ]
        for lock in lock_patterns:
            try:
                if os.path.exists(lock):
                    os.remove(lock)
            except Exception:
                pass

    logger.info("[selenium] force_kill_session complete — browser processes killed")


class SeleniumLLMBase:
    def __init__(
        self,
        service_url: str,
        model_limits_map: Dict[str, int],
        default_model: str,
        headless: Optional[bool] = None,
        profile_dir: Optional[str] = None,
        allow_unlogged: bool = False,
    ):
        self.service_url = service_url
        self.model_limits_map = model_limits_map
        self.default_model = default_model
        self.allow_unlogged = allow_unlogged
        self.driver = None
        self.media_config: dict[str, Any] = {}
        self.paid_account_selector: Optional[str] = None

        if headless is None:
            env_headless = os.getenv("CHROMIUM_HEADLESS", "0")
            try:
                self.headless = bool(int(env_headless))
            except Exception:
                self.headless = False
        else:
            self.headless = headless
        self._initialized = False

        self.profile_dir = profile_dir or os.getenv(
            "CHROMIUM_PROFILE_DIR", "/config/.config/chromium-synth"
        )
        os.makedirs(self.profile_dir, exist_ok=True)
        logger.info(f"[selenium] Chrome profile_dir={self.profile_dir}")

        self._last_login_state: Optional[bool] = None

        # Selector lists used by _sync_generate_response — override in subclasses.
        self.prompt_area_selectors: list[str] = [
            "textarea",
            "div[contenteditable='true']",
        ]
        self.send_button_selectors: list[str] = [
            "button[type='submit']",
            "button[aria-label*='Send']",
        ]
        self.response_area_selectors: list[str] = [
            ".assistant-message",
            "div.markdown",
        ]
        self.stop_selectors: list[str] = [
            "button[aria-label*='Stop']",
            "[data-testid='stop-button']",
        ]
        self.accept_button_selectors: list[str] = []
        self.limit_selectors: list[str] = []
        # CSS selectors that match a transient error banner/toast/snackbar the web
        # UI shows when a generation fails (e.g. "Something went wrong (1076)").
        # Such errors are typically rendered outside the response area, so the
        # normal response-text detection never sees them and the request would
        # otherwise stall until timeout. Populated from the JSON "error_indicators"
        # key; empty by default to keep the core engine-agnostic.
        self.error_indicator_selectors: list[str] = []
        # CSS selectors whose matching elements must never be clicked as send button
        self.send_button_blacklist: list[str] = []

        # Cloudflare CAPTCHA challenge detectors
        self.captcha_challenge_selectors: list[str] = [
            "iframe#cf-chl-widget-ezspn",
            "iframe[src*='challenges.cloudflare.com/cdn-cgi/challenge-platform']",
        ]

        # Selector cache: remember the last working selector to try it first
        self._cached_prompt_selector: Optional[str] = None
        self._cached_send_selector: Optional[str] = None
        # Set by _post_send_check when a genuine cross-host redirect is seen so
        # the retry loop can fail fast on deterministic auth redirects.
        self._last_send_offsite_redirect: bool = False

        # Prompt chunking: split prompts that exceed the model char limit.
        # This is a safety cap — the actual number of parts is computed
        # dynamically from prompt length / model limit and never exceeds this.
        # Default 10 is generous enough to handle very large prompts without
        # creating too many tiny chunks.
        self._split_prompt_parts: int = max(2, int(os.getenv("SELENIUM_SPLIT_PROMPT_PARTS", "10")))
        self._skip_split_for_next: bool = False

        # Per-engine response timeout override (seconds).  Set by JsonEngine from the
        # JSON config key "response_max_wait".  None means use the built-in default.
        self._response_max_wait: int | None = None

        # Set to True when _wait_for_response observes the stop-button during the
        # last generation attempt.  Used by _sync_generate_response to decide whether
        # to reset the driver on a detection-timeout: if the model was actively
        # generating (slow thinking model) we skip the reset; if the stop-button was
        # never seen (stuck/dead session) we reset as before.
        self._generation_was_active: bool = False

        # If the uc.Chrome session dies after initialization, prefer the
        # standard webdriver.Chrome fallback on the next driver init attempt.
        self._prefer_webdriver_fallback: bool = False

        # Set per-request by generate_response(): when True the request is part
        # of the agentic (tool-calling) lane and short JSON replies must not be
        # discarded by the page-state junk filter.
        self._agent_mode: bool = False

        # Per-engine total timeout override (seconds). Set by JsonEngine from
        # the JSON config key "total_timeout". None means use the computed default.
        self._total_timeout: int | None = None

        # Some engines (Gemini) work more reliably when response detection uses
        # stable text instead of comparing against prior baseline text.
        self._use_baseline_comparison: bool = True

        # Per-engine "silent freeze" threshold (seconds): how long the stop button
        # may stay visible with no response-text activity before the page is
        # considered stuck.  Engines that render the whole answer in one block
        # (no incremental streaming, e.g. Gemini) need a larger value so that the
        # initial "thinking" delay isn't mistaken for a freeze.  None means use
        # the SELENIUM_SILENT_FREEZE_THRESHOLD env default.
        self._silent_freeze_threshold: float | None = None

        # Silent freeze recovery: track page refresh attempts before driver reset
        self._page_refresh_attempts: int = 0
        self._max_page_refresh_attempts: int = int(os.getenv("SELENIUM_MAX_PAGE_REFRESH", "2"))

        # Timestamp of the last cookie save — used to rate-limit periodic saves.
        self._last_cookie_save: float = 0.0
        # Minimum interval (seconds) between periodic cookie saves.
        self._cookie_save_interval: int = int(os.getenv("COOKIE_SAVE_INTERVAL", "300"))

        # Whether this instance has already restored its cookies into the
        # shared driver.  Reset on stop/reset so a fresh restore happens
        # on the next use.
        self._cookies_restored: bool = False

        # Optional prefix prepended to the prompt when media items are present.
        # Useful to prevent engines that run inside an existing chat context
        # (e.g. Gemini Web with SyntH persona) from responding in their
        # trained JSON/action schema instead of plain descriptive text.
        self._vision_prompt_prefix: str = ""

        # Optional prefix prepended to the prompt when NO media items are present.
        # Useful to instruct the LLM to respond inline (e.g. avoid canvas/document
        # mode) and to prevent the engine from entering a stuck state that shows
        # "Something went wrong" errors.
        # Can be overridden per-engine via the "inline_response_prefix" JSON key.
        self._inline_response_prefix: str = (
            "IMPORTANT: Always respond inline, do NOT use canvas, documents, or "
            "any separate UI mode. Write the full response directly in this chat "
            "as plain text. Do NOT activate canvas, artifacts, or separate "
            "documents. The response must be complete and self-contained within "
            "this chat window.\n\n"
        )

    def get_supported_models(self) -> list[str]:
        return list(self.model_limits_map.keys())

    def get_current_model(self) -> str:
        # Return 'unlogged' when the engine is not logged in, supports it, and has this model
        if (
            not self.is_user_logged_in()
            and self.allow_unlogged
            and "unlogged" in self.model_limits_map
        ):
            return "unlogged"
        return self.default_model

    def _get_model_limit(self, model_name: str) -> int:
        model_name = model_name.lower().strip()
        if model_name in self.model_limits_map:
            return self.model_limits_map[model_name]
        if "default" in self.model_limits_map:
            return self.model_limits_map["default"]
        return 10000

    def get_interface_limits(self) -> dict[str, Any]:
        return {
            "max_prompt_chars": self._get_model_limit(self.get_current_model()),
            "model_name": self.get_current_model(),
        }

    def _locate_chromium_binary(self) -> Optional[str]:
        possible = [
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/opt/google/chrome/chrome",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
        ]
        for path in possible:
            if os.path.exists(path):
                logger.info(f"[selenium] Found Chromium at: {path}")
                return path
        logger.warning("[selenium] Chromium binary not found in common locations")
        return None

    def _locate_chromedriver_binary(self) -> Optional[str]:
        candidates = [
            "/usr/bin/chromedriver",
            "/usr/bin/chromium-driver",
            "/usr/local/bin/chromedriver",
            "/usr/local/bin/chromium-driver",
            "/opt/chromedriver/chromedriver",
            shutil.which("chromedriver") or "",
            shutil.which("chromium-driver") or "",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                logger.info(f"[selenium] Found ChromeDriver binary: {path}")
                return path
        try:
            from webdriver_manager.chrome import ChromeDriverManager

            logger.warning(
                "[selenium] ChromeDriver not found, attempting webdriver-manager install"
            )
            path = ChromeDriverManager().install()
            logger.info(
                f"[selenium] webdriver-manager installed ChromeDriver at {path}"
            )
            return path
        except Exception as e:
            logger.warning(
                f"[selenium] webdriver-manager ChromeDriver install failed: {e}"
            )
        logger.warning("[selenium] ChromeDriver binary not found")
        return None

    def _writable_chromedriver_copy(self, chromedriver_path: str) -> str:
        """Return a chromedriver path that the current user can patch in place.

        undetected_chromedriver patches the driver binary in place to strip
        automation fingerprints. When chromedriver is installed from a system
        package it lives in a root-owned location (e.g. /usr/bin/chromedriver)
        that the runtime user (abc) cannot write to, so uc.Chrome fails with
        ``[Errno 13] Permission denied``. Copy it into the writable profile
        directory and hand uc that copy instead.
        """
        try:
            if os.access(chromedriver_path, os.W_OK):
                return chromedriver_path
        except OSError:
            pass
        try:
            dest_dir = os.path.join(self.profile_dir, "chromedriver")
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, "chromedriver")
            if not os.path.exists(dest) or os.path.getmtime(dest) < os.path.getmtime(
                chromedriver_path
            ):
                shutil.copy2(chromedriver_path, dest)
            os.chmod(dest, 0o755)
            logger.info(
                "[selenium] Using writable chromedriver copy: %s (from %s)",
                dest,
                chromedriver_path,
            )
            return dest
        except Exception as e:
            logger.warning(
                "[selenium] Could not create writable chromedriver copy from %s: %s",
                chromedriver_path,
                e,
            )
            return chromedriver_path

    def _get_chromium_major_version(
        self, chromium_binary: Optional[str] = None
    ) -> Optional[int]:
        binary = chromium_binary or self._locate_chromium_binary()
        if not binary:
            return None
        try:
            result = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # output: "Chromium 130.0.6723.58 ..." or "Google Chrome 130.0.6723.58 ..."
            version_str = result.stdout.strip()
            for part in version_str.split():
                if "." in part:
                    try:
                        major = int(part.split(".")[0])
                        if major > 50:  # sanity check
                            logger.info(f"[selenium] Chromium major version: {major}")
                            return major
                    except ValueError:
                        continue
        except Exception as e:
            logger.warning(f"[selenium] Could not get Chromium version: {e}")
        return None

    def _build_options(self) -> Options:
        """Build Chrome options matching SyntH's working configuration.

        Uses standard selenium Options (not uc.ChromeOptions) — this is what
        the working Synthetic Heart implementation uses.
        """
        options = Options()

        essential_args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-extensions",
            "--disable-plugins",
            "--disable-web-security",
            "--allow-running-insecure-content",
            # Chrome only honours a single --disable-features flag, so all
            # disabled features must be listed together (comma-separated).
            # CalculateNativeWinOcclusion is added for renderer stability: with
            # multiple heavy SPA tabs sharing a single browser process, an
            # out-of-memory renderer crash ("Unable to receive message from
            # renderer" / "invalid session id") can kill the whole shared
            # session. These flags reduce per-renderer memory pressure and stop
            # Chrome from discarding backgrounded tabs.
            "--disable-features=VizDisplayCompositor,CalculateNativeWinOcclusion",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--restore-last-session",
            "--disable-hang-monitor",
            "--disable-ipc-flooding-protection",
            "--disable-back-forward-cache",
            "--js-flags=--max-old-space-size=512",
        ]
        if self.headless:
            essential_args.append("--headless=new")

        for arg in essential_args:
            options.add_argument(arg)

        options.add_argument("--window-size=1280,900")
        options.add_argument(f"--user-data-dir={self.profile_dir}")

        # Tell Chrome the previous session ended cleanly so it restores
        # session cookies and skips the "restore pages?" dialog.
        options.add_experimental_option("prefs", {
            "profile.exit_type": "Normal",
            "profile.exited_cleanly": True,
        })
        return options

    def _init_driver(self) -> Any:
        """Initialize or reuse the shared Chrome driver.

        All engine instances share a single Chrome process to prevent
        profile-dir lock conflicts and to stop ``_cleanup_chromium_remnants``
        from killing another engine's browser (and its login session).
        """
        global _shared_driver

        if self.driver is not None and self.driver is _shared_driver:  # fast path — no lock needed
            return self.driver

        with _CHROMIUM_INIT_LOCK:
            # Re-check after acquiring: another thread may have initialised already.
            if self.driver is not None:
                return self.driver

            # ---- reuse existing shared driver if alive ----
            if _shared_driver is not None:
                try:
                    _ = _shared_driver.current_url  # ping — throws if dead
                    self.driver = _shared_driver
                    self._initialized = True
                    logger.info("[selenium] Reusing shared Chrome driver")
                    return self.driver
                except Exception:
                    logger.warning(
                        "[selenium] Shared driver is dead, creating a new one"
                    )
                    _shared_driver = None

            # ---- create a fresh driver ----
            logger.info("[selenium] Initializing Chrome driver...")
            # Light cleanup - only kill processes if they exist
            self._cleanup_chromium_remnants(light=True)

            chromium_binary = self._locate_chromium_binary() or "/usr/bin/chromium"
            chromedriver_path = (
                self._locate_chromedriver_binary() or "/usr/bin/chromedriver"
            )

            # Get Chromium major version for uc compatibility
            chromium_major = self._get_chromium_major_version(chromium_binary)

            max_retries = 3
            self.driver = None
            last_err: Optional[Exception] = None

            # Auto-detect: undetected_chromedriver 3.5.5 does not support
            # Chromium >= 149.  Skip straight to the browser's native driver.
            if chromium_major is not None and chromium_major >= 149:
                logger.info(
                    "[selenium] Chromium %d detected — skipping uc.Chrome "
                    "(unsupported by undetected_chromedriver 3.5.5), "
                    "using webdriver.Chrome directly",
                    chromium_major,
                )
                self._prefer_webdriver_fallback = True

            for attempt in range(max_retries if not self._prefer_webdriver_fallback else 0):
                options = self._build_options()
                options.binary_location = chromium_binary
                try:
                    if self._prefer_webdriver_fallback:
                        raise RuntimeError("Skipping uc.Chrome because webdriver fallback is preferred")

                    logger.info(
                        f"[selenium] Driver initialization attempt {attempt + 1}/{max_retries}"
                    )
                    # uc patches the driver binary in place, which requires a
                    # writable path. System chromedriver is root-owned, so hand
                    # uc a writable copy under the profile dir.
                    uc_driver_path = self._writable_chromedriver_copy(
                        chromedriver_path
                    )
                    uc_kwargs: dict[str, Any] = {
                        "options": options,
                        "service": Service(executable_path=uc_driver_path),
                        # Pin the system chromedriver so undetected_chromedriver
                        # does NOT auto-download a mismatched driver (e.g. it
                        # fetches 150 for a 148 browser). uc.Chrome ignores the
                        # Service executable_path but honours
                        # driver_executable_path, which takes precedence over
                        # its built-in auto-downloader.
                        "driver_executable_path": uc_driver_path,
                    }
                    if chromium_major is not None:
                        uc_kwargs["version_main"] = chromium_major
                    self.driver = uc.Chrome(**uc_kwargs)

                    # Clean up extra windows (SyntH pattern)
                    if len(self.driver.window_handles) > 1:
                        logger.info(
                            f"[selenium] Driver created with {len(self.driver.window_handles)} windows, cleaning up..."
                        )
                        for handle in self.driver.window_handles[1:]:
                            try:
                                self.driver.switch_to.window(handle)
                                self.driver.close()
                            except Exception:
                                pass
                        self.driver.switch_to.window(self.driver.window_handles[0])

                    logger.info(
                        f"[selenium] Driver created with {len(self.driver.window_handles)} window(s)"
                    )
                    break
                except Exception as err:
                    last_err = err
                    logger.warning(
                        f"[selenium] Attempt {attempt + 1}/{max_retries} failed: {err}"
                    )
                    self._cleanup_chromium_remnants()
                    if attempt < max_retries - 1:
                        time.sleep(1)

            if self.driver is None:
                # Fallback: standard webdriver (no anti-detection patching)
                if self._prefer_webdriver_fallback:
                    logger.info(
                        "[selenium] Using webdriver.Chrome fallback due to prior uc.Chrome instability"
                    )
                else:
                    logger.warning(
                        "[selenium] uc.Chrome failed after all retries, trying webdriver.Chrome fallback"
                    )
                try:
                    fallback_options = self._build_options()
                    fallback_options.binary_location = chromium_binary
                    self.driver = webdriver.Chrome(
                        service=Service(executable_path=chromedriver_path),
                        options=fallback_options,
                    )
                    logger.info("[selenium] webdriver.Chrome fallback succeeded")
                except Exception as fallback_err:
                    logger.error(
                        f"[selenium] webdriver.Chrome fallback also failed: {fallback_err}"
                    )
                    raise RuntimeError(
                        f"Driver initialization failed (uc: {last_err!r}, fallback: {fallback_err!r})"
                    ) from fallback_err

            self.driver.set_page_load_timeout(120)
            self.driver.set_script_timeout(120)
            self._initialized = True
            _shared_driver = self.driver
            logger.info("[selenium] Driver initialized successfully (shared)")
            return self.driver

    def _cleanup_chromium_remnants(self, light: bool = False) -> None:
        """Clean up Chromium processes and lock files.
        
        Args:
            light: If True, do minimal cleanup (skip sleep, faster kill).
        """
        try:
            logger.info("[selenium] Cleaning up Chromium remnants...")

            # Kill only processes that are using this engine's profile directory
            patterns: list[str] = []
            if self.profile_dir:
                patterns.append(self.profile_dir)
            else:
                patterns.extend(
                    [
                        "chromium",
                        "chrome",
                        "chromedriver",
                        "undetected_chromedriver",
                    ]
                )

            for pattern in patterns:
                try:
                    # Graceful shutdown: SIGTERM first so Chrome can flush cookies/profile to disk
                    subprocess.run(
                        ["pkill", "-15", "-f", pattern],
                        check=False,
                        capture_output=True,
                        timeout=2,
                    )
                except Exception:
                    pass

            # Give Chrome time to flush profile (cookies, session storage) before SIGKILL
            if not light:
                time.sleep(1)

            for pattern in patterns:
                try:
                    # SIGKILL fallback for any stragglers that ignored SIGTERM
                    subprocess.run(
                        ["pkill", "-9", "-f", pattern],
                        check=False,
                        capture_output=True,
                        timeout=2,
                    )
                except Exception:
                    pass

            # Brief wait after SIGKILL before removing lock files
            if not light:
                time.sleep(0.2)
            logger.info("[selenium] Chromium processes killed")

            # Clean up temp dir lock files
            temp_dir = tempfile.gettempdir()
            lock_patterns = [
                os.path.join(temp_dir, ".org.chromium.Chromium.*"),
                os.path.join(temp_dir, "selenium_*_profile", "SingletonLock"),
                os.path.join(temp_dir, "selenium_*_profile", "SingletonCookie"),
                os.path.join(
                    temp_dir, "selenium_*_profile", ".org.chromium.Chromium.*"
                ),
            ]
            for pattern in lock_patterns:
                for lock_file in glob.glob(pattern):
                    try:
                        os.remove(lock_file)
                        logger.info(f"[selenium] Removed lock file: {lock_file}")
                    except Exception:
                        pass

            # Clean up profile directory lock files
            if os.path.exists(self.profile_dir):
                for lock_pat in [
                    "SingletonLock",
                    "SingletonCookie",
                    ".org.chromium.Chromium.*",
                ]:
                    for lock_file in glob.glob(
                        os.path.join(self.profile_dir, lock_pat)
                    ):
                        try:
                            os.remove(lock_file)
                            logger.info(
                                f"[selenium] Removed profile lock file: {lock_file}"
                            )
                        except Exception:
                            pass

            if not light:
                time.sleep(0.2)
            logger.info("[selenium] Chromium cleanup completed")
        except Exception as e:
            logger.warning(f"[selenium] Error during Chromium cleanup: {e}")

    # ------------------------------------------------------------------ cookie persistence

    def _cookie_path(self) -> str:
        """Return the file path for persisted cookies of this engine."""
        engine_name = getattr(self, "ENGINE_NAME", "default")
        return os.path.join(self.profile_dir, f"cookies_{engine_name}.json")

    def _save_cookies(self) -> None:
        """Persist current browser cookies to a JSON file (atomic write)."""
        pass

    def _restore_cookies(self) -> None:
        """Load previously saved cookies into the browser session."""
        self._cookies_restored = True

    def _maybe_save_cookies(self) -> None:
        """Disabled."""
        pass

    # ------------------------------------------------------------------ readiness

    def _ensure_ready(self) -> None:
        global _shared_driver
        if self.driver is not None and self.driver is not _shared_driver:
            logger.warning("[selenium] Invalidating stale driver reference because shared driver was reset")
            self.driver = None
            self._initialized = False
            self._cookies_restored = False

        if not self._initialized or self.driver is None:
            self._init_driver()
        # Restore this engine's cookies once per session (per-domain).
        if not self._cookies_restored:
            self._restore_cookies()
            self._cookies_restored = True

    def _ensure_logged_in(self, driver) -> bool:
        # Implemented by subclasses.
        raise NotImplementedError()

    def is_user_logged_in(self) -> bool:
        # Avoid initializing a browser for a simple state check when not yet used.
        if not self._initialized or self.driver is None:
            if self._last_login_state is not None:
                return self._last_login_state
            return False

        try:
            logged = self._ensure_logged_in(self.driver)
            self._last_login_state = logged
            # Persist cookies every time we detect a logged-in state.
            if logged:
                self._maybe_save_cookies()
            return logged
        except Exception as e:
            logger.warning(f"Unable to determine login state: {e}")
            return False

    async def start_login_flow(self, timeout: int = 60) -> dict[str, Any]:
        """Open the service URL in the browser (non-blocking) and return login state."""

        def _sync_start_login() -> dict[str, Any]:
            self._ensure_ready()
            assert self.driver is not None, (
                "_ensure_ready() must have initialized driver"
            )
            drv = cast(webdriver.Chrome, self.driver)
            drv.get(self.service_url)
            time.sleep(2)
            logged = self.is_user_logged_in()
            state = "logged" if logged else "unlogged"
            return {"logged_in": logged, "login_state": state}

        try:
            return await asyncio.to_thread(_sync_start_login)
        except Exception as e:
            logger.error(f"start_login_flow error: {e}")
            return {"logged_in": False, "login_state": "unknown", "error": str(e)}

    async def check_login_state(self) -> dict[str, Any]:
        """Return current login state without navigating."""
        try:
            if not self._initialized or self.driver is None:
                logged = bool(self._last_login_state)
                state = "logged" if logged else "unlogged"
                return {"logged_in": logged, "login_state": state}

            logged = await asyncio.to_thread(self.is_user_logged_in)
            state = "logged" if logged else "unlogged"
            return {"logged_in": logged, "login_state": state}
        except Exception as e:
            logger.error(f"check_login_state error: {e}")
            return {"logged_in": False, "login_state": "unknown", "error": str(e)}

    async def generate_response(
        self, prompt: str, media: list[Any] | None = None, timeout: int | None = None,
        agent_mode: bool = False,
    ) -> str:
        """Send prompt and optional media to the LLM service and return the response text.

        All Selenium calls are executed in a thread pool via asyncio.to_thread
        so that blocking I/O never stalls the FastAPI event loop.

        Parameters
        ----------
        timeout:
            Maximum number of seconds to wait for the full generation cycle
            (navigation + input + response).  ``None`` uses the default of
            300 s (5 min), overridable via the ``SELENIUM_TOTAL_TIMEOUT``
            environment variable.
        agent_mode:
            When True the request is part of the agentic (tool-calling) lane.
            Structured JSON replies must NOT be discarded as page-state junk,
            so ``_looks_like_response_text_junk`` is relaxed for this request.
        """
        # Signal downstream helpers (junk filter) that JSON output is expected.
        self._agent_mode = bool(agent_mode)
        # Compute effective timeout: per-call > per-engine from JSON > env var > computed default
        if timeout is not None:
            total_timeout = int(timeout)
        elif self._total_timeout is not None:
            total_timeout = self._total_timeout
        elif os.getenv("SELENIUM_TOTAL_TIMEOUT"):
            total_timeout = int(os.getenv("SELENIUM_TOTAL_TIMEOUT"))
        else:
            effective_max_wait = self._response_max_wait or 120
            total_timeout = effective_max_wait + 180  # generous overhead

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._sync_generate_response, prompt, media),
                timeout=total_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                "[selenium] generate_response timed out after %ds — force-resetting driver",
                total_timeout,
            )
            # Force-reset in a separate thread to unblock the stuck thread.
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._force_reset_driver),
                    timeout=15,
                )
            except Exception as exc:
                logger.warning("[selenium] _force_reset_driver error: %s", exc)
            raise RuntimeError(
                f"selenium_response_detection_timeout: generate_response "
                f"exceeded total timeout of {total_timeout}s"
            )

        # Periodically persist cookies after a successful prompt so that login
        # state survives even if the container is killed abruptly.
        try:
            self._maybe_save_cookies()
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------ session health

    def _is_dead_session(self, exc: Exception) -> bool:
        """Return True if *exc* signals a crashed/dead chromedriver session."""
        msg = str(exc).lower()
        return any(
            marker in msg
            for marker in (
                "connection refused",
                "errno 111",
                "errno 113",
                "failed to establish a new connection",
                "max retries exceeded",
                "no such session",
                "invalid session id",
                "no such window",
                "target window already closed",
                "web view not found",
            )
        )

    def _is_redirect_stall(self, exc: Exception) -> bool:
        """Return True if *exc* signals a redirect-stall (prompt submitted but page navigated away)."""
        return "redirect-stall:" in str(exc).lower()

    def _is_offsite_redirect_stall(self, exc: Exception) -> bool:
        """Return True if *exc* signals an off-site redirect-stall.

        An off-site redirect (the page navigated to a different host, typically
        an authentication/login domain) is deterministic: retrying will not
        resolve it because the service requires the user to sign in. Such a
        stall must fail fast instead of consuming the whole retry budget.
        """
        return "redirect-stall (off-site)" in str(exc).lower()

    def _is_response_detection_timeout(self, exc: Exception) -> bool:
        """Return True if *exc* signals that the LLM response was not detected in time."""
        return "selenium_response_detection_timeout" in str(exc)

    def _is_page_refresh_required(self, exc: Exception) -> bool:
        """Return True if *exc* signals that a page refresh was performed due to silent freeze."""
        return "page_refresh_required" in str(exc)

    def _is_engine_error_response(self, text: str) -> bool:
        """Return True if *text* looks like an LLM web UI error rather than a valid response.

        Detects common error patterns displayed by LLM web interfaces (e.g.
        "Something went wrong", "Something went wrong (13)", etc.) so they
        can be treated as transient failures and retried.
        """
        if not text:
            return False
        stripped = text.strip()
        # Common error patterns from LLM web UIs
        error_patterns = [
            "something went wrong",
            "qualcosa è andato storto",
            "an error occurred",
            "si è verificato un errore",
            "internal server error",
            "errore interno del server",
            "request failed",
            "richiesta fallita",
            "service unavailable",
            "servizio non disponibile",
        ]
        lower = stripped.lower()
        for pattern in error_patterns:
            if pattern in lower:
                logger.warning(
                    "[selenium] Detected engine error response: %r",
                    stripped[:200],
                )
                return True
        # Numeric error codes shown by some web UIs on a bare error line, e.g.
        # "Something went wrong (1076)" or "Error 1076". Only treat a short line
        # as an error so a long, legitimate reply that merely mentions "error"
        # somewhere is not misclassified.
        if len(stripped) <= 80 and _NUMERIC_ERROR_CODE_RE.search(stripped):
            logger.warning(
                "[selenium] Detected numeric error-code response: %r",
                stripped[:200],
            )
            return True
        return False

    def _quit_driver_with_timeout(self, driver: Any, timeout: int = 5) -> None:
        """Attempt ``driver.quit()`` with a hard timeout.

        If the driver process is dead or unresponsive, ``quit()`` blocks
        indefinitely waiting for an HTTP reply from ChromeDriver.  This
        wrapper caps the wait so that callers can proceed to cleanup.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(driver.quit)
            try:
                fut.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "[selenium] driver.quit() did not finish within %ds", timeout
                )
            except Exception as exc:
                logger.warning("[selenium] driver.quit() error: %s", exc)

    def _kill_chromium_processes(self) -> None:
        """Immediately SIGKILL all Chromium processes (fast variant of cleanup)."""
        patterns: list[str] = []
        if self.profile_dir:
            patterns.append(self.profile_dir)
        else:
            patterns.extend(["chromium", "chrome", "chromedriver"])

        for pattern in patterns:
            try:
                subprocess.run(
                    ["pkill", "-9", "-f", pattern],
                    check=False, capture_output=True, timeout=5,
                )
            except Exception:
                pass
        logger.info("[selenium] Chromium processes force-killed")

    def _force_reset_driver(self) -> None:
        """Force-kill the driver without trying ``driver.quit()``.

        Used after a total-timeout when the Selenium thread is stuck.
        Skips the polite quit() call (which would also hang) and goes
        straight to SIGKILL.
        """
        global _shared_driver
        logger.warning("[selenium] Force-resetting driver (skipping quit)…")
        with _CHROMIUM_INIT_LOCK:
            self.driver = None
            _shared_driver = None
            self._initialized = False
            self._cookies_restored = False
            self._page_refresh_attempts = 0
        self._kill_chromium_processes()

    def _reset_driver(self) -> None:
        """Kill the existing (dead) driver and reset state so _ensure_ready re-inits."""
        global _shared_driver
        logger.warning("[selenium] Resetting dead driver session…")
        if self.driver is not None:
            self._quit_driver_with_timeout(self.driver, timeout=5)
            self.driver = None
        _shared_driver = None  # force all engines to re-create on next use
        self._initialized = False
        self._cookies_restored = False
        self._page_refresh_attempts = 0
        self._cleanup_chromium_remnants()

    def _is_captcha_present(self, driver: Any) -> bool:
        """Return True if the page includes a known Cloudflare captcha challenge widget."""
        for selector in self.captcha_challenge_selectors:
            try:
                els = driver.find_elements(By.CSS_SELECTOR, selector)
                if els:
                    logger.debug(f"[selenium] Captcha selector matched: {selector}")
                    return True
            except Exception:
                pass
        return False

    def _is_limit_present(self, driver: Any) -> bool:
        """Return True if the page contains a known usage-limit warning."""
        for selector in self.limit_selectors:
            try:
                els = driver.find_elements(By.CSS_SELECTOR, selector)
                if els and any(self._element_is_displayed(el) for el in els):
                    logger.debug(f"[selenium] Limit selector matched: {selector}")
                    return True
            except Exception:
                pass
        return False

    def _get_error_indicator_text(self, driver: Any) -> Optional[str]:
        """Return the text of a visible error banner/toast if one is present.

        Some web UIs render generation failures (e.g. "Something went wrong
        (1076)") in a transient snackbar/toast *outside* the response area, so
        the normal response-text detection never sees them and the request would
        stall until timeout. This checks the configured ``error_indicator``
        selectors and, when one matches a visible element, returns its text (or a
        generic marker) so the caller can fail fast and retry.

        Returns ``None`` when no error indicator is present or none are
        configured (keeping behaviour unchanged for engines without the key).
        """
        for selector in self.error_indicator_selectors:
            try:
                els = driver.find_elements(By.CSS_SELECTOR, selector)
            except Exception:
                continue
            for el in els:
                try:
                    if not self._element_is_displayed(el):
                        continue
                except Exception:
                    continue
                try:
                    text = (el.text or "").strip()
                except Exception:
                    text = ""
                logger.warning(
                    "[selenium] Error indicator matched selector %r: %r",
                    selector,
                    text[:200],
                )
                return text or "engine error indicator detected"
        return None

    def _wait_for_page_ready(self, driver: Any, timeout: float = 30.0) -> bool:
        """Wait for page readyState complete, then wait for any prompt-area selector
        to appear in the DOM so that SPA engines (e.g. Gemini Angular app) have
        finished mounting their editor before we try to interact with it.
        """
        # Phase 1 — readyState complete (basic DOM loaded)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                state = driver.execute_script("return document.readyState")
                if state == "complete":
                    logger.debug("[selenium] Page readyState complete")
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            logger.debug("[selenium] Page did not reach readyState complete")
            return False

        # Phase 2 — wait for at least one prompt-area selector to exist in the DOM
        # (SPA frameworks render the editor asynchronously after initial load)
        remaining = max(5.0, deadline - time.time())
        poll_end = time.time() + remaining
        while time.time() < poll_end:
            for sel in self.prompt_area_selectors:
                try:
                    els = driver.find_elements(By.CSS_SELECTOR, sel)
                    if els:
                        logger.debug(
                            f"[selenium] Prompt area detected in DOM after navigation: {sel}"
                        )
                        return True
                except Exception:
                    pass
            time.sleep(0.5)

        logger.warning(
            "[selenium] Prompt area selector never appeared after page load — proceeding anyway"
        )
        return False

    # ------------------------------------------------------------------ prompt chunking

    def _should_split_prompt(self, prompt: str) -> bool:
        """Return True if *prompt* exceeds the current model's char limit and chunking is enabled."""
        if self._split_prompt_parts <= 1:
            return False
        limit = self._get_model_limit(self.get_current_model())
        return len(prompt) > limit

    def _split_prompt_into_parts(self, prompt: str, n: int) -> list[str]:
        """Split *prompt* into *n* roughly equal text chunks.

        If the prompt carries a protected tail marker (an opaque boundary the
        caller inserts to keep a trailing instruction intact), everything from
        that marker onward is kept as a single final chunk so it reaches the
        model whole — never split across chunk boundaries. The marker itself is
        stripped from the text that is actually sent. This layer treats the
        marker as opaque and never inspects the tail's contents.
        """
        marker_pos = prompt.rfind(AGENT_TAIL_MARKER)
        if marker_pos != -1:
            head = prompt[:marker_pos]
            tail = prompt[marker_pos + len(AGENT_TAIL_MARKER):].lstrip("\n")
            # Reserve the final slot for the protected tail; split the head into
            # the remaining n-1 parts (at least 1).
            head_parts_n = max(n - 1, 1)
            head_chunk_size = max(math.ceil(len(head) / head_parts_n), 1)
            head_parts = [
                head[i : i + head_chunk_size]
                for i in range(0, len(head), head_chunk_size)
            ]
            if not head_parts:
                head_parts = [""]
            return head_parts + [tail]
        chunk_size = math.ceil(len(prompt) / n)
        return [prompt[i : i + chunk_size] for i in range(0, len(prompt), chunk_size)]

    def _execute_chunked_send(self, prompt: str, driver: Any) -> str:
        """Send an oversized *prompt* in sequential chunks, keeping the session open.

        Parts 1..N-1 are prefixed with an instruction telling the LLM not to
        respond yet.  Only the final part triggers a real reply.

        Optimisation: after each intermediate chunk is accepted (_post_send_check
        sees the stop button), the *next* chunk is pre-filled into the input area
        immediately — while the model is still generating its acknowledgement.
        Sending is deferred until just the send button becomes available again
        (_wait_for_send_ready), skipping the full _wait_for_response round-trip
        and its 1-second stability window for every intermediate chunk.
        """
        limit = self._get_model_limit(self.get_current_model())
        # Calculate the minimum number of parts that keeps every chunk inside the limit.
        # Never exceed the configured maximum.
        min_parts = math.ceil(len(prompt) / limit)
        n = min(self._split_prompt_parts, max(min_parts, 2))
        parts = self._split_prompt_into_parts(prompt, n)
        # The tail-marker path may return a different part count than requested
        # (it reserves a dedicated final chunk for the protected tail), so keep
        # n in sync with what was actually produced to drive the send loop.
        n = len(parts)
        chunk_t0 = time.time()
        logger.info(
            f"[selenium] Prompt chunking: {len(prompt)} chars split into {n} parts "
            f"(limit={limit}, env_max={self._split_prompt_parts})"
        )
        _engine_name = getattr(self, "ENGINE_NAME", "default")

        def _intermediate_text(idx: int, part: str) -> str:
            header = (
                f"[PART {idx}/{n}] Reply ONLY: OK\n\n"
            )
            return header + part

        # --- Send first chunk ---
        first_text = _intermediate_text(1, parts[0])
        logger.debug(f"[selenium] Sending chunk 1/{n} ({len(first_text)} chars)")
        debug_mode.record_event(
            "chunk", _engine_name, index=1, total=n, size=len(first_text),
            text=first_text,
        )
        input_el = self._find_interactable_element(
            driver, self.prompt_area_selectors, timeout=20.0,
            cache_attr="_cached_prompt_selector",
        )
        if input_el is None:
            raise RuntimeError(f"Could not find prompt input area for chunk 1/{n}")
        self._fill_input(driver, input_el, first_text)
        self._click_accept_buttons(driver, timeout=2.0)
        self._click_send(driver, input_el)
        self._click_accept_buttons(driver, timeout=2.0)
        # For chunked mode, use shorter timeout - just need to see generation started
        if not self._post_send_check(driver, timeout=8.0):
            self._cached_prompt_selector = None
            self._cached_send_selector = None
            raise RuntimeError(f"redirect-stall: chunk 1/{n} not accepted after redirect")
        logger.info(f"[timing] chunk 1/{n} sent+accepted: {time.time() - chunk_t0:.2f}s")

        # --- Send remaining chunks sequentially ---
        # Pipeline: fill chunk idx -> wait for send button (LLM finished idx-1) -> send chunk idx
        for idx in range(2, n + 1):
            is_final = idx == n
            chunk_start = time.time()

            # Step 1: Fill chunk idx IMMEDIATELY after previous send
            next_text = parts[idx - 1] if not is_final else parts[-1]
            if not is_final:
                next_text = _intermediate_text(idx, next_text)

            input_el = self._find_interactable_element(
                driver, self.prompt_area_selectors, timeout=5.0,
                cache_attr="_cached_prompt_selector",
            )
            if input_el is None:
                raise RuntimeError(f"Could not find prompt input area for chunk {idx}/{n}")

            self._fill_input(driver, input_el, next_text)
            logger.debug(f"[selenium] Filled chunk {idx} ({len(next_text)} chars)")
            debug_mode.record_event(
                "chunk", _engine_name, index=idx, total=n, size=len(next_text),
                text=next_text,
            )

            # Step 2: Wait for send button (LLM finished processing chunk idx-1)
            chunk_timeout = 30.0 if is_final else 5.0
            send_ready_start = time.time()
            if not self._wait_for_send_ready(driver, timeout=chunk_timeout):
                if is_final:
                    logger.warning(
                        f"[selenium] Send button not found after {chunk_timeout}s for final chunk, proceeding..."
                    )
                else:
                    logger.debug(
                        f"[selenium] Chunk {idx-1} not ready after {chunk_timeout}s, proceeding"
                    )

            send_elapsed = time.time() - send_ready_start
            logger.info(f"[timing] chunk {idx}/{n} wait={send_elapsed:.2f}s")

            # Step 3: Send the chunk (which is already filled)
            self._click_accept_buttons(driver, timeout=2.0)
            try:
                self._click_send(driver, input_el)
            except Exception:
                logger.warning("[selenium] Click send failed, using Enter key fallback")
                try:
                    input_el.send_keys(Keys.RETURN)
                except Exception:
                    pass
            self._click_accept_buttons(driver, timeout=2.0)
            logger.debug(f"[selenium] Sent chunk {idx}")

            if is_final:
                # Final chunk: verify it was accepted and wait for response
                self._skip_split_for_next = True
                try:
                    if not self._post_send_check(driver):
                        self._cached_prompt_selector = None
                        self._cached_send_selector = None
                        raise RuntimeError(
                            "redirect-stall: final chunk not accepted by UI after send"
                        )
                    response = self._wait_for_response(driver)

                    # ── post-response cleanup ────────────────────────────────
                    # Same as in _sync_generate_response_once: if the stop button
                    # is still visible after a successful response, refresh the
                    # page so the next request starts clean.
                    try:
                        if self._stop_button_present(driver):
                            logger.info(
                                "[selenium] Stop button still visible after chunked "
                                "response detection — refreshing page for clean state"
                            )
                            driver.refresh()
                            self._wait_for_page_ready(driver, timeout=30.0)
                            self._cached_prompt_selector = None
                            self._cached_send_selector = None
                    except Exception as refresh_err:
                        logger.warning(
                            "[selenium] Post-chunked-response cleanup refresh failed: %s",
                            refresh_err,
                        )

                    logger.info(
                        f"[timing] chunk {idx}/{n} (final) response: {time.time() - chunk_start:.2f}s"
                    )
                    logger.info(
                        f"[timing] TOTAL chunked send: {time.time() - chunk_t0:.2f}s"
                    )
                    return response
                finally:
                    self._skip_split_for_next = False
            else:
                # Verify intermediate chunk was accepted (stop button appears)
                if not self._post_send_check(driver, timeout=5.0):
                    self._cached_prompt_selector = None
                    self._cached_send_selector = None
                    raise RuntimeError(
                        f"redirect-stall: chunk {idx}/{n} not accepted by UI after send"
                    )
                logger.debug(f"[selenium] Chunk {idx}/{n} accepted")

        # Never reached: n >= 2 is guaranteed by max(min_parts, 2).
        raise RuntimeError("_execute_chunked_send: unexpected exit after loop")

    # ------------------------------------------------------------------ core flow

    def _sync_generate_response(self, prompt: str, media: list[Any] | None = None) -> str:
        """Synchronous core of generate_response — runs in a worker thread."""
        max_attempts = 5
        # Reset the page-refresh budget once per *request* (not per attempt) so
        # that stuck-stop-button refreshes accumulate across retries. Resetting
        # it inside every attempt would let the budget replenish each cycle and
        # cause an infinite paste → send → refresh loop.
        self._page_refresh_attempts = 0
        for attempt in range(max_attempts):
            try:
                try:
                    return self._sync_generate_response_once(prompt, media)
                except TypeError:
                    return self._sync_generate_response_once(prompt)
            except (RuntimeError, TimeoutException) as e:
                # Off-site redirect-stall is deterministic (the service bounced
                # to a login/auth host): retrying wastes the whole budget, so
                # fail fast and surface the error to the caller immediately.
                if self._is_offsite_redirect_stall(e):
                    logger.warning(
                        f"[selenium] Off-site redirect-stall on attempt {attempt + 1}: "
                        "service requires authentication — failing fast without retry."
                    )
                    raise

                is_chunking_freeze = any(msg in str(e) for msg in [
                    "Send button did not become ready",
                    "redirect-stall",
                    "not accepted by UI",
                    "Could not find prompt input area",
                    "selenium_response_detection_timeout",
                    "TimeoutException",
                    "disconnected"
                ])
                
                # Dynamic prompt resizing: only for truly massive prompts that fail
                # multiple times - don't reset driver on every timeout
                if is_chunking_freeze and self._should_split_prompt(prompt) and attempt < max_attempts - 1:
                    self._split_prompt_parts += 1
                    logger.warning(
                        f"[selenium] Chunking failure on attempt {attempt + 1}: {e}. "
                        f"Dynamically resizing split_prompt_parts to {self._split_prompt_parts} and retrying."
                    )
                    self._reset_driver()
                    continue

                if attempt >= max_attempts - 1:
                    raise

                if isinstance(e, RuntimeError):
                    if self._is_page_refresh_required(e):
                        if self._page_refresh_attempts < self._max_page_refresh_attempts:
                            logger.warning(
                                f"[selenium] Page refresh was performed on attempt {attempt + 1}, "
                                "retrying without driver reset…"
                            )
                            # Driver already refreshed in _wait_for_response; just retry
                            continue
                        else:
                            logger.warning(
                                f"[selenium] Max page refresh attempts reached on attempt {attempt + 1}, "
                                "resetting driver and retrying…"
                            )
                            self._reset_driver()
                            continue
                    if self._is_dead_session(e):
                        if not getattr(self, "_prefer_webdriver_fallback", False):
                            logger.warning(
                                "[selenium] Dead session on attempt %s, forcing webdriver.Chrome fallback on next init",
                                attempt + 1,
                            )
                            self._prefer_webdriver_fallback = True
                        else:
                            logger.warning(
                                f"[selenium] Dead session on attempt {attempt + 1}, resetting and retrying…"
                            )
                        self._reset_driver()
                        continue
                    if self._is_redirect_stall(e):
                        logger.warning(
                            f"[selenium] Redirect-stall on attempt {attempt + 1}, retrying without driver reset…"
                        )
                        continue
                    if self._is_response_detection_timeout(e):
                        if getattr(self, "_generation_was_active", False):
                            logger.warning(
                                f"[selenium] Response detection timeout on attempt {attempt + 1} "
                                "(model was actively generating — slow/thinking model, skipping driver reset)"
                            )
                        else:
                            logger.warning(
                                f"[selenium] Response detection timeout on attempt {attempt + 1}, "
                                "resetting driver and retrying…"
                            )
                            self._reset_driver()
                        continue
                else: 
                    # TimeoutException
                    if self._is_response_detection_timeout(e):
                        if getattr(self, "_generation_was_active", False):
                            logger.warning(
                                f"[selenium] Response detection timeout on attempt {attempt + 1} "
                                "(model was actively generating — slow/thinking model, skipping driver reset)"
                            )
                        else:
                            logger.warning(
                                f"[selenium] Response detection timeout on attempt {attempt + 1}, "
                                "resetting driver and retrying…"
                            )
                            self._reset_driver()
                        continue

        raise RuntimeError("_sync_generate_response exhausted retries")

    def _sync_generate_response_once(self, prompt: str, media: list[Any] | None = None) -> str:
        """Single attempt of the core generate flow.

        Note: ``_page_refresh_attempts`` is intentionally *not* reset here. The
        budget is reset once per request in ``_sync_generate_response`` so that
        stuck-stop-button page refreshes accumulate across attempts and can
        eventually exhaust the budget (triggering a driver reset) instead of
        looping forever.
        """
        t0 = time.time()
        self._ensure_ready()

        unlogged = not self.is_user_logged_in()
        if unlogged:
            logger.warning(
                "[selenium] User is unlogged; continuing with unlogged mode (restricted/unreliable)."
            )

        assert self.driver is not None, "_ensure_ready() must have set self.driver"
        driver = cast(webdriver.Chrome, self.driver)

        # Skip navigation if already on the service page (avoids full page reload)
        needs_nav = True
        try:
            current_url = driver.current_url or ""
            if current_url.startswith(self.service_url):
                needs_nav = False
                logger.debug("[selenium] Already on service URL, skipping navigation")
        except Exception:
            pass  # dead session or no URL — navigate anyway

        if needs_nav:
            try:
                driver.get(self.service_url)
            except Exception as nav_err:
                if self._is_dead_session(nav_err):
                    self._reset_driver()
                    raise RuntimeError(
                        f"Driver session died during navigation: {nav_err}"
                    ) from nav_err
                raise
            self._wait_for_page_ready(driver, timeout=30.0)

        t1 = time.time()
        logger.info(f"[timing] page_ready: {t1 - t0:.2f}s")

        # Dismiss any gate/consent dialog (e.g. a Terms-of-Service acknowledgement)
        # that may block the prompt area before it can be located. This is a no-op
        # for engines that do not configure accept_button selectors.
        self._click_accept_buttons(driver, timeout=2.0)

        if self._is_captcha_present(driver):
            logger.warning("[selenium] Cloudflare captcha challenge detected on page")
            return (
                "⚠️ Cloudflare CAPTCHA detected. "
                "Please complete the CAPTCHA on the page and try again."
            )

        if self._is_limit_present(driver):
            logger.warning("[selenium] Limit warning detected on page")
            return (
                "⚠️ The service appears to have hit a usage limit. "
                "Please upgrade or wait, usually until tomorrow, before retrying."
            )

        if media:
            tier = self._check_account_tier(driver)
            limit_error = self._check_media_limits(media, tier)
            if limit_error:
                return limit_error
            if not self._upload_media(media, driver):
                return (
                    "⚠️ Media upload failed. "
                    "Please verify the file and try again."
                )
            # When a vision prompt prefix is configured, prepend it so the engine
            # responds in plain text rather than the trained JSON/action schema.
            if self._vision_prompt_prefix:
                prompt = self._vision_prompt_prefix + prompt
        else:
            # When no media is present, prepend the inline response prefix to
            # instruct the LLM to respond inline (e.g. avoid canvas/document
            # mode) and prevent stuck states that show "Something went wrong".
            if self._inline_response_prefix:
                prompt = self._inline_response_prefix + prompt

        # Prompt chunking: split oversized prompts into sequential parts.
        # (The tail marker, if any, is handled inside _execute_chunked_send.)
        if not self._skip_split_for_next and not media and self._should_split_prompt(prompt):
            return self._execute_chunked_send(prompt, driver)

        # Non-chunked path: the tail marker is only a chunk-boundary hint, so
        # strip it before sending the prompt as a single message.
        if AGENT_TAIL_MARKER in prompt:
            prompt = prompt.replace(AGENT_TAIL_MARKER, "")

        stale_retries = 0
        while True:
            try:
                input_el = self._find_interactable_element(
                    driver, self.prompt_area_selectors, timeout=20.0,
                    cache_attr="_cached_prompt_selector",
                )
                if input_el is None:
                    raise RuntimeError("Could not find prompt input area")

                t2 = time.time()
                logger.info(f"[timing] find_element: {t2 - t1:.2f}s")

                self._fill_input(driver, input_el, prompt)
                t3 = time.time()
                logger.info(f"[timing] fill_input: {t3 - t2:.2f}s ({len(prompt)} chars)")

                self._click_accept_buttons(driver, timeout=2.0)
                if media:
                    if not self._wait_for_send_button_after_media_upload(driver):
                        logger.warning(
                            "[selenium] Send button not ready after media upload"
                        )
                        return (
                            "⚠️ Media upload failed. "
                            "Please verify the file and try again."
                        )
                self._click_send(driver, input_el)
                t4 = time.time()
                logger.info(f"[timing] click_send: {t4 - t3:.2f}s")

                self._click_accept_buttons(driver, timeout=2.0)
                if not self._post_send_check(driver):
                    self._cached_prompt_selector = None
                    self._cached_send_selector = None
                    if getattr(self, "_last_send_offsite_redirect", False):
                        # Deterministic cross-host redirect (e.g. the service
                        # bounced to a login/auth domain). Retrying cannot help,
                        # so mark it off-site to trigger fail-fast in the caller.
                        raise RuntimeError(
                            "redirect-stall (off-site): send not accepted after redirect"
                        )
                    raise RuntimeError(
                        "redirect-stall: send not accepted after redirect"
                    )
                t5 = time.time()
                logger.info(f"[timing] post_send_check: {t5 - t4:.2f}s")

                response = self._wait_for_response(driver)

                # ── engine error detection ───────────────────────────────────
                # If the response text looks like an LLM web UI error (e.g.
                # "Something went wrong (13)"), treat it as a transient failure
                # and raise so the retry loop in _sync_generate_response can
                # reset the driver and try again.
                if self._is_engine_error_response(response):
                    logger.warning(
                        "[selenium] Engine error detected in response: %r — "
                        "raising for retry with driver reset",
                        response[:200],
                    )
                    self._reset_driver()
                    raise RuntimeError(
                        "selenium_response_detection_timeout: engine error "
                        "response detected — driver reset for retry"
                    )

                # ── post-response cleanup ────────────────────────────────────
                # If the stop button is still visible after a successful response
                # detection (the UI framework may keep it visible briefly or
                # indefinitely when stuck), refresh the page so the next request
                # starts with a clean slate.
                try:
                    if self._stop_button_present(driver):
                        logger.info(
                            "[selenium] Stop button still visible after response "
                            "detection — refreshing page for clean state"
                        )
                        driver.refresh()
                        self._wait_for_page_ready(driver, timeout=30.0)
                        self._cached_prompt_selector = None
                        self._cached_send_selector = None
                except Exception as refresh_err:
                    logger.warning(
                        "[selenium] Post-response cleanup refresh failed: %s",
                        refresh_err,
                    )

                t6 = time.time()
                logger.info(f"[timing] wait_for_response: {t6 - t5:.2f}s")
                logger.info(f"[timing] TOTAL generate: {t6 - t0:.2f}s")
                return response

            except StaleElementReferenceException as e:
                stale_retries += 1
                self._cached_prompt_selector = None
                self._cached_send_selector = None
                if stale_retries >= 2:
                    logger.error(
                        "[selenium] _sync_generate_response_once failed after stale element retry: %s",
                        e,
                    )
                    raise
                logger.warning(
                    "[selenium] Stale element detected mid-prompt; retrying prompt flow (%s/2): %s",
                    stale_retries,
                    e,
                )
                continue

            except Exception as e:
                if self._is_dead_session(e):
                    self._reset_driver()
                    raise RuntimeError(f"Driver session died mid-prompt: {e}") from e
                logger.error(f"[selenium] _sync_generate_response_once failed: {e}")
                if unlogged:
                    return f"⚠️ Unlogged session: could not run full prompt flow. Error: {e}"
                raise

    # ------------------------------------------------------------------ media helpers

    def _check_account_tier(self, driver: Any) -> str:
        if not self.is_user_logged_in():
            return "unlogged"
        if getattr(self, "base_account_selector", None):
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, self.base_account_selector)
                for element in elements:
                    if self._element_is_displayed(element):
                        return "base"
            except Exception:
                pass
        if self.paid_account_selector:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, self.paid_account_selector)
                for element in elements:
                    if self._element_is_displayed(element):
                        return "paid"
            except Exception:
                pass
        return "base"

    def _check_media_limits(self, media: list[Any], tier: str) -> str | None:
        counts: dict[str, int] = {}
        for item in media:
            media_type = getattr(item, "media_type", None)
            if not media_type:
                continue
            counts[media_type] = counts.get(media_type, 0) + 1

        configured_types = [
            key
            for key in self.media_config.keys()
            if key not in {
                "paid_account_selector",
                "base_account_selector",
                "total_limits",
                "shared_limits",
            }
        ]
        if not configured_types:
            return None

        for media_type, count in counts.items():
            cfg = self.media_config.get(media_type)
            if cfg is None:
                logger.debug(
                    f"[selenium] No media_support entry for '{media_type}', will try upload anyway"
                )
                continue

            limit_error = self._evaluate_media_limit(
                media_type, cfg, count, tier
            )
            if limit_error:
                return limit_error

        total_cfg = self.media_config.get("total_limits") or self.media_config.get("shared_limits")
        if total_cfg is not None:
            total_count = sum(counts.values())
            total_error = self._evaluate_media_limit(
                "media",
                total_cfg,
                total_count,
                tier,
                total=True,
            )
            if total_error:
                return total_error

        return None

    def _evaluate_media_limit(
        self,
        media_type: str,
        cfg: Any,
        count: int,
        tier: str,
        total: bool = False,
    ) -> str | None:
        if cfg is None:
            return None

        supported_models = cfg.get("supported_models")
        if supported_models is not None:
            current_model = self.get_current_model()
            if "all" not in supported_models:
                if "not-unlogged" in supported_models:
                    if current_model == "unlogged":
                        logger.debug(
                            f"[selenium] Media '{media_type}' explicitly not supported for unlogged sessions"
                        )
                elif current_model not in supported_models:
                    logger.debug(
                        f"[selenium] Model '{current_model}' not listed for '{media_type}' media; will still attempt upload"
                    )

        limits = cfg.get("limits", {})
        if not limits:
            return (
                f"⚠️ Media type '{media_type}' is not supported by this engine."
            )

        tier_limit = limits.get(tier, 0)
        if tier_limit == -1 or tier_limit == 0:
            logger.debug(
                f"[selenium] Media '{media_type}' has zero/unlimited limit for tier '{tier}'; attempting upload anyway"
            )
            return None
        if count > tier_limit:
            kind = "media" if total else media_type
            return (
                f"⚠️ The use of '{kind}' is exhausted for today. Please try again tomorrow."
            )
        return None

    def _element_is_displayed(self, element: Any) -> bool:
        try:
            return bool(element.is_displayed())
        except Exception:
            return False

    def _wait_for_media_upload_complete(
        self,
        item: Any,
        driver: Any,
        timeout: float = 20.0,
    ) -> bool:
        selectors = self.media_config.get(item.media_type, {}).get(
            "upload_complete_selectors", []
        )
        if not selectors:
            return True

        start = time.time()
        timeout = float(os.getenv("SELENIUM_MEDIA_UPLOAD_COMPLETE_WAIT", str(timeout)))
        deadline = start + timeout

        while time.time() < deadline:
            all_satisfied = True
            for sel in selectors:
                absent = False
                raw_sel = sel
                if sel.startswith("!"):
                    absent = True
                    raw_sel = sel[1:]

                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, raw_sel)
                except Exception:
                    elements = []

                if absent:
                    if elements:
                        all_satisfied = False
                        break
                else:
                    if not any(self._element_is_displayed(el) for el in elements):
                        all_satisfied = False
                        break

            if all_satisfied:
                logger.debug(
                    f"[selenium] Media upload completion selectors satisfied: {selectors}"
                )
                return True
            time.sleep(0.25)

        logger.warning(
            f"[selenium] Media upload completion wait timed out for selectors: {selectors}"
        )
        return False

    def _upload_media(self, media: list[Any], driver: Any) -> bool:
        for item in media:
            tmp_path = self._write_temp_media_file(item)
            try:
                upload_success = self._upload_via_file_input(item, tmp_path, driver)
                if not upload_success:
                    upload_success = self._upload_via_clipboard(item, tmp_path, driver)

                if not upload_success:
                    logger.warning(
                        f"[selenium] Media upload failed for type '{item.media_type}'"
                    )
                    return False

                # Some services display an intermediate consent/dialog step after
                # the file is selected. Click any configured agree/accept buttons
                # immediately before continuing the prompt flow.
                self._click_accept_buttons(driver, timeout=5.0)

                if not self._wait_for_media_upload_complete(item, driver):
                    logger.warning(
                        f"[selenium] Media upload did not reach completion state for type '{item.media_type}'"
                    )
                    return False
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        return True

    def _wait_for_send_button_after_media_upload(
        self,
        driver: Any,
        timeout: float = 15.0,
    ) -> bool:
        start = time.time()
        timeout = float(os.getenv("SELENIUM_MEDIA_UPLOAD_WAIT", str(timeout)))
        deadline = start + timeout

        while time.time() < deadline:
            for sel in self.send_button_selectors:
                try:
                    buttons = driver.find_elements(By.CSS_SELECTOR, sel)
                    for button in buttons:
                        try:
                            if button.is_displayed() and button.is_enabled():
                                elapsed = time.time() - start
                                logger.debug(
                                    f"[selenium] Send button ready after media upload "
                                    f"for selector {sel} in {elapsed:.2f}s"
                                )
                                return True
                        except Exception:
                            pass
                except Exception:
                    pass
            time.sleep(0.25)

        logger.warning(
            "[selenium] Send button did not become ready after media upload "
            f"within {timeout:.1f}s"
        )
        return False

    def _write_temp_media_file(self, item: Any) -> str:
        suffix = ""
        if item.mime_type and "/" in item.mime_type:
            ext = mimetypes.guess_extension(item.mime_type)
            if ext:
                suffix = ext
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(item.data)
        tmp.flush()
        tmp.close()
        return tmp.name

    def _upload_via_file_input(self, item: Any, path: str, driver: Any) -> bool:
        selectors = self.media_config.get(item.media_type, {}).get(
            "upload_selectors", ["input[type='file']"]
        )
        for sel in selectors:
            try:
                inputs = driver.find_elements(By.CSS_SELECTOR, sel)
            except Exception as exc:
                logger.debug(f"[selenium] File input selector failed: {sel}: {exc}")
                continue

            if not inputs:
                logger.debug(f"[selenium] File input selector found no elements: {sel}")
                continue

            for inp in inputs:
                try:
                    if inp.tag_name.lower() != "input":
                        continue
                    if (inp.get_attribute("type") or "").lower() != "file":
                        continue
                    send_ok = False
                    try:
                        inp.send_keys(path)
                        send_ok = True
                    except Exception as exc:
                        logger.debug(
                            f"[selenium] File input send_keys error for selector {sel}: {exc}"
                        )
                        try:
                            driver.execute_script(
                                "arguments[0].style.display='block';"
                                "arguments[0].style.visibility='visible';"
                                "arguments[0].style.opacity='1';"
                                "arguments[0].style.position='fixed';"
                                "arguments[0].style.width='1px';"
                                "arguments[0].style.height='1px';",
                                inp,
                            )
                        except Exception:
                            pass
                        try:
                            inp.send_keys(path)
                            send_ok = True
                        except Exception as exc2:
                            logger.warning(
                                f"[selenium] File input send_keys failed after visibility fallback for selector {sel}: {exc2}"
                            )
                            continue

                    end_time = time.time() + 2.0
                    while time.time() < end_time:
                        try:
                            value = inp.get_attribute("value")
                            files_count = driver.execute_script(
                                "return arguments[0].files ? arguments[0].files.length : 0;",
                                inp,
                            )
                            if value or files_count:
                                logger.debug(
                                    f"[selenium] Media file input received value/files for selector {sel}"
                                )
                                return True
                        except Exception:
                            pass
                        time.sleep(0.1)
                    # SPA frameworks (e.g. Angular) may clear .value/.files immediately
                    # after processing the upload. If send_keys completed without error,
                    # trust that the file was accepted.
                    if send_ok:
                        logger.debug(
                            f"[selenium] File input value/files empty after send_keys "
                            f"(SPA reset?); trusting send for selector {sel}"
                        )
                        return True
                    logger.debug(
                        f"[selenium] Media file input did not report a value/files for selector {sel}"
                    )
                    continue
                except Exception:
                    continue
        return False

    def _upload_via_clipboard(self, item: Any, path: str, driver: Any) -> bool:
        clipboard_cmd = None
        if shutil.which("xclip"):
            clipboard_cmd = ["xclip", "-selection", "clipboard", "-t", item.mime_type, "-i", path]
        elif shutil.which("xsel"):
            clipboard_cmd = ["xsel", "--clipboard", "--input"]
        elif shutil.which("wl-copy"):
            clipboard_cmd = ["wl-copy", "--type", item.mime_type]

        if clipboard_cmd is None:
            logger.warning("[selenium] Clipboard upload skipped: no clipboard utility found")
            return False

        xclip_proc = None
        try:
            if clipboard_cmd[0] in ("xsel", "wl-copy"):
                with open(path, "rb") as fh:
                    subprocess.run(
                        clipboard_cmd,
                        input=fh.read(),
                        check=True,
                        capture_output=True,
                        timeout=15,
                    )
            else:
                # xclip -i blocks until the clipboard is consumed by a reader;
                # run it in the background so that the paste below can succeed.
                xclip_proc = subprocess.Popen(
                    clipboard_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(0.3)  # allow xclip to register as clipboard owner
            logger.debug(f"[selenium] Copied media to clipboard using {clipboard_cmd[0]}")
        except Exception as exc:
            logger.warning(f"[selenium] Clipboard transfer failed with {clipboard_cmd[0]}: {exc}")
            return False

        input_el = self._find_interactable_element(
            driver,
            self.prompt_area_selectors,
            timeout=10.0,
            cache_attr="_cached_prompt_selector",
        )
        if input_el is None:
            logger.warning("[selenium] Clipboard upload failed: prompt area not found")
            return False

        try:
            try:
                driver.execute_script("arguments[0].focus();", input_el)
            except Exception:
                pass
            try:
                input_el.click()
            except Exception:
                pass
            time.sleep(0.1)
            try:
                ActionChains(driver).key_down(Keys.CONTROL).send_keys("v").key_up(Keys.CONTROL).perform()
            except Exception:
                input_el.send_keys(Keys.CONTROL, "v")
            time.sleep(0.5)
            # Terminate background xclip process after paste is delivered
            if xclip_proc is not None:
                try:
                    xclip_proc.terminate()
                except Exception:
                    pass
            logger.debug("[selenium] Clipboard paste attempted")
            return True
        except Exception as exc:
            if xclip_proc is not None:
                try:
                    xclip_proc.terminate()
                except Exception:
                    pass
            logger.warning(f"[selenium] Clipboard paste failed: {exc}")
            return False

    # ------------------------------------------------------------------ helpers

    def _looks_like_response_text_junk(self, text: str) -> bool:
        """Return True for text that appears to be page state JSON rather than chat output."""
        if not text:
            return True

        stripped = text.strip()
        if len(stripped) > 300:
            return False

        # In agent mode the model is expected to emit short structured JSON
        # (tool calls / final content). Do not treat that JSON as junk — only
        # keep the memory/recovery state-dump guards below active.
        if not self._agent_mode and (
            stripped.startswith("{") or stripped.startswith("[")
        ):
            try:
                json.loads(stripped)
                logger.debug(
                    "[selenium] _looks_like_response_text_junk: rejected JSON-like text"
                )
                return True
            except Exception:
                pass

        if stripped.startswith("Gemini said"):
            after = stripped[len("Gemini said"):].strip()
            if not after:
                logger.debug(
                    "[selenium] _looks_like_response_text_junk: rejected bare Gemini said prefix"
                )
                return True
            if after.startswith("{") or after.startswith("["):
                try:
                    json.loads(after)
                    logger.debug(
                        "[selenium] _looks_like_response_text_junk: rejected Gemini state dump"
                    )
                    return True
                except Exception:
                    pass
            if "\"memory_search\"" in after or "\"recovery_actions\"" in after:
                logger.debug(
                    "[selenium] _looks_like_response_text_junk: rejected Gemini state prefix"
                )
                return True

        if "\"memory_search\"" in stripped or "\"recovery_actions\"" in stripped:
            logger.debug(
                "[selenium] _looks_like_response_text_junk: rejected memory/search state text"
            )
            return True

        return False

    def _get_response_text_js(self, driver: Any) -> str:
        """Return fallback response text using JavaScript when CSS selectors fail."""
        try:
            script = """
            const selectors = [
                '.assistant-message',
                '.assistant',
                '.gemini-response',
                'message-content .markdown-main-panel',
                '.markdown-main-panel',
                'model-response .markdown',
                'model-response',
                '.model-response',
                '.response-container',
                '.presented-response-container',
                '.structured-content-container',
                '.markdown',
                'message-content',
                '.message-content',
                'div[role="article"]',
                'article'
            ];
            for (const sel of selectors) {
                const els = Array.from(document.querySelectorAll(sel));
                for (let i = els.length - 1; i >= 0; i--) {
                    const el = els[i];
                    if (el && el.textContent && el.textContent.trim()) {
                        return el.textContent.trim();
                    }
                }
            }
            return '';
            """
            result = driver.execute_script(script)
            if isinstance(result, str):
                result = result.strip()
                if not self._looks_like_response_text_junk(result):
                    return result
        except Exception:
            pass
        return ""

    def _get_latest_response_text(self, driver: Any) -> str:
        """Return latest non-empty text from response selectors, or empty if none."""
        candidates: list[str] = []
        logger.debug(
            "[selenium] _get_latest_response_text: trying selectors=%s",
            self.response_area_selectors,
        )
        for sel in self.response_area_selectors:
            try:
                els = driver.find_elements(By.CSS_SELECTOR, sel)
                logger.debug(
                    "[selenium] _get_latest_response_text: selector=%s found %d elements",
                    sel,
                    len(els),
                )
                if not els:
                    continue

                for elem in reversed(els):
                    text = elem.text.strip()
                    tc = ""
                    try:
                        tc = (elem.get_attribute("textContent") or "").strip()
                    except Exception:
                        pass

                    if tc and self._looks_like_response_text_junk(tc):
                        logger.debug(
                            "[selenium] _get_latest_response_text: selector=%s text=%r tc=%r rejected as junk",
                            sel,
                            text[:120],
                            tc[:120],
                        )
                        continue

                    if text and not self._looks_like_response_text_junk(text):
                        logger.debug(
                            "[selenium] _get_latest_response_text: selector=%s returned visible text=%r",
                            sel,
                            text[:120],
                        )
                        return text
                    if tc and not self._looks_like_response_text_junk(tc):
                        logger.debug(
                            "[selenium] _get_latest_response_text: selector=%s used textContent fallback=%r",
                            sel,
                            tc[:120],
                        )
                        return tc

                    if text:
                        candidates.append(text)
                    if tc and tc != text:
                        candidates.append(tc)
            except Exception:
                pass

        for candidate in candidates:
            if not self._looks_like_response_text_junk(candidate):
                return candidate

        return self._get_response_text_js(driver)

    def _send_button_present(self, driver: Any) -> bool:
        """Return True if a send button is currently visible and enabled on the page.

        Used as a fallback generation-complete signal: LLMs typically hide the
        send button while streaming and re-show it once the response is done.
        Send button present → generation finished; send button absent → still generating.
        """
        for sel in self.send_button_selectors:
            try:
                btns = driver.find_elements(By.CSS_SELECTOR, sel)
                for b in btns:
                    try:
                        if b.is_displayed() and b.is_enabled():
                            return True
                    except Exception:
                        pass
            except Exception:
                pass
        return False

    def _response_area_present(self, driver: Any) -> bool:
        """Return True if any configured response selector exists in the DOM."""
        for sel in self.response_area_selectors:
            try:
                if driver.find_elements(By.CSS_SELECTOR, sel):
                    return True
            except Exception:
                pass
        return False

    def _stop_button_present(self, driver: Any) -> bool:
        """Return True if the stop button is visible (model is still generating).

        Used for early freeze detection: if stop button is visible but there's
        no text expansion for > 20s, it's a silent hang.
        """
        for sel in self.stop_selectors:
            try:
                btns = driver.find_elements(By.CSS_SELECTOR, sel)
                for b in btns:
                    try:
                        if b.is_displayed():
                            return True
                    except Exception:
                        pass
            except Exception:
                pass
        return False

    def _find_response_container_element(
        self,
        driver: Any,
    ) -> tuple[Any | None, str | None]:
        """Return the most relevant response container element for configured selectors."""
        for sel in self.response_area_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, sel)
                logger.debug(
                    "[selenium] _find_response_container_element: selector=%s found %d elements",
                    sel,
                    len(elements),
                )
                if not elements:
                    continue

                visible = [el for el in elements if self._element_is_displayed(el)]
                if visible:
                    return visible[-1], sel
                return elements[-1], sel
            except Exception as exc:
                logger.debug(
                    "[selenium] _find_response_container_element: selector=%s error=%s",
                    sel,
                    exc,
                )
        return None, None

    def _get_response_container_stats(
        self,
        driver: Any,
        element: Any,
    ) -> tuple[int, int]:
        """Return generic container metrics used by the response watcher."""
        try:
            result = driver.execute_script(
                "const el = arguments[0];"
                "return [el?.innerText?.length || 0, el?.childElementCount || 0];",
                element,
            )
            if (
                isinstance(result, list)
                and len(result) == 2
                and isinstance(result[0], int)
                and isinstance(result[1], int)
            ):
                return result[0], result[1]
        except Exception as exc:
            logger.debug(
                "[selenium] _get_response_container_stats: JS failed: %s",
                exc,
            )

        try:
            text = (element.text or "").strip()
            count = int(getattr(element, "childElementCount", 0) or 0)
            return len(text), count
        except Exception:
            return 0, 0

    def _extract_response_text_from_element(
        self,
        driver: Any,
        element: Any,
    ) -> str:
        """Extract the response text from the watched container element."""
        try:
            result = driver.execute_script(
                "const el = arguments[0];"
                "return el && (el.innerText || el.textContent) ? (el.innerText || el.textContent).trim() : '';",
                element,
            )
            if isinstance(result, str):
                return result.strip()
        except Exception as exc:
            logger.debug(
                "[selenium] _extract_response_text_from_element: JS failed: %s",
                exc,
            )

        try:
            return (element.text or "").strip()
        except Exception:
            return ""

    def _log_response_container_diagnostics(
        self,
        selector: str | None,
        current_text_length: int,
        current_child_count: int,
        previous_text_length: int,
        previous_child_count: int,
        stable_counter: int,
        iteration: int,
    ) -> None:
        """Log detailed watcher diagnostics for each monitoring iteration."""
        logger.debug(
            "[selenium] watcher iteration=%d selector=%s current_text_length=%d current_child_count=%d previous_text_length=%d previous_child_count=%d stable_counter=%d",
            iteration,
            selector,
            current_text_length,
            current_child_count,
            previous_text_length,
            previous_child_count,
            stable_counter,
        )

    def _find_interactable_element(
        self,
        driver: Any,
        selectors: list[str],
        timeout: float = 20.0,
        cache_attr: Optional[str] = None,
    ) -> Optional[Any]:
        """Try CSS selectors in order; return first element that is clickable.

        If *cache_attr* is given (e.g. ``'_cached_prompt_selector'``), the last
        successful selector is tried first on subsequent calls.
        """
        # Build an ordered list: cached selector first (if valid), then the rest
        ordered: list[str] = list(selectors)
        cached: Optional[str] = getattr(self, cache_attr, None) if cache_attr else None
        if cached and cached in ordered:
            ordered.remove(cached)
            ordered.insert(0, cached)

        per = max(1.5, timeout / max(len(ordered), 1))
        for sel in ordered:
            try:
                condition = EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                # For compatibility with different Selenium versions, expose locator
                # on the callable condition object so unit tests and mocks can inspect it.
                if not hasattr(condition, "locator"):
                    try:
                        setattr(condition, "locator", (By.CSS_SELECTOR, sel))
                    except Exception:
                        pass

                el = WebDriverWait(driver, per).until(condition)
                logger.debug(f"[selenium] Found clickable element: {sel}")
                if cache_attr:
                    setattr(self, cache_attr, sel)
                return el
            except StaleElementReferenceException as e:
                logger.warning(
                    f"[selenium] Stale element for selector {sel}; invalidating cache and continuing: {e}"
                )
                if cache_attr and getattr(self, cache_attr, None) == sel:
                    setattr(self, cache_attr, None)
                continue
            except Exception as e:
                if self._is_dead_session(e):
                    raise  # propagate to _sync_generate_response for driver reset
        # Fallback: some rich editor areas are visible but not considered clickable
        for sel in ordered:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, sel)
                for el in elements:
                    try:
                        if el.is_displayed():
                            logger.debug(
                                f"[selenium] Found visible element fallback: {sel}"
                            )
                            if cache_attr:
                                setattr(self, cache_attr, sel)
                            return el
                    except Exception:
                        continue
            except Exception:
                pass

        logger.warning("[selenium] No interactable element found")
        return None

    def _normalize_input_text(self, text: str) -> str:
        return " ".join(text.replace("\r\n", "\n").strip().split())

    def _get_input_text(self, driver: Any, element: Any) -> str:
        try:
            tag = (element.tag_name or "").lower()
            if tag in ("textarea", "input"):
                return str(element.get_attribute("value") or "")
            actual = driver.execute_script(
                "const el = arguments[0];"
                "if (el.value !== undefined) return el.value || '';"
                "const text = el.innerText || el.textContent || '';"
                "return text;",
                element,
            )
            return str(actual or "")
        except Exception:
            return ""

    @staticmethod
    def _strip_all_whitespace(text: str) -> str:
        """Collapse a string to its non-whitespace characters only.

        Rich-text editors (ProseMirror, Lexical, Quill, …) frequently
        re-flow whitespace when text is inserted programmatically: they may
        split paragraphs, drop trailing spaces per line, or collapse runs of
        spaces. That makes a strict whitespace-preserving comparison too
        brittle, so this helper is used as a tolerant fallback that compares
        only the meaningful (non-whitespace) content.
        """
        return "".join(text.split())

    def _verify_input_text(self, driver: Any, element: Any, expected: str) -> bool:
        expected_norm = self._normalize_input_text(expected)
        expected_compact = self._strip_all_whitespace(expected)
        end_time = time.time() + 1.0
        actual = ""
        while time.time() < end_time:
            actual = self._get_input_text(driver, element)
            if self._normalize_input_text(actual) == expected_norm:
                return True
            # Tolerant fallback: contenteditable editors may re-flow
            # whitespace (paragraph splits, dropped trailing spaces). Accept
            # the fill when the meaningful, non-whitespace content matches.
            if self._strip_all_whitespace(actual) == expected_compact:
                logger.debug(
                    "[selenium] fill_input matched after whitespace-insensitive "
                    "comparison (editor re-flowed whitespace)"
                )
                return True
            time.sleep(0.1)
        actual_norm = self._normalize_input_text(actual)
        logger.warning(
            "[selenium] fill_input verification failed: expected %s chars, got %s chars",
            len(expected_norm),
            len(actual_norm),
        )
        logger.warning(
            "[selenium] fill_input expected_tail=%r actual_tail=%r",
            expected_norm[-160:],
            actual_norm[-160:],
        )
        return False

    def _fill_input(self, driver: Any, element: Any, text: str) -> None:
        """Type *text* into a textarea or contenteditable element."""
        attempts = 0
        while True:
            try:
                try:
                    driver.execute_script(
                        "arguments[0].scrollIntoView({block:'center'});", element
                    )
                    time.sleep(0.3)
                except Exception:
                    pass

                try:
                    element.click()
                except Exception:
                    driver.execute_script("arguments[0].click();", element)
                    time.sleep(0.1)

                tag = (element.tag_name or "").lower()
                if tag in ("textarea", "input"):
                    # Standard form inputs: clear via clear() then type
                    try:
                        element.clear()
                    except Exception as exc:
                        if isinstance(exc, StaleElementReferenceException):
                            raise
                        try:
                            element.send_keys(Keys.CONTROL + "a")
                            element.send_keys(Keys.DELETE)
                        except Exception as exc2:
                            if isinstance(exc2, StaleElementReferenceException):
                                raise
                            pass
                    element.send_keys(text)
                else:
                    # contenteditable (ProseMirror, Lexical, Quill, …)
                    # First try JS execCommand: instant, no char-by-char latency.  Some
                    # rich-text editors (notably ProseMirror-based ones) intercept and
                    # ignore execCommand, leaving stale content untouched, so we always
                    # verify the DOM afterwards and fall back to native key events, which
                    # produce real keyboard input the editor cannot ignore.
                    try:
                        # Clear any existing (possibly stale) content, then insert.
                        driver.execute_script(
                            "const el = arguments[0];"
                            "el.focus();"
                            "document.execCommand('selectAll', false, null);"
                            "document.execCommand('delete', false, null);",
                            element,
                        )
                        time.sleep(0.05)
                        driver.execute_script(
                            "const el = arguments[0];"
                            "el.focus();"
                            "document.execCommand('selectAll', false, null);"
                            "document.execCommand('insertText', false, arguments[1]);",
                            element,
                            text,
                        )
                    except Exception:
                        pass

                    # Verify execCommand actually populated the editor.  If the content
                    # does not match (editor ignored execCommand or retained stale text),
                    # fall back to native key events: select-all + delete + type.
                    exec_ok = False
                    try:
                        current = self._get_input_text(driver, element)
                        exec_ok = self._strip_all_whitespace(
                            current
                        ) == self._strip_all_whitespace(text)
                    except Exception:
                        exec_ok = False

                    if not exec_ok:
                        try:
                            element.click()
                        except Exception:
                            try:
                                driver.execute_script(
                                    "arguments[0].focus();", element
                                )
                            except Exception:
                                pass
                        try:
                            element.send_keys(Keys.CONTROL + "a")
                            element.send_keys(Keys.DELETE)
                            time.sleep(0.05)
                            element.send_keys(text)
                        except Exception as e:
                            if isinstance(e, StaleElementReferenceException):
                                raise
                            logger.error(
                                f"[selenium] fill_input send_keys failed: {e}"
                            )
                            raise
                    else:
                        # Some frameworks distinguish programmatic updates from user key
                        # events.  Trigger a whitespace keypress + backspace to force UI
                        # internals to re-evaluate and enable the send button.
                        try:
                            element.send_keys(Keys.SPACE, Keys.BACKSPACE)
                        except Exception:
                            pass

                # Dispatch input/change events so that React/Vue/framework state machines
                # immediately enable the send button without waiting for synthetic events.
                try:
                    driver.execute_script(
                        "const el = arguments[0];"
                        "const evOpt = {bubbles:true, cancelable:true, composed:true};"
                        "el.dispatchEvent(new InputEvent('input', evOpt));"
                        "el.dispatchEvent(new KeyboardEvent('keyup', evOpt));"
                        "el.dispatchEvent(new Event('change', evOpt));"
                        "el.dispatchEvent(new Event('blur', evOpt));",
                        element,
                    )
                except Exception:
                    pass
                logger.debug(f"[selenium] Filled input ({len(text)} chars)")
                if not self._verify_input_text(driver, element, text):
                    raise RuntimeError(
                        "[selenium] fill_input verification failed: prompt content did not match expected text"
                    )
                return
            except StaleElementReferenceException as exc:
                attempts += 1
                logger.warning(
                    "[selenium] fill_input stale element reference; retrying (%s/2): %s",
                    attempts,
                    exc,
                )
                if attempts >= 2:
                    raise
                element = self._find_interactable_element(
                    driver,
                    self.prompt_area_selectors,
                    timeout=5.0,
                    cache_attr="_cached_prompt_selector",
                )
                if element is None:
                    raise RuntimeError(
                        "Could not find prompt input area after stale element during fill_input"
                    )
                continue

    def _click_send(self, driver: Any, input_el: Any) -> None:
        """Click the send button, or fall back to the Enter key.

        Retries stale element references and transient failures to avoid dropped
        prompts where the UI updates between finding and clicking the button.

        If the stop button is visible when this method is called (meaning a
        previous generation is still in progress), wait for it to disappear
        before attempting to find and click the send button.  This prevents the
        method from spending 30+ seconds polling for a send button that cannot
        appear until the running generation finishes.
        """
        # Pre-check: if the stop button is visible, the LLM is still generating
        # from a previous request.  Wait for it to finish first.
        if self._stop_button_present(driver):
            logger.info(
                "[selenium] _click_send: stop button visible (previous generation in progress), "
                "waiting for it to disappear before sending…"
            )
            wait_deadline = time.time() + 30.0
            while time.time() < wait_deadline:
                if not self._stop_button_present(driver):
                    logger.info("[selenium] _click_send: stop button disappeared, generation complete")
                    break
                time.sleep(0.5)
            else:
                logger.warning(
                    "[selenium] _click_send: stop button did not disappear within 30s — refreshing page"
                )
                if self._page_refresh_attempts < self._max_page_refresh_attempts:
                    try:
                        driver.refresh()
                    except Exception as refresh_err:
                        logger.warning(
                            "[selenium] _click_send driver.refresh() failed: %s",
                            refresh_err,
                        )
                    self._wait_for_page_ready(driver, timeout=30.0)
                    self._cached_prompt_selector = None
                    self._cached_send_selector = None
                    self._page_refresh_attempts += 1
                    logger.info(
                        "[selenium] _click_send: page refreshed (attempt %d/%d), retrying",
                        self._page_refresh_attempts,
                        self._max_page_refresh_attempts,
                    )
                    raise RuntimeError(
                        "page_refresh_required: stop button stuck before send — "
                        "page refreshed, retry without driver reset"
                    )
                else:
                    logger.warning(
                        "[selenium] _click_send: max page refresh attempts (%d) reached, proceeding anyway",
                        self._max_page_refresh_attempts,
                    )

        max_attempts = int(os.getenv("SELENIUM_SEND_CLICK_RETRIES", "1"))

        def _is_button_blacklisted(btn: Any) -> bool:
            for bl_sel in self.send_button_blacklist:
                try:
                    matches = driver.find_elements(By.CSS_SELECTOR, bl_sel)
                    if any(m == btn for m in matches):
                        logger.debug(
                            f"[selenium] Skipping blacklisted button for selector: {bl_sel}"
                        )
                        return True
                except Exception:
                    pass
            return False

        def _safe_click(btn: Any, selector: str) -> bool:
            if _is_button_blacklisted(btn):
                logger.debug(
                    f"[selenium] Skipping blacklisted button for selector: {selector}"
                )
                return False
            try:
                btn.click()
                logger.debug(f"[selenium] Sent via button click: {selector}")
                return True
            except Exception as e:
                logger.warning(f"[selenium] Button.click() failed for {selector}: {e}")
            try:
                driver.execute_script("arguments[0].click();", btn)
                logger.debug(f"[selenium] Sent via JS click: {selector}")
                return True
            except Exception as e:
                logger.warning(f"[selenium] JS click failed for {selector}: {e}")
            return False

        def _resolve_click_target(element: Any) -> Any:
            """If selector matches icon SVG or inner span, resolve up to a parent button or role='button'."""
            try:
                tag = (element.tag_name or "").lower()
                if tag in ("svg", "path", "span", "mat-icon", "i"):
                    # Climb up to find something that looks like a click target
                    parent = driver.execute_script(
                        "let el = arguments[0];"
                        "while (el && el.tagName !== 'BODY') {"
                        "  if (el.tagName === 'BUTTON' || el.getAttribute('role') === 'button' || el.onclick) return el;"
                        "  el = el.parentElement;"
                        "}"
                        "return arguments[0];",
                        element
                    )
                    return parent if parent else element
            except Exception:
                pass
            return element

        def _attempt_click() -> bool:
            ordered_send: list[str] = list(self.send_button_selectors)
            if self._cached_send_selector and self._cached_send_selector in ordered_send:
                ordered_send.remove(self._cached_send_selector)
                ordered_send.insert(0, self._cached_send_selector)

            for sel in ordered_send:
                try:
                    btn = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                    )
                    btn = _resolve_click_target(btn)
                    try:
                        if _safe_click(btn, sel):
                            self._cached_send_selector = sel
                            return True
                    except StaleElementReferenceException as e:
                        logger.warning(
                            f"[selenium] Stale send button for selector {sel}; invalidating cache and continuing: {e}"
                        )
                        if self._cached_send_selector == sel:
                            self._cached_send_selector = None
                        continue
                except StaleElementReferenceException as e:
                    logger.warning(
                        f"[selenium] Stale element in clickable wait for {sel}; continuing: {e}"
                    )
                    if self._cached_send_selector == sel:
                        self._cached_send_selector = None
                    continue
                except Exception as e:
                    logger.debug(f"[selenium] selector {sel} not clickable: {e}")

            for sel in ordered_send:
                try:
                    candidates = driver.find_elements(By.CSS_SELECTOR, sel)
                    for btn in candidates:
                        try:
                            resolved_btn = _resolve_click_target(btn)
                            if resolved_btn.is_displayed():
                                enabled = resolved_btn.is_enabled()
                                # Some frameworks use 'disabled' attribute on non-button elements
                                aria_disabled = resolved_btn.get_attribute("aria-disabled") == "true"
                                if enabled and not aria_disabled:
                                    if _safe_click(resolved_btn, sel):
                                        self._cached_send_selector = sel
                                        return True
                                else:
                                    logger.debug(f"[selenium] Button {sel} found but disabled (enabled={enabled}, aria_disabled={aria_disabled})")
                        except StaleElementReferenceException as e:
                            logger.warning(
                                f"[selenium] Stale send button in fallback for selector {sel}; continuing: {e}"
                            )
                            if self._cached_send_selector == sel:
                                self._cached_send_selector = None
                            continue
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(
                        f"[selenium] selector {sel} not found for fallback click: {e}"
                    )

            return False

        for attempt in range(max_attempts):
            if attempt > 0:
                logger.info(
                    f"[selenium] _click_send retry {attempt + 1}/{max_attempts}"
                )
                self._cached_prompt_selector = None
                self._cached_send_selector = None
                # Try to escape any popups or overlays that might be blocking the click
                try:
                    driver.execute_script("document.body.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));")
                    input_el.send_keys(Keys.ESCAPE)
                except Exception:
                    pass
                time.sleep(0.5)

            try:
                if _attempt_click():
                    return
            except StaleElementReferenceException as e:
                logger.warning(
                    f"[selenium] Stale element during send click attempt, retrying: {e}"
                )
                continue
            except Exception as e:
                logger.warning(
                    f"[selenium] Error during send click attempt, retrying: {e}"
                )
                continue

        # Final fallback: Enter key on the input element after retries.
        try:
            input_el.send_keys(Keys.RETURN)
            logger.debug("[selenium] Sent via Enter key")
            return
        except StaleElementReferenceException as e:
            logger.warning(
                f"[selenium] Input element stale on Enter key fallback; invalidating cache: {e}"
            )
            self._cached_prompt_selector = None
            self._cached_send_selector = None
            return
        except Exception as e:
            logger.error(f"[selenium] Could not send prompt: {e}")

    def _wait_for_send_ready(self, driver: Any, timeout: float = 10.0) -> bool:
        """Wait until the send button appears (LLM finished generating).

        Used by chunked sending - we only care that the LLM has finished generating,
        then we can fill the next chunk and click send when ready.
        Polls every 0.3 s up to *timeout* seconds.
        Env var: ``SELENIUM_SEND_READY_TIMEOUT`` overrides the default.
        """
        timeout = float(os.getenv("SELENIUM_SEND_READY_TIMEOUT", str(timeout)))
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Check: Send button is ready (LLM finished generating)
            for sel in self.send_button_selectors:
                try:
                    btns = driver.find_elements(By.CSS_SELECTOR, sel)
                    for btn in btns:
                        try:
                            if btn.is_displayed() and btn.is_enabled():
                                # Additional check: verify it's not a blacklisted element by checking attributes
                                aria_label = btn.get_attribute("aria-label") or ""
                                data_testid = btn.get_attribute("data-testid") or ""
                                tag = btn.tag_name or ""
                                
                                # Skip if it looks like stop/cancel/mic button
                                skip = False
                                for pattern in ["stop", "cancel", "mic", "voice"]:
                                    if pattern in aria_label.lower() or pattern in data_testid.lower():
                                        skip = True
                                        break
                                if tag == "mat-icon" and "mic" in (btn.get_attribute("fonticon") or "").lower():
                                    skip = True
                                    
                                if not skip:
                                    logger.debug(f"[selenium] Send button ready: {sel}")
                                    return True
                        except Exception:
                            pass
                except Exception as e:
                    if self._is_dead_session(e):
                        raise
            time.sleep(0.2)
        logger.warning(f"[selenium] Send button did not become ready within {timeout:.1f}s")
        return False

    def _is_same_site(self, current_url: str) -> bool:
        """Return True if *current_url* is on the same host as service_url.

        Some engines create a new conversation on send and navigate to a
        different path on the same host (e.g. from ``/c`` to ``/chat/<id>``).
        That in-site navigation is normal and must NOT be treated as a
        redirect-stall. A genuine stall sends the browser to a different host
        (e.g. an auth/login domain). Comparing only the host — instead of a
        full URL prefix — makes the check robust to per-conversation paths.
        """
        if not self.service_url or not current_url:
            return True
        try:
            base_host = urlparse(self.service_url).netloc
            cur_host = urlparse(current_url).netloc
        except Exception:
            return True
        if not base_host or not cur_host:
            return True
        return base_host == cur_host

    def _post_send_check(self, driver: Any, timeout: float = 15.0) -> bool:
        """Return True if the LLM accepted the prompt (stop button or new text appeared).

        After clicking send, poll for *timeout* seconds to confirm that either:
        - a stop-button becomes visible (streaming started), or
        - new response text different from the current baseline appears.

        If neither signal is seen by the deadline, inspects the current URL:
        - URL navigated to a different host → a redirect occurred → return False (stall detected).
        - URL still on the same host → model is just slow (or opened a new
          conversation path) → return True (let _wait_for_response decide).
        """
        baseline = self._get_latest_response_text(driver)
        # Reset the off-site flag: it is set only when a genuine cross-host
        # redirect is observed, so the caller can fail fast instead of retrying.
        self._last_send_offsite_redirect = False
        timeout = float(os.getenv("SELENIUM_POST_SEND_TIMEOUT", str(timeout)))
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Check stop button
            for sel in self.stop_selectors:
                try:
                    btns = driver.find_elements(By.CSS_SELECTOR, sel)
                    for b in btns:
                        try:
                            if b.is_displayed():
                                logger.debug("[selenium] post_send_check: stop button visible — send accepted")
                                return True
                        except Exception:
                            pass
                except Exception:
                    pass
            # Check new response text
            cur = self._get_latest_response_text(driver)
            if cur and cur != baseline:
                logger.debug("[selenium] post_send_check: new response text appeared — send accepted")
                return True
            # Fallback: if the send button has disappeared the LLM is generating.
            # Guard: if we are already on a redirect URL, don't interpret absent
            # send button as "generation started" — it means the page changed.
            _fb_url = ""
            try:
                _fb_url = driver.current_url or ""
            except Exception:
                pass
            if self.service_url and _fb_url and not self._is_same_site(_fb_url):
                logger.warning(
                    f"[selenium] post_send_check: off-site redirect detected ('{_fb_url}') — returning False"
                )
                self._last_send_offsite_redirect = True
                return False
            if not self._send_button_present(driver):
                if self._response_area_present(driver):
                    logger.debug(
                        "[selenium] post_send_check fallback: send button absent and response area detected — generation in progress"
                    )
                    return True
                logger.debug(
                    "[selenium] post_send_check fallback: send button absent but no response area detected yet"
                )
            time.sleep(0.2)

        # Timeout expired — check if we are still on the expected page
        cur_url = ""
        try:
            cur_url = driver.current_url or ""
        except Exception:
            pass

        if self.service_url and cur_url and not self._is_same_site(cur_url):
            logger.warning(
                f"[selenium] post_send_check: timeout with off-site URL '{cur_url}' — redirect-stall detected"
            )
            self._last_send_offsite_redirect = True
            return False

        logger.debug("[selenium] post_send_check: timeout but URL looks ok — assuming slow model")
        return True

    def _wait_for_response(self, driver: Any, max_wait: int = 120) -> str:
        """Wait for the LLM response to fully stream, then return its text.

        Strategy:
        1. Wait for the response container to appear and begin emitting text.
        2. After a 3 second grace delay, poll the container every 1 second.
        3. Track innerText length and childElementCount.
        4. If either metric changes, generation is still in progress.
        5. If both remain unchanged for 2 consecutive checks after activity starts,
           assume generation has finished.
        """
        self._click_accept_buttons(driver, timeout=2.0)

        # NOTE: A stop button visible at the start of _wait_for_response does NOT
        # by itself mean the page is stuck. Many engines (e.g. Gemini) render the
        # whole answer in one block after a "thinking" delay, so the stop button
        # legitimately stays visible for a while before any text appears.
        # Refreshing on a bare stop-button presence would throw away a response
        # that is still being generated and cause an infinite paste → send →
        # refresh loop.
        #
        # Stuck-page detection is therefore delegated entirely to the main
        # monitoring loop below, which distinguishes a genuine freeze from active
        # generation by watching the response text: it only refreshes when the
        # stop button is visible AND no text activity has occurred for the
        # engine's silent-freeze threshold (see `is_silent_freeze`).

        baseline = ""
        if self._use_baseline_comparison:
            baseline = self._get_latest_response_text(driver)
            logger.debug("[selenium] Baseline text length: %d", len(baseline))
        else:
            logger.debug("[selenium] Skipping baseline comparison for this engine")

        self._generation_was_active = False

        # Reset last_activity_time after potential refresh so the freeze detection
        # doesn't inherit stale timing from a previous generation cycle.
        last_activity_time = time.time()

        logger.debug("[selenium] watcher initial delay: 3.0s before response monitoring")
        time.sleep(3.0)

        _engine_max_wait = getattr(self, "_response_max_wait", None)
        effective_max_wait = int(_engine_max_wait) if _engine_max_wait else max_wait
        max_wait = int(os.getenv("SELENIUM_RESPONSE_MAX_WAIT", str(effective_max_wait)))
        wait_start = time.time()
        deadline = wait_start + max_wait

        previous_text_length = -1
        previous_child_count = -1
        stable_counter = 0
        iteration = 0
        last_container = None
        last_activity_time = time.time()
        current_text_length = 0
        current_child_count = 0

        # Wall-clock guard: a single slow WebDriver call (e.g. execute_script on
        # a page stuck in a loading state) can only be interrupted at the driver
        # command timeout, so the loop condition below may be checked well after
        # `deadline` has passed. This absolute cap ensures the watcher can never
        # run away far beyond its budget and leave an orphaned worker thread
        # spinning (which previously produced multi-minute "wait_for_response"
        # times). It grants a small grace window on top of max_wait.
        hard_cap = max_wait + max(30, int(max_wait * 0.5))

        while time.time() < deadline and (time.time() - wait_start) < hard_cap:
            iteration += 1
            container, selector = self._find_response_container_element(driver)
            if container is None:
                logger.debug(
                    "[selenium] watcher iteration=%d no response container element found",
                    iteration,
                )
                self._log_response_container_diagnostics(
                    None,
                    0,
                    0,
                    previous_text_length,
                    previous_child_count,
                    stable_counter,
                    iteration,
                )
            else:
                last_container = container
                current_text_length, current_child_count = self._get_response_container_stats(
                    driver, container
                )
                self._log_response_container_diagnostics(
                    selector,
                    current_text_length,
                    current_child_count,
                    previous_text_length,
                    previous_child_count,
                    stable_counter,
                    iteration,
                )

                if (
                    previous_text_length != -1 and previous_child_count != -1
                ):
                    if (
                        current_text_length != previous_text_length
                        or current_child_count != previous_child_count
                    ):
                        stable_counter = 0
                        self._generation_was_active = True
                        last_activity_time = time.time()
                    elif current_text_length > 0 or current_child_count > 0:
                        # Text is present and stable (not changing between iterations).
                        # This is the key condition: if text stays the same for 2+ iterations,
                        # generation is complete or stuck.
                        stable_counter += 1
                    else:
                        stable_counter = 0
                previous_text_length = current_text_length
                previous_child_count = current_child_count

                if stable_counter >= 2:
                    response = self._extract_response_text_from_element(driver, container)
                    if response:
                        logger.debug(
                            "[selenium] watcher detected stable response after %d iterations",
                            iteration,
                        )
                        return response

            # Early freeze detection — checked EVERY iteration regardless of container presence.
            # When the response container is not found (selectors don't match the stuck page),
            # the code used to loop forever because this detection was inside the `else` block.
            # Now it runs unconditionally.
            silent_freeze_threshold = self._silent_freeze_threshold
            if silent_freeze_threshold is None:
                silent_freeze_threshold = float(
                    os.getenv("SELENIUM_SILENT_FREEZE_THRESHOLD", "10")
                )
            is_silent_freeze = (
                self._stop_button_present(driver) 
                and (time.time() - last_activity_time) > silent_freeze_threshold
            )
            is_ui_stuck = (
                self._stop_button_present(driver) 
                and stable_counter >= 2 
                and (current_text_length > 0 or current_child_count > 0)
            )
            # No-progress freeze: some engines never expose a detectable stop
            # button (or hide it while "thinking"), so the two checks above can
            # stay False forever and the watcher would spin until the deadline.
            # If generation never started AND no response text/children have
            # appeared for well beyond the silent-freeze threshold, treat it as
            # a stall so the caller can recover instead of blocking the worker.
            no_progress_grace = max(silent_freeze_threshold * 2, 30.0)
            is_no_progress_freeze = (
                not self._generation_was_active
                and current_text_length == 0
                and current_child_count == 0
                and (time.time() - wait_start) > no_progress_grace
            )

            if is_silent_freeze or is_ui_stuck or is_no_progress_freeze:
                if is_silent_freeze:
                    freeze_reason = "silent freeze"
                elif is_ui_stuck:
                    freeze_reason = "UI stuck after completion"
                else:
                    freeze_reason = "no-progress freeze (no response detected)"
                logger.warning(
                    "[selenium] %s detected: Stop button visible but no proper UI transition (inactive %.1fs, text_len=%d, stable=%d)",
                    freeze_reason,
                    time.time() - last_activity_time,
                    current_text_length,
                    stable_counter,
                )
                # Attempt page refresh before giving up — preserves session/cookies
                if self._page_refresh_attempts < self._max_page_refresh_attempts:
                    refreshed = False
                    try:
                        logger.info(
                            "[selenium] Attempting page refresh for %s recovery "
                            "(attempt %d/%d)",
                            freeze_reason,
                            self._page_refresh_attempts + 1,
                            self._max_page_refresh_attempts,
                        )
                        driver.refresh()
                        self._wait_for_page_ready(driver, timeout=30.0)
                        # Invalidate cached selectors — DOM has changed after refresh
                        self._cached_prompt_selector = None
                        self._cached_send_selector = None
                        self._page_refresh_attempts += 1
                        refreshed = True
                    except Exception as refresh_err:
                        logger.error("[selenium] Page refresh failed: %s", refresh_err)
                        # Fall through to the original TimeoutException if refresh fails
                    # Raise the retry signal *outside* the try/except so it isn't
                    # swallowed by the ``except Exception`` above (which would
                    # otherwise degrade it into a plain TimeoutException and
                    # prevent the caller from retrying without a driver reset).
                    if refreshed:
                        raise RuntimeError(
                            "page_refresh_required: %s — page refreshed, retry without driver reset" % freeze_reason
                        )
                raise TimeoutException("%s detected during generation" % freeze_reason)

            error_text = self._get_error_indicator_text(driver)
            if error_text:
                try:
                    debug_mode.record_event(
                        "error",
                        getattr(self, "ENGINE_NAME", "default"),
                        stage="response_wait",
                        detail="error_indicator_detected",
                        text=error_text[:500],
                    )
                except Exception:
                    pass
                # A transient engine error (e.g. "Something went wrong (1076)")
                # leaves a toast/snackbar and, for chunked prompts, a partial
                # conversation on the page. Navigate back to the service home so
                # a fresh, empty chat is started (a plain refresh could restore
                # the partial conversation), then let the caller resend the
                # *entire* prompt from scratch (re-chunking if needed). This is
                # faster than a full driver reset and preserves the login
                # session/cookies.
                if self._page_refresh_attempts < self._max_page_refresh_attempts:
                    recovered = False
                    try:
                        logger.info(
                            "[selenium] Engine error indicator detected — "
                            "navigating to service home to recover "
                            "(attempt %d/%d): %s",
                            self._page_refresh_attempts + 1,
                            self._max_page_refresh_attempts,
                            error_text[:120],
                        )
                        driver.get(self.service_url)
                        self._wait_for_page_ready(driver, timeout=30.0)
                        self._cached_prompt_selector = None
                        self._cached_send_selector = None
                        self._page_refresh_attempts += 1
                        recovered = True
                    except Exception as nav_err:
                        logger.error(
                            "[selenium] Navigation to service home after engine "
                            "error failed: %s",
                            nav_err,
                        )
                    if recovered:
                        # Force a full prompt resend from the top of the flow so
                        # a chunked prompt is re-sent in its entirety rather than
                        # resuming mid-sequence on the fresh page.
                        self._skip_split_for_next = False
                        raise RuntimeError(
                            "page_refresh_required: engine error indicator "
                            "detected — returned to service home, resend full "
                            "prompt: %s" % error_text[:200]
                        )
                # Recovery budget exhausted (or navigation failed): fall back to
                # the driver-reset retry path so we do not loop forever on a page
                # that keeps showing the error toast.
                raise RuntimeError(
                    "selenium_response_detection_timeout: engine error indicator "
                    "detected during response wait: %s" % error_text[:200]
                )

            if self._is_limit_present(driver):
                raise RuntimeError(
                    "selenium_response_detection_timeout: service limit detected "
                    "during response wait (quota/rate-limit page shown)"
                )
            if self._is_captcha_present(driver):
                raise RuntimeError(
                    "selenium_response_detection_timeout: CAPTCHA detected "
                    "during response wait"
                )

            time.sleep(1.0)

        if last_container is not None:
            result = self._extract_response_text_from_element(driver, last_container)
        else:
            result = self._get_latest_response_text(driver)

        if result:
            logger.warning(
                "[selenium] Response wait timed out, returning best-effort result"
            )
            return result

        raise RuntimeError(
            "selenium_response_detection_timeout: no new response text appeared "
            "in expected selectors within the allotted time"
        )

    def _click_accept_buttons(self, driver: Any, timeout: float = 2.0) -> None:
        """Click any configured accept buttons that appear before continuing."""
        if not self.accept_button_selectors:
            return

        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            clicked_any = False
            for sel in self.accept_button_selectors:
                try:
                    buttons = driver.find_elements(By.CSS_SELECTOR, sel)
                except Exception:
                    continue

                for button in buttons:
                    try:
                        if button.is_displayed() and button.is_enabled():
                            button.click()
                            clicked_any = True
                            logger.debug(
                                f"[selenium] Clicked accept button with selector: {sel}"
                            )
                    except Exception:
                        continue

            if clicked_any:
                return
            time.sleep(0.25)

    # ------------------------------------------------------------------ /helpers

    async def stop(self) -> None:
        """Detach from the shared driver and persist cookies.

        The shared Chrome process is **not** quit here — other engines may
        still need it.  Call :func:`shutdown_shared_driver` (via
        ``EngineManager.stop_all``) to actually terminate Chrome.
        """

        def _sync_stop() -> None:
            if self.driver is not None:
                self._save_cookies()

        try:
            await asyncio.wait_for(asyncio.to_thread(_sync_stop), timeout=10)
        except TimeoutError:
            logger.warning("[selenium] stop() cookie-save timed out")
        except Exception as e:
            logger.warning(f"[selenium] stop() error: {e}")
        finally:
            self.driver = None
            self._initialized = False
            self._cookies_restored = False
