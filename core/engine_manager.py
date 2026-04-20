"""EngineManager — discovers, registers, and manages Selenium LLM engines.

Engines are loaded from the ``engines/`` directory (relative to this file's
project root) and can be defined in two ways:

JSON definition (recommended for simple engines)
    Any ``*.json`` file inside the ``engines/`` directory is loaded as a
    :class:`~core.json_engine.JsonEngine`.  See ``DEVELOPERS.md`` for the
    full schema.

Python definition (required for complex / custom logic)
    Any ``*.py`` file (excluding files whose name starts with ``_``) inside the
    ``engines/`` directory is imported dynamically.  The module must expose
    **exactly one** class that:

    * inherits from :class:`~core.selenium_llm_base.SeleniumLLMBase`
    * defines a class-level ``ENGINE_NAME: str`` attribute
    * optionally defines ``ENGINE_ALIASES: list[str]``

Both file types can coexist — a ``.py`` engine with the same name as a
``.json`` engine is ignored (JSON takes precedence).

If no ``engines/`` directory exists (or it is empty), the manager starts
with no engines registered and logs a warning.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from core.selenium_llm_base import SeleniumLLMBase

logger = logging.getLogger("engine_manager")

# Root of the project (parent of the ``core/`` package directory).
_PROJECT_ROOT = Path(__file__).parent.parent
_ENGINES_DIR = _PROJECT_ROOT / "engines"


# ---------------------------------------------------------------------------
# Descriptor (lightweight — no browser started)
# ---------------------------------------------------------------------------


@dataclass
class EngineDescriptor:
    """Metadata about an engine without instantiating it."""

    name: str
    aliases: list[str]
    display_name: str
    service_url: str
    models: dict[str, int]
    default_model: str
    source: str  # "json" | "python"
    source_path: str  # filesystem path
    allow_unlogged: bool = False
    notes: str | None = None
    # Maximum number of parallel workers for this engine.  1 = fully serial
    # (default).  Values > 1 are reserved for future parallel-session support;
    # the queue infrastructure already handles them correctly.
    max_workers: int = 1
    media_capabilities: list[str] = field(default_factory=list)
    media_support: dict[str, Any] = field(default_factory=dict)

    def limits_dict(self) -> dict:
        """Return interface-limits metadata without starting a browser."""
        model_name = self.default_model
        limit = self.models.get(model_name) or self.models.get("default") or 10000
        return {"max_prompt_chars": limit, "model_name": model_name}

    def supported_models_list(self) -> list[str]:
        """Return the list of model names known at descriptor level."""
        return list(self.models.keys())

    def to_dict(self) -> dict:
        data = {
            "name": self.name,
            "display_name": self.display_name,
            "aliases": self.aliases,
            "service_url": self.service_url,
            "models": self.models,
            "default_model": self.default_model,
            "allow_unlogged": self.allow_unlogged,
            "max_workers": self.max_workers,
            "source": self.source,
            "source_path": self.source_path,
        }
        if self.notes:
            data["notes"] = self.notes
        if self.media_support:
            data["media_support"] = self.media_support
        data["media_capabilities"] = list(self.media_capabilities)
        return data


# ---------------------------------------------------------------------------
# Engine scanning helpers
# ---------------------------------------------------------------------------


def _supports_media_capability(cfg_item: Any) -> bool:
    if not isinstance(cfg_item, dict):
        return False

    limits = cfg_item.get("limits")
    if isinstance(limits, dict):
        for value in limits.values():
            if value == -1:
                return True
            if isinstance(value, int) and value > 0:
                return True

    supported_models = cfg_item.get("supported_models")
    if isinstance(supported_models, list) and len(supported_models) > 0:
        return True

    return False


def _scan_json(path: Path) -> Optional[EngineDescriptor]:
    try:
        with path.open(encoding="utf-8") as fh:
            cfg = json.load(fh)
        name = cfg.get("name")
        media_support = cfg.get("media_support", {}) or {}
        if not name:
            logger.warning(f"[engine_manager] JSON engine without 'name': {path}")
            return None
        media_support = cfg.get("media_support", {}) or {}
        capabilities = [
            key
            for key in ("image", "audio", "document")
            if key in media_support and _supports_media_capability(media_support[key])
        ]
        return EngineDescriptor(
            name=name,
            aliases=list(cfg.get("aliases", [name])),
            display_name=cfg.get("display_name", name),
            service_url=cfg.get("service_url", ""),
            models=dict(cfg.get("models", {"default": 10000})),
            default_model=cfg.get("default_model", "default"),
            allow_unlogged=bool(cfg.get("allow_unlogged", False)),
            notes=cfg.get("notes"),
            max_workers=int(cfg.get("max_workers", 1)),
            source="json",
            source_path=str(path),
            media_capabilities=capabilities,
            media_support=media_support,
        )
    except Exception as exc:
        logger.warning(f"[engine_manager] Failed to scan JSON engine {path}: {exc}")
        return None


def _scan_python(path: Path) -> Optional[EngineDescriptor]:
    if path.name.startswith("_"):
        return None  # private / template files
    from core.selenium_llm_base import SeleniumLLMBase  # lazy import to avoid heavy deps at startup
    try:
        spec = importlib.util.spec_from_file_location(f"engines._dyn.{path.stem}", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj is not SeleniumLLMBase
                and issubclass(obj, SeleniumLLMBase)
                and hasattr(obj, "ENGINE_NAME")
            ):
                engine_name: str = obj.ENGINE_NAME
                return EngineDescriptor(
                    name=engine_name,
                    aliases=list(getattr(obj, "ENGINE_ALIASES", [engine_name])),
                    display_name=getattr(obj, "ENGINE_DISPLAY_NAME", engine_name),
                    service_url=getattr(obj, "ENGINE_SERVICE_URL", ""),
                    models=dict(getattr(obj, "ENGINE_MODELS", {"default": 10000})),
                    default_model=getattr(obj, "ENGINE_DEFAULT_MODEL", "default"),
                    allow_unlogged=bool(getattr(obj, "ENGINE_ALLOW_UNLOGGED", False)),
                    max_workers=int(getattr(obj, "ENGINE_MAX_WORKERS", 1)),
                    source="python",
                    source_path=str(path),
                )
        logger.warning(
            f"[engine_manager] No SeleniumLLMBase subclass with ENGINE_NAME in {path}"
        )
        return None
    except Exception as exc:
        logger.warning(f"[engine_manager] Failed to scan Python engine {path}: {exc}")
        return None


def scan_engines(engines_dir: Path) -> dict[str, EngineDescriptor]:
    """Scan *engines_dir* and return a name→descriptor mapping.

    JSON files take precedence over Python files with the same engine name.
    """
    descriptors: dict[str, EngineDescriptor] = {}

    if not engines_dir.is_dir():
        logger.warning(
            f"[engine_manager] engines/ directory not found at {engines_dir} — "
            "falling back to built-in engines"
        )
        return descriptors

    # --- pass 1: JSON ---
    for path in sorted(engines_dir.glob("*.json")):
        desc = _scan_json(path)
        if desc and desc.name not in descriptors:
            descriptors[desc.name] = desc
            for alias in desc.aliases:
                if alias != desc.name:
                    logger.debug(
                        f"[engine_manager] Registered JSON engine '{desc.name}' "
                        f"(alias: {alias})"
                    )
            logger.info(
                f"[engine_manager] Loaded JSON engine '{desc.name}' from {path.name}"
            )

    # --- pass 2: Python ---
    for path in sorted(engines_dir.glob("*.py")):
        desc = _scan_python(path)
        if desc and desc.name not in descriptors:
            descriptors[desc.name] = desc
            logger.info(
                f"[engine_manager] Loaded Python engine '{desc.name}' from {path.name}"
            )
        elif desc:
            logger.debug(
                f"[engine_manager] Python engine '{desc.name}' skipped "
                f"(already registered from JSON)"
            )

    return descriptors


# ---------------------------------------------------------------------------
# Engine instantiation
# ---------------------------------------------------------------------------


def _instantiate(descriptor: EngineDescriptor, **kwargs) -> "SeleniumLLMBase":
    """Create a live engine instance from its descriptor."""
    from core.selenium_llm_base import SeleniumLLMBase  # lazy import to avoid heavy deps at startup
    if descriptor.source == "json":
        from core.json_engine import JsonEngine

        return JsonEngine(Path(descriptor.source_path), **kwargs)

    if descriptor.source == "python":
        spec = importlib.util.spec_from_file_location(
            f"engines._dyn.{descriptor.name}", descriptor.source_path
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj is not SeleniumLLMBase
                and issubclass(obj, SeleniumLLMBase)
                and getattr(obj, "ENGINE_NAME", None) == descriptor.name
            ):
                return obj(**kwargs)
        raise RuntimeError(
            f"Python engine class for '{descriptor.name}' not found in "
            f"{descriptor.source_path}"
        )

    raise ValueError(f"Unknown engine source type: {descriptor.source!r}")


# ---------------------------------------------------------------------------
# FIFO queue helpers
# ---------------------------------------------------------------------------


@dataclass
class _PromptResult:
    """Result of a queued prompt job."""

    text: str
    model_name: str


@dataclass
class _PromptJob:
    """A single prompt task placed on a per-engine FIFO queue."""

    prompt: str
    future: asyncio.Future  # type: ignore[type-arg]
    media: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EngineManager singleton
# ---------------------------------------------------------------------------


class EngineManager:
    """Thread-safe singleton that manages engine lifecycle and discovery."""

    _instance: Optional["EngineManager"] = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        self.engines: dict[str, SeleniumLLMBase] = {}
        self.active_engine: Optional[SeleniumLLMBase] = None
        self.default_engine: str | None = None
        self._descriptors: dict[str, EngineDescriptor] = {}
        self._alias_map: dict[str, str] = {}  # alias → canonical name
        # Per-engine FIFO queues and their worker tasks.
        # Each engine gets exactly max_workers coroutine workers consuming
        # the queue; this serialises requests (max_workers=1) while
        # providing the scaffolding for future parallel-session support.
        self._job_queues: dict[str, asyncio.Queue[_PromptJob]] = {}
        self._queue_workers: dict[str, list[asyncio.Task]] = {}  # type: ignore[type-arg]
        self._load_descriptors()

    @classmethod
    def get(cls) -> "EngineManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = EngineManager()
        return cls._instance

    # ---------------------------------------------------------------------- discovery

    def _load_descriptors(self) -> None:
        """Scan engines/ and build the descriptor + alias maps."""
        self._descriptors.clear()
        self._alias_map.clear()

        raw = scan_engines(_ENGINES_DIR)
        if not raw:
            logger.warning(
                "[engine_manager] No engines found in engines/ — no engines registered"
            )

        for desc in raw.values():
            if desc.name in self._descriptors:
                continue  # already registered
            self._descriptors[desc.name] = desc
            for alias in desc.aliases:
                self._alias_map[alias.strip().lower()] = desc.name

    def reload_engines(self) -> list[dict]:
        """Re-scan the engines/ directory and refresh the registry.

        Already-running engine instances are **not** stopped — they
        remain available until :meth:`stop_all` is called.
        """
        logger.info("[engine_manager] Reloading engine registry…")
        self._load_descriptors()
        return self.list_engines()

    def list_engines(self) -> list[dict]:
        """Return metadata for all registered engines (no browser started)."""
        seen: set[str] = set()
        result = []
        for desc in self._descriptors.values():
            if desc.name not in seen:
                result.append(desc.to_dict())
                seen.add(desc.name)
        return result

    # ---------------------------------------------------------------------- access

    def _resolve(self, name: str) -> str:
        """Resolve an alias or name to the canonical engine name."""
        key = name.strip().lower()
        if key in self._alias_map:
            return self._alias_map[key]
        # Also try direct name match
        if key in self._descriptors:
            return key
        raise ValueError(f"Unknown engine: '{name}'")

    def get_engine(self, name: str) -> SeleniumLLMBase:
        """Return (and lazy-instantiate) the engine identified by *name* or an alias."""
        canonical = self._resolve(name)
        if canonical not in self.engines:
            desc = self._descriptors[canonical]
            logger.info(f"[engine_manager] Instantiating engine '{canonical}'…")
            self.engines[canonical] = _instantiate(desc)
        return self.engines[canonical]

    def set_active_engine(self, name: str) -> SeleniumLLMBase:
        engine = self.get_engine(name)
        self.active_engine = engine
        return engine

    def set_default_engine(self, name: str) -> str:
        canonical = self._resolve(name)
        if canonical not in self._descriptors:
            raise ValueError(f"Unknown engine: '{name}'")
        self.default_engine = canonical
        return canonical

    def get_default_engine(self) -> str:
        if self.default_engine and self.default_engine in self._descriptors:
            return self.default_engine
        if self._descriptors:
            # Use the first loaded engine as fallback
            return next(iter(self._descriptors))
        raise ValueError("No engines registered")

    def get_active_engine(self) -> SeleniumLLMBase:
        if not self.active_engine:
            raise RuntimeError("No active engine set")
        return self.active_engine

    def get_descriptor(self, name: str) -> Optional[EngineDescriptor]:
        """Return the descriptor for *name* (or an alias) without starting a browser."""
        try:
            canonical = self._resolve(name)
            return self._descriptors.get(canonical)
        except ValueError:
            return None

    # ---------------------------------------------------------------------- FIFO queue

    def _get_or_create_queue(self, canonical: str) -> asyncio.Queue[_PromptJob]:
        if canonical not in self._job_queues:
            self._job_queues[canonical] = asyncio.Queue()
        return self._job_queues[canonical]

    def _ensure_workers(self, canonical: str) -> None:
        """Start worker coroutines for *canonical* if they are not already running.

        ``max_workers`` from the descriptor controls the pool size.  With the
        default value of 1 the engine is fully serial: requests are processed
        one at a time in FIFO order.
        """
        desc = self._descriptors.get(canonical)
        target = max(1, desc.max_workers if desc else 1)

        existing = self._queue_workers.get(canonical, [])
        # Prune finished workers before counting live ones
        alive = [t for t in existing if not t.done()]
        self._queue_workers[canonical] = alive

        needed = target - len(alive)
        for _ in range(needed):
            task = asyncio.ensure_future(self._queue_worker_loop(canonical))
            self._queue_workers[canonical].append(task)

    async def _queue_worker_loop(self, engine_name: str) -> None:
        """Coroutine that processes prompt jobs for *engine_name* sequentially."""
        queue = self._get_or_create_queue(engine_name)
        while True:
            job = await queue.get()
            queued_at = getattr(job, "_queued_at", None)
            worker_start = __import__("time").time()
            if queued_at is not None:
                logger.info(
                    f"[timing] queue_wait ({engine_name}): "
                    f"{worker_start - queued_at:.2f}s"
                )
            try:
                engine = self.get_engine(engine_name)  # lazy-init browser here
                self.active_engine = engine
                try:
                    result_text = await engine.generate_response(job.prompt, job.media)
                except TypeError:
                    result_text = await engine.generate_response(job.prompt)
                model_name = engine.get_current_model()
                elapsed = __import__("time").time() - worker_start
                logger.info(
                    f"[timing] worker_total ({engine_name}): {elapsed:.2f}s"
                )
                if not job.future.done():
                    job.future.set_result(_PromptResult(text=result_text, model_name=model_name))
            except asyncio.CancelledError:
                if not job.future.done():
                    job.future.cancel()
                raise
            except Exception as exc:
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                queue.task_done()

    async def enqueue(
        self,
        engine_name: str,
        prompt: str,
        media: list[Any] | None = None,
    ) -> _PromptResult:
        """Submit *prompt* and optional media to the named engine's FIFO queue.

        The engine browser is started lazily by the worker, not by the HTTP
        handler.  Concurrent callers on the same engine are serialised
        automatically (max_workers=1 default).
        """
        canonical = self._resolve(engine_name)
        queue = self._get_or_create_queue(canonical)
        loop = asyncio.get_event_loop()
        future: asyncio.Future[_PromptResult] = loop.create_future()
        job = _PromptJob(prompt=prompt, future=future, media=media or [])
        job._queued_at = __import__("time").time()  # type: ignore[attr-defined]
        self._ensure_workers(canonical)
        await queue.put(job)
        return await future

    async def drain_queues(self) -> None:
        """Cancel all worker tasks and drain pending jobs from every queue.

        Called during reset so in-flight and queued requests are terminated
        cleanly before the engine instances are destroyed.
        """
        # Cancel all workers
        all_workers = [
            t for tasks in self._queue_workers.values() for t in tasks
        ]
        for t in all_workers:
            t.cancel()
        if all_workers:
            await asyncio.gather(*all_workers, return_exceptions=True)
        self._queue_workers.clear()

        # Drain remaining queued jobs (cancel their futures)
        for queue in self._job_queues.values():
            while not queue.empty():
                try:
                    job = queue.get_nowait()
                    if not job.future.done():
                        job.future.cancel()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
        self._job_queues.clear()

    # ---------------------------------------------------------------------- lifecycle

    async def stop_all(self) -> None:
        """Save cookies for every engine, then quit the shared Chrome driver."""
        for engine in self.engines.values():
            try:
                await engine.stop()
            except Exception:
                pass
        # Actually terminate the shared browser now that all engines detached.
        from core.selenium_llm_base import shutdown_shared_driver
        shutdown_shared_driver()
