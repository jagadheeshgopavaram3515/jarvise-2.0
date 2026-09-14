"""
Resilience toolkit (Phase 6/7) — small, dependency-free safety wrappers.

These are the building blocks for "no uncaught exception ever escapes": wrap a
risky call (network, filesystem, JSON, subprocess, embeddings, browser) and get
a graceful fallback instead of a crash. They are ADDITIVE helpers — existing hot
paths already carry their own try/except; new code uses these, and they're here
for any gap that needs closing without touching protected modules.

Logging levels (Phase 7) follow stdlib semantics:
    INFO     normal operation        ERROR     a request/operation failed
    WARNING  degraded but continuing  CRITICAL  a service died (triggers recovery)

Nothing here imports a Jarvis subsystem, so it can be used from anywhere
(including Bluye) without circular imports.
"""
from __future__ import annotations

import functools
import json
import os
import threading
import time
from typing import Any, Callable, Iterable, Optional, Tuple, Type

from assistant.core.log import get

log = get("resilience")

# A broad-but-sane default: catch ordinary errors, but let control-flow
# exceptions (KeyboardInterrupt, SystemExit, GeneratorExit) propagate so the
# process can still be stopped.
_DEFAULT_EXC: Tuple[Type[BaseException], ...] = (Exception,)


def safe_call(fn: Callable[..., Any], *args: Any,
              default: Any = None,
              label: str = "",
              exceptions: Tuple[Type[BaseException], ...] = _DEFAULT_EXC,
              level: str = "warning",
              **kwargs: Any) -> Any:
    """Call ``fn(*args, **kwargs)``; on failure log and return ``default``.

    Never raises (for the listed exception types). Use for any one-off risky
    call where a sensible fallback exists.
    """
    try:
        return fn(*args, **kwargs)
    except exceptions as e:  # noqa: BLE001 - intentional broad guard
        _log_at(level, "[SAFE] %s failed: %s", label or getattr(fn, "__name__", "call"), e)
        return default


def safe(default: Any = None, label: str = "",
         exceptions: Tuple[Type[BaseException], ...] = _DEFAULT_EXC,
         level: str = "warning") -> Callable:
    """Decorator form of :func:`safe_call` — wraps a function so it can never
    raise the listed exceptions, returning ``default`` instead."""
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except exceptions as e:  # noqa: BLE001
                _log_at(level, "[SAFE] %s failed: %s", label or fn.__name__, e)
                return default
        return wrapper
    return deco


def retry(fn: Callable[..., Any], *args: Any,
          attempts: int = 3,
          delay: float = 0.5,
          backoff: float = 2.0,
          exceptions: Tuple[Type[BaseException], ...] = _DEFAULT_EXC,
          default: Any = None,
          label: str = "",
          **kwargs: Any) -> Any:
    """Call ``fn`` with retries + exponential backoff; return ``default`` if all
    attempts fail. For transient faults (network blips, locked files)."""
    wait = delay
    last = None
    for i in range(1, max(1, attempts) + 1):
        try:
            return fn(*args, **kwargs)
        except exceptions as e:  # noqa: BLE001
            last = e
            log.warning("[RETRY] %s attempt %d/%d failed: %s",
                        label or getattr(fn, "__name__", "call"), i, attempts, e)
            if i < attempts:
                time.sleep(wait)
                wait *= backoff
    log.error("[RETRY] %s gave up after %d attempts: %s",
              label or getattr(fn, "__name__", "call"), attempts, last)
    return default


# --------------------------------------------------------------------------- #
# JSON / filesystem helpers — never crash a caller on a corrupt or missing file.
# --------------------------------------------------------------------------- #
def safe_json_load(path: str, default: Any = None) -> Any:
    """Load JSON from ``path``; return ``default`` on missing/corrupt/locked."""
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        log.warning("[JSON] load failed for %s: %s", path, e)
        return default


def safe_json_dump(obj: Any, path: str, *, atomic: bool = True) -> bool:
    """Write ``obj`` as JSON to ``path``. Returns True on success, False on
    failure (never raises). Atomic write avoids a half-written file on crash."""
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if atomic:
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        return True
    except (OSError, TypeError, ValueError) as e:
        log.error("[JSON] dump failed for %s: %s", path, e)
        return False


# --------------------------------------------------------------------------- #
# Supervised loop — the core of thread survivability (Phase 2).
# --------------------------------------------------------------------------- #
def supervise_loop(name: str,
                   stop: Callable[[], bool],
                   body: Callable[[], None],
                   *,
                   on_error: Optional[Callable[[BaseException], None]] = None,
                   idle_sleep: float = 0.0) -> None:
    """Run ``body()`` repeatedly until ``stop()`` is True, swallowing *every*
    ordinary exception so the calling thread can never die from a bad iteration.

    This is the canonical "while running: try: process() except: log; continue"
    pattern, factored out so each service can adopt it in one line.

        supervise_loop("STT", bus.shutdown.is_set, self._tick)
    """
    log.info("[SUPERVISE] %s loop started", name)
    while not stop():
        try:
            body()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:  # noqa: BLE001 - the whole point: never die
            log.exception("[SUPERVISE] %s iteration error — skipped, loop alive", name)
            if on_error is not None:
                try:
                    on_error(e)
                except Exception:
                    log.debug("[SUPERVISE] %s on_error hook failed", name, exc_info=True)
            if idle_sleep:
                time.sleep(idle_sleep)
    log.info("[SUPERVISE] %s loop exited (stop requested)", name)


def _log_at(level: str, msg: str, *args: Any) -> None:
    getattr(log, level if level in {"debug", "info", "warning", "error", "critical"}
            else "warning")(msg, *args)


__all__ = [
    "safe", "safe_call", "retry",
    "safe_json_load", "safe_json_dump",
    "supervise_loop",
]
