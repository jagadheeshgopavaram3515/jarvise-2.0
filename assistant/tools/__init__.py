"""
Jarvis V2 — Desktop Tools Layer.

A small, reusable tool-calling framework that lets the assistant control the
desktop (apps), the file system (read-only) and the browser (Playwright,
DOM-only), selected via Gemini tool-calling. Designed to drop into Bluye later:
nothing here depends on Jarvis's voice pipeline except the optional ``router``
integration, which only uses ``bus.speak``.

Public surface
--------------
    get_registry()            -> the process-wide ToolRegistry (allow-list)
    get_executor()            -> the process-wide ToolExecutor (timeout+audit)
    ToolRegistry, ToolExecutor, ToolSpec, tool_result, ok, fail

The registry/executor are built lazily on first use and cached, so importing
this package is cheap and free of optional deps (psutil/playwright load only
when a tool that needs them actually runs).
"""
from __future__ import annotations

import threading
from typing import Optional

from assistant.tools.schemas import ToolSpec, fail, ok, tool_result  # noqa: F401

_registry = None          # type: Optional["ToolRegistry"]
_executor = None          # type: Optional["ToolExecutor"]
# Reentrant: get_executor() holds the lock and then calls get_registry(), which
# re-acquires it — a plain Lock would self-deadlock.
_lock = threading.RLock()


def get_registry():
    """Return the singleton ToolRegistry, building the default one on demand."""
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                from assistant.tools.registry import build_default_registry
                _registry = build_default_registry()
    return _registry


def get_executor():
    """Return the singleton ToolExecutor wired to the default registry."""
    global _executor
    if _executor is None:
        with _lock:
            if _executor is None:
                from assistant.tools.executor import ToolExecutor
                _executor = ToolExecutor(get_registry())
    return _executor


# Re-export the classes for callers/tests that build their own instances.
from assistant.tools.registry import ToolRegistry  # noqa: E402,F401
from assistant.tools.executor import ToolExecutor  # noqa: E402,F401

__all__ = [
    "get_registry", "get_executor",
    "ToolRegistry", "ToolExecutor", "ToolSpec",
    "tool_result", "ok", "fail",
]
