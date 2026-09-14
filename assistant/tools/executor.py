"""
ToolExecutor — the single, safe entry point for running any tool.

Everything funnels through ``ToolExecutor.execute(tool_name, args)``:

  * Allow-list (Phase 6): a name not in the registry is rejected outright —
    arbitrary command execution is impossible.
  * Timeout (Phase 1): each tool runs on a worker thread; if it overruns
    ``timeout`` seconds we return a clean failure instead of hanging the
    Dispatcher. (A genuinely runaway thread is left to die as a daemon — we
    never block the caller.)
  * Safe error handling: any exception inside a tool becomes a normal
    ``{"success": False, ...}`` result, never a crash that could take down the
    voice pipeline.
  * Logging (Phase 7): every call is written to ``logs/tools.log`` as
    ``timestamp tool args success duration``.

The result is ALWAYS the canonical dict from ``schemas`` —
``{"success": bool, "message": str, ...}`` — so callers never have to guess.
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict, Optional

from assistant import config
from assistant.core.log import get
from assistant.tools.registry import ToolRegistry
from assistant.tools.schemas import fail, tool_result

log = get("tools")


def _make_tools_logger() -> logging.Logger:
    """A dedicated file logger for logs/tools.log (separate from jarvis.log)."""
    logger = logging.getLogger("tools.audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    try:
        os.makedirs(os.path.dirname(config.TOOLS_LOG_FILE), exist_ok=True)
        handler = logging.FileHandler(config.TOOLS_LOG_FILE, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s",
                                               datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    except Exception as e:  # logging must never break tool execution
        log.warning("[TOOLS] could not open %s: %s", config.TOOLS_LOG_FILE, e)
    return logger


class ToolExecutor:
    """Runs registered tools with timeout, isolation and audit logging."""

    def __init__(self, registry: ToolRegistry, timeout: Optional[float] = None) -> None:
        self.registry = registry
        self.timeout = float(timeout if timeout is not None else config.TOOLS_TIMEOUT_S)
        self._audit = _make_tools_logger()
        # A small, reusable pool so we don't spin a fresh thread per call. Tools
        # are I/O-bound (launch a process, fetch a page), so a handful is plenty.
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")

    # ---------------------------------------------------------------- execute
    def execute(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run ``tool_name`` with ``args`` and return the canonical result dict.

        Never raises: a bad name, bad args, a tool exception or a timeout all
        come back as ``{"success": False, "message": ...}``.
        """
        args = dict(args or {})
        started = time.perf_counter()

        # ---- Allow-list enforcement (Phase 6) ------------------------------
        spec = self.registry.get_tool(tool_name)
        if spec is None:
            result = fail(f"I don't have a tool called '{tool_name}', sir.",
                          error="unknown_tool")
            self._audit_log(tool_name, args, result, time.perf_counter() - started)
            return result

        # ---- Run with a timeout on an isolated worker ----------------------
        try:
            future = self._pool.submit(self._invoke, spec.handler, args)
            raw = future.result(timeout=self.timeout)
            result = self._normalise(raw)
        except FutureTimeout:
            result = fail(f"The {tool_name} tool took too long and timed out, sir.",
                          error="timeout")
        except Exception as e:  # defensive: _invoke already guards, but be safe
            log.exception("[TOOLS] unexpected error in %s", tool_name)
            result = fail(f"The {tool_name} tool failed: {e}", error=str(e))

        self._audit_log(tool_name, args, result, time.perf_counter() - started)
        return result

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _invoke(handler, args: Dict[str, Any]) -> Any:
        """Call the handler with kwargs, falling back to a single positional arg.

        Tool handlers take named parameters (``open_app(app=...)``). If the LLM
        hands us a positional-only shape we still try to be forgiving.
        """
        try:
            return handler(**args)
        except TypeError:
            # Last-ditch: a one-arg handler called with a single value.
            if len(args) == 1:
                return handler(next(iter(args.values())))
            raise

    @staticmethod
    def _normalise(raw: Any) -> Dict[str, Any]:
        """Coerce whatever a handler returned into the canonical result dict."""
        if isinstance(raw, dict) and "success" in raw and "message" in raw:
            return raw
        if isinstance(raw, dict):
            # A dict missing the canonical keys — wrap it as data.
            msg = str(raw.get("message", "Done."))
            return tool_result(bool(raw.get("success", True)), msg, data=raw)
        if isinstance(raw, str):
            return tool_result(True, raw)
        if raw is None:
            return tool_result(True, "Done.")
        return tool_result(True, "Done.", data=raw)

    def _audit_log(self, tool: str, args: Dict[str, Any], result: Dict[str, Any],
                   duration: float) -> None:
        """Phase 7: append a structured line to logs/tools.log."""
        try:
            args_str = json.dumps(args, ensure_ascii=False, default=str)
        except Exception:
            args_str = str(args)
        success = bool(result.get("success"))
        line = (f"[TOOLS] {tool} {args_str} "
                f"success={str(success).lower()} duration={duration:.2f}s")
        try:
            self._audit.info(line)
        except Exception:
            pass
        # Also surface on the normal pipeline log at a glance.
        log.info(line)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
