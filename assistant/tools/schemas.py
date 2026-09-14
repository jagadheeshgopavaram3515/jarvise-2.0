"""
Tool schemas — the declarative contract every desktop tool is described by.

This module is intentionally dependency-free (no app/file/browser imports) so
it can be imported anywhere — by the registry, the executor, the Gemini
tool-selection layer, and the tests — without pulling in psutil/playwright.

Two small dataclasses:

* ``ToolSpec``  — a tool's name, human description, JSON-schema parameters, the
  callable that runs it, and example phrases. The parameters block doubles as a
  Gemini function-declaration so the LLM can pick the tool and fill its args.
* ``ToolResult`` — the *single* shape every tool returns. The executor
  normalises whatever a handler returns into this, so callers (Dispatcher,
  router, tests) always see ``{"success": bool, "message": str, ...}``.

Designed to be reusable for Bluye later: nothing here is Jarvis-specific.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# A JSON-schema-style parameter block, e.g.
#   {"type": "object",
#    "properties": {"app": {"type": "string", "description": "App name"}},
#    "required": ["app"]}
ParamSchema = Dict[str, Any]


@dataclass
class ToolSpec:
    """Everything needed to describe, advertise and run one tool."""
    name: str
    description: str
    handler: Callable[..., Any]
    parameters: ParamSchema = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )
    examples: List[str] = field(default_factory=list)
    # Tools that only read / observe (safe). All current tools are read-or-launch
    # only; nothing mutates or deletes. Kept explicit so the allow-list and any
    # future confirmation policy can reason about a tool's blast radius.
    read_only: bool = True

    def to_function_declaration(self) -> Dict[str, Any]:
        """Render as a Gemini function-declaration dict (tool-calling schema)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def tool_result(
    success: bool,
    message: str,
    data: Optional[Any] = None,
    error: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build the canonical tool-result dict.

    Keys are stable so every caller can rely on them:
      success : bool  — did the tool do what was asked?
      message : str   — short, speakable, user-facing summary
      data    : any   — optional structured payload (paths, titles, lists…)
      error   : str   — optional machine-readable error string (None on success)
    """
    out: Dict[str, Any] = {"success": bool(success), "message": message}
    if data is not None:
        out["data"] = data
    if error is not None:
        out["error"] = error
    out.update(extra)
    return out


def ok(message: str, data: Optional[Any] = None, **extra: Any) -> Dict[str, Any]:
    """Shorthand for a successful result."""
    return tool_result(True, message, data=data, **extra)


def fail(message: str, error: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """Shorthand for a failed result."""
    return tool_result(False, message, error=error or message, **extra)
