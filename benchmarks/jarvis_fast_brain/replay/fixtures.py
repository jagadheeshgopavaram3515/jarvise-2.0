from __future__ import annotations

import copy
from typing import Any, Mapping


class FixtureExecutor:
    """Deterministic lookup-only executor; it never imports or calls tools."""

    def __init__(self, fixtures: Mapping[str, Any] | None = None):
        self.fixtures = copy.deepcopy(dict(fixtures or {}))
        self.calls: list[dict[str, Any]] = []

    def execute(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        args = dict(arguments or {})
        self.calls.append({"name": name, "arguments": copy.deepcopy(args)})
        entries = self.fixtures.get("tool_results", {})
        key = name
        if isinstance(entries, list):
            for item in entries:
                if item.get("name") == name and item.get("arguments", {}) == args:
                    return copy.deepcopy(item.get("result", {"success": False, "error": "missing_fixture"}))
        if isinstance(entries, Mapping) and key in entries:
            return copy.deepcopy(entries[key])
        return {"success": False, "error": "missing_fixture", "message": f"No fixture for {name}"}
