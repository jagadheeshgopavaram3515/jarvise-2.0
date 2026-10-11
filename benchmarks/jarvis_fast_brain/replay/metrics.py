from __future__ import annotations

from statistics import mean
from typing import Any


def failure_category(result: dict[str, Any], expected: dict[str, Any]) -> str | None:
    status = result.get("status")
    if status in {"OLLAMA_TIMEOUT", "OLLAMA_ERROR", "PARSER_ERROR"}:
        return status
    parsed = result.get("parsed_output") or {}
    if expected.get("requires_clarification") and parsed.get("action") != "clarify":
        return "CLARIFICATION_ERROR"
    if expected.get("selected_capability") and parsed.get("tool_name") != expected["selected_capability"]:
        return "CAPABILITY_MISMATCH"
    if expected.get("arguments") and parsed.get("tool_args") != expected["arguments"]:
        return "QWEN_ARGUMENT_ERROR"
    return None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in results if r.get("status") not in {"NOT_IMPLEMENTED"}]
    latencies = [float(r.get("latency_ms", 0)) for r in completed]
    return {"cases": len(results), "completed": len(completed),
            "status_counts": {s: sum(r.get("status") == s for r in results) for s in sorted({r.get("status") for r in results})},
            "latency_ms": {"mean": mean(latencies) if latencies else None,
                           "min": min(latencies) if latencies else None,
                           "max": max(latencies) if latencies else None}}
