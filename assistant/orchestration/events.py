"""
Structured event emitter for the JARVIS Agentic Orchestration Layer.

Produces JSON-serializable event envelopes that record agent lifecycle steps,
tool executions, observations, and milestones. Compatible with the future
dashboard/projector visualizer without breaking existing GUI message queues.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from assistant.core.log import get

log = get("agent.events")

# Phase 1 Supported Event Types
EVENT_AGENT_STARTED = "agent_started"
EVENT_PLAN_UPDATED = "plan_updated"
EVENT_TOOL_STARTED = "tool_started"
EVENT_TOOL_COMPLETED = "tool_completed"
EVENT_OBSERVATION_RECEIVED = "observation_received"
EVENT_AGENT_COMPLETED = "agent_completed"
EVENT_AGENT_ERROR = "agent_error"


def make_event(
    event_type: str,
    agent: str = "local",
    goal: str = "",
    tool: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a canonical JSON-serializable agent event."""
    event: Dict[str, Any] = {
        "type": event_type,
        "agent": agent,
        "timestamp": time.time(),
        "iso_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    if goal:
        event["goal"] = goal
    if tool:
        event["tool"] = tool
    if metadata:
        event["metadata"] = metadata
    if error:
        event["error"] = error
    return event


def emit_event(bus: Any, event: Dict[str, Any]) -> None:
    """Safely publish an agent event to the bus GUI queue if available."""
    try:
        if bus is not None and hasattr(bus, "post_gui"):
            bus.post_gui("agent_event", event)
    except Exception as e:
        log.debug("[EVENT] Could not post event to GUI bus: %s", e)

