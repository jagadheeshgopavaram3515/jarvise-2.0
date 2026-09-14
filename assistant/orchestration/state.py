"""
State definition for the JARVIS LangGraph Orchestration Layer.
State definition for the JARVIS LangGraph Orchestration & Agent Layer.
"""
from typing import Optional, TypedDict
from typing import Any, Dict, List, Optional, TypedDict


class OrchestratorState(TypedDict, total=False):
    """Minimal state schema for the routing graph."""
class AgentState(TypedDict, total=False):
    """Typed state schema for the JARVIS Agentic loop and routing graph."""
    # Input & Goal
    user_text: str
    route: Optional[str]      # "LOCAL" or "GEMINI"
    response: Optional[str]   # Final response if populated
    error: Optional[str]      # Error description if any
    goal: str
    task_type: str               # "DIRECT" or "TASK"
    complexity: str              # "LOW", "MEDIUM", "HIGH"
    route: Optional[str]         # "LOCAL" or "GEMINI" (for direct routing compatibility)
    history: Optional[List[Dict[str, str]]]  # Multi-turn conversation context for follow-ups

    # Agent selection
    selected_agent: str          # "local" (Phase 1)

    # Planning
    plan: List[Dict[str, Any]]   # [{"step": 1, "description": "...", "status": "pending"}]
    current_step: int
    completed_steps: List[str]
    pending_steps: List[str]

    # Tool Execution & Observation
    tool_name: Optional[str]
    tool_args: Dict[str, Any]
    tool_result: Optional[Dict[str, Any]]
    observation: Optional[str]
    observations: List[str]

    # Control & Safety
    errors: List[str]
    retry_count: int
    max_iterations: int
    requires_replan: bool

    # Output & Termination
    final_response: Optional[str]
    response: Optional[str]      # Alias for backward compatibility
    error: Optional[str]         # Alias for backward compatibility
    finished: bool

    # Structured Audit / Visualizer Events
    events: List[Dict[str, Any]]


# Backward compatibility alias
OrchestratorState = AgentState
