"""
LangGraph Orchestration package for JARVIS.
LangGraph Orchestration & Agent package for JARVIS.
"""
from assistant.orchestration.graph import Orchestrator, build_routing_graph
from assistant.orchestration.agent import LocalAgent
from assistant.orchestration.graph import Orchestrator, build_agent_graph, build_routing_graph
from assistant.orchestration.router import QwenClassifier
from assistant.orchestration.state import OrchestratorState
from assistant.orchestration.state import AgentState, OrchestratorState

_orchestrator_instance: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    """Singleton / factory accessor for the LangGraph Orchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator()
    return _orchestrator_instance


__all__ = [
    "AgentState",
    "LocalAgent",
    "Orchestrator",
    "OrchestratorState",
    "QwenClassifier",
    "build_agent_graph",
    "build_routing_graph",
    "get_orchestrator",
]

