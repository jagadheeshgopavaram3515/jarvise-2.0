"""
LangGraph Orchestration & Agentic Graph for JARVIS.

Constructs:
1. Routing Graph: Low-latency query classification between direct responses (LOCAL / GEMINI)
   and agentic tasks (TASK).
2. Agent Graph: Full autonomous local loop (understand -> plan -> select_tool ->
   execute_tool -> observe -> decide -> format_final) bounded by safety limits.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterator, Optional, Tuple

from langgraph.graph import END, START, StateGraph

from assistant.core.log import get
from assistant.llm.gemini import GeminiClient
from assistant.llm.ollama import OllamaClient
from assistant.orchestration.agent import LocalAgent
from assistant.orchestration.events import (
    EVENT_AGENT_COMPLETED,
    EVENT_AGENT_STARTED,
    EVENT_OBSERVATION_RECEIVED,
    EVENT_PLAN_UPDATED,
    EVENT_TOOL_COMPLETED,
    EVENT_TOOL_STARTED,
    emit_event,
    make_event,
)
from assistant.orchestration.router import QwenClassifier
from assistant.orchestration.state import AgentState, OrchestratorState

log = get("orchestrator")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")


def build_agent_graph(agent: LocalAgent, bus: Any = None):
    """Builds and compiles the Phase 1 Local Agent StateGraph:

    START -> understand -> plan -> select_tool -> execute_tool -> observe -> decide:
        continue -> select_tool (loop)
        finish   -> format_final -> END
    """
    workflow = StateGraph(AgentState)

    def understand_node(state: AgentState) -> dict:
        text = state.get("user_text", "")
        goal = state.get("goal") or text
        event = make_event(EVENT_AGENT_STARTED, agent="local", goal=goal)
        emit_event(bus, event)
        events = list(state.get("events") or [])
        events.append(event)
        return {
            "goal": goal,
            "selected_agent": "local",
            "task_type": "TASK",
            "complexity": "LOW",
            "retry_count": 0,
            "max_iterations": state.get("max_iterations") or 8,
            "finished": False,
            "events": events,
            "errors": list(state.get("errors") or []),
            "completed_steps": list(state.get("completed_steps") or []),
            "observations": list(state.get("observations") or []),
        }

    def plan_node(state: AgentState) -> dict:
        goal = state.get("goal", "")
        history = state.get("history")
        plan = agent.create_plan(goal, history=history)
        event = make_event(EVENT_PLAN_UPDATED, agent="local", goal=goal, metadata={"plan": plan})
        emit_event(bus, event)
        events = list(state.get("events") or [])
        events.append(event)
        pending = [s.get("description", f"Step {i+1}") for i, s in enumerate(plan)]
        return {
            "plan": plan,
            "current_step": 1,
            "pending_steps": pending,
            "events": events,
        }

    def select_tool_node(state: AgentState) -> dict:
        if state.get("finished"):
            return {}

        goal = state.get("goal", "")
        plan = state.get("plan", [])
        completed = state.get("completed_steps", [])
        obs = state.get("observation")
        errors = state.get("errors", [])
        history = state.get("history")

        decision = agent.select_action(
            goal=goal,
            plan=plan,
            completed_steps=completed,
            last_observation=obs,
            errors=errors,
            history=history,
        )

        action = decision.get("action", "")
        if action == "finish":
            return {
                "finished": True,
                "final_response": decision.get("final_response", ""),
                "tool_name": None,
                "tool_args": {},
            }
        elif action == "tool":
            return {
                "tool_name": decision.get("tool_name"),
                "tool_args": decision.get("tool_args", {}),
                "finished": False,
            }
        else:
            err_msg = decision.get("error", "Failed selecting valid tool")
            errs = list(state.get("errors") or [])
            errs.append(err_msg)
            return {
                "errors": errs,
                "tool_name": None,
                "tool_args": {},
            }

    def execute_tool_node(state: AgentState) -> dict:
        if state.get("finished"):
            return {}
        tool_name = state.get("tool_name")
        tool_args = state.get("tool_args") or {}

        events = list(state.get("events") or [])
        if not tool_name:
            return {"tool_result": None}

        event_start = make_event(EVENT_TOOL_STARTED, agent="local", tool=tool_name, metadata={"args": tool_args})
        emit_event(bus, event_start)
        events.append(event_start)

        res = agent.execute_tool(tool_name, tool_args)

        event_done = make_event(
            EVENT_TOOL_COMPLETED,
            agent="local",
            tool=tool_name,
            metadata={"success": res.get("success", False)},
        )
        emit_event(bus, event_done)
        events.append(event_done)

        return {
            "tool_result": res,
            "events": events,
        }

    def observe_node(state: AgentState) -> dict:
        if state.get("finished"):
            return {}
        tool_name = state.get("tool_name")
        res = state.get("tool_result")
        if not tool_name or res is None:
            return {}

        obs = agent.format_observation(tool_name, res)
        events = list(state.get("events") or [])
        event_obs = make_event(
            EVENT_OBSERVATION_RECEIVED,
            agent="local",
            tool=tool_name,
            metadata={"observation_len": len(obs)},
        )
        emit_event(bus, event_obs)
        events.append(event_obs)

        completed = list(state.get("completed_steps") or [])
        completed.append(f"Ran {tool_name}: {res.get('message', '')}")

        observations = list(state.get("observations") or [])
        observations.append(obs)

        retries = state.get("retry_count", 0) + 1
        return {
            "observation": obs,
            "observations": observations,
            "completed_steps": completed,
            "retry_count": retries,
            "events": events,
        }

    def decide_node(state: AgentState) -> dict:
        retries = state.get("retry_count", 0)
        max_iter = state.get("max_iterations", 8)

        if state.get("finished"):
            return {}

        if retries >= max_iter:
            log.warning("[AGENT] Iteration limit (%d) reached. Halting.", max_iter)
            return {"finished": True}

        return {}

    def decide_condition(state: AgentState) -> str:
        if state.get("finished", False):
            return "finish"
        if state.get("retry_count", 0) >= state.get("max_iterations", 8):
            return "finish"
        return "continue"

    def format_final_node(state: AgentState) -> dict:
        goal = state.get("goal", "")
        obs = list(state.get("observations") or [])
        if not obs:
            obs = list(state.get("completed_steps") or [])
        if not obs and state.get("observation"):
            obs = [state.get("observation")]
        candidate = state.get("final_response")

        final_resp = agent.synthesize_final_response(goal, obs, candidate)

        events = list(state.get("events") or [])
        event_end = make_event(EVENT_AGENT_COMPLETED, agent="local", goal=goal)
        emit_event(bus, event_end)
        events.append(event_end)

        return {
            "final_response": final_resp,
            "response": final_resp,
            "finished": True,
            "events": events,
        }

    workflow.add_node("understand", understand_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("select_tool", select_tool_node)
    workflow.add_node("execute_tool", execute_tool_node)
    workflow.add_node("observe", observe_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("format_final", format_final_node)

    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "plan")
    workflow.add_edge("plan", "select_tool")
    workflow.add_edge("select_tool", "execute_tool")
    workflow.add_edge("execute_tool", "observe")
    workflow.add_edge("observe", "decide")

    workflow.add_conditional_edges(
        "decide",
        decide_condition,
        {
            "continue": "select_tool",
            "finish": "format_final",
        },
    )
    workflow.add_edge("format_final", END)

    return workflow.compile()


def build_routing_graph(classifier: QwenClassifier, agent_graph: Any = None):
    """Builds and compiles the minimal LangGraph routing graph:

    START -> classify -> conditional_route:
        LOCAL  -> local_response -> END
        GEMINI -> gemini_response -> END
        TASK   -> task_agent -> END
    """
    workflow = StateGraph(OrchestratorState)

    def classify_node(state: OrchestratorState) -> dict:
        text = state.get("user_text", "")
        route, latency_ms = classifier.classify(text)
        return {"route": route, "error": None}

    def local_node(state: OrchestratorState) -> dict:
        return {"route": "LOCAL"}

    def gemini_node(state: OrchestratorState) -> dict:
        return {"route": "GEMINI"}

    def task_node(state: OrchestratorState) -> dict:
        return {"route": "TASK"}

    def route_condition(state: OrchestratorState) -> str:
        route = (state.get("route") or "GEMINI").upper()
        if "TASK" in route:
            return "task_agent"
        if "LOCAL" in route:
            return "local_response"
        return "gemini_response"

    workflow.add_node("classify", classify_node)
    workflow.add_node("local_response", local_node)
    workflow.add_node("gemini_response", gemini_node)
    workflow.add_node("task_agent", task_node)

    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        route_condition,
        {
            "local_response": "local_response",
            "gemini_response": "gemini_response",
            "task_agent": "task_agent",
        },
    )
    workflow.add_edge("local_response", END)
    workflow.add_edge("gemini_response", END)
    workflow.add_edge("task_agent", END)

    return workflow.compile()


class Orchestrator:
    """Orchestrates routing decisions via LangGraph and streams through the selected provider/agent."""

    def __init__(
        self,
        classifier: QwenClassifier | None = None,
        gemini_client: GeminiClient | None = None,
        ollama_client: OllamaClient | None = None,
        agent: LocalAgent | None = None,
        bus: Any = None,
    ):
        self.bus = bus
        self.classifier = classifier or QwenClassifier()
        self.gemini_client = gemini_client or GeminiClient()
        self.ollama_client = ollama_client or OllamaClient()
        self.agent = agent or LocalAgent()
        self.agent_graph = build_agent_graph(self.agent, bus=self.bus)
        self.graph = build_routing_graph(self.classifier, agent_graph=self.agent_graph)

    def select_route(self, user_text: str) -> Tuple[str, dict]:
        """Runs the LangGraph orchestration flow to select the target provider or task.

        Returns:
            (route, stats_dict)
        """
        t0 = time.perf_counter()
        initial_state: OrchestratorState = {
            "user_text": user_text,
            "goal": user_text,
            "route": None,
            "response": None,
            "error": None,
        }
        try:
            final_state = self.graph.invoke(initial_state)
            route = final_state.get("route") or "GEMINI"
        except Exception as e:
            log.warning("[ORCHESTRATOR] LangGraph invocation failed: %s, falling back to GEMINI", e)
            route = "GEMINI"

        total_routing_ms = (time.perf_counter() - t0) * 1000.0
        stats = {
            "route": route,
            "total_routing_ms": total_routing_ms,
        }
        return route, stats

    def get_client(self, route: str):
        """Returns the appropriate LLM client instance for direct routes."""
        if route == "LOCAL":
            return self.ollama_client
        return self.gemini_client

    def run_agent(self, user_text: str, history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, AgentState]:
        """Execute the Phase 1 Local Agent StateGraph synchronously."""
        if history is None:
            try:
                from assistant.memory import store
                history = store.load_history()[-4:]
            except Exception:
                history = []

        initial_state: AgentState = {
            "user_text": user_text,
            "goal": user_text,
            "task_type": "TASK",
            "selected_agent": "local",
            "history": history or [],
            "plan": [],
            "current_step": 0,
            "completed_steps": [],
            "pending_steps": [],
            "tool_name": None,
            "tool_args": {},
            "tool_result": None,
            "observation": None,
            "errors": [],
            "retry_count": 0,
            "max_iterations": 8,
            "requires_replan": False,
            "final_response": None,
            "finished": False,
            "events": [],
            "observations": [],
        }
        final_state = self.agent_graph.invoke(initial_state)
        resp = final_state.get("final_response") or "Task completed, sir."
        return resp, final_state

    def stream(self, user_text: str) -> Iterator[str]:
        """Classify route via LangGraph and stream sentences from the chosen provider or agent."""
        route, stats = self.select_route(user_text)
        log.info("[ORCHESTRATOR] Route selected: %s in %.1fms", route, stats["total_routing_ms"])

        if route == "TASK":
            # Progress lead-in for voice streaming
            yield "I'll inspect that for you, sir."
            resp_text, _ = self.run_agent(user_text)
            try:
                from assistant.memory import store
                store.add_exchange(user_text, resp_text)
            except Exception as e:
                log.warning("[ORCHESTRATOR] Failed saving TASK exchange to memory: %s", e)
            # Yield sentence by sentence to preserve TTS streaming
            parts = _SENTENCE_SPLIT.split(resp_text)
            for part in parts:
                s = part.strip()
                if s:
                    yield s
            return

        client = self.get_client(route)
        yield from client.stream(user_text)
