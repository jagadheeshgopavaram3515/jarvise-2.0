"""
Local Agent implementation for JARVIS using local Qwen (qwen2.5-coder:3b) via Ollama.

Executes autonomous inspection, search, and testing tasks within D:\\JARVIS
using the existing ToolRegistry and ToolExecutor.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from assistant import config
from assistant.core.log import get
from assistant.tools import get_executor, get_registry
from assistant.tools.executor import ToolExecutor
from assistant.tools.registry import ToolRegistry

log = get("local_agent")

LOCAL_AGENT_SYSTEM_PROMPT = """You are JARVIS's local task agent.
Your job is to accomplish the user's task using the available tools.

PROJECT ARCHITECTURE & SUBSYSTEMS:
- Root: D:\\JARVIS
- Main Python package: assistant/ (e.g. assistant/orchestration/, assistant/tools/, assistant/llm/, assistant/core/, assistant/config.py)
- Speech / Voice Input & Audio: assistant/stt/ (recognizer.py, cloud.py, faster-whisper), assistant/core/audio.py, WebRTC VAD
- Text-to-Speech Output: assistant/tts/
- Models & Configuration: assistant/config.py (OLLAMA_MODEL, GEMINI_MODEL), assistant/orchestration/router.py (QwenClassifier)
- Orchestration & Graphs: assistant/orchestration/ (graph.py, agent.py, router.py)
- Memory Engine: assistant/memory/
- Tools & Registry: assistant/tools/ (registry.py, executor.py, files.py, location.py, weather.py)
- Tests: tests/ (e.g. tests/test_agent.py, tests/test_orchestration.py)
- Entry point & Threading: main.py

AGENT PROCESS & RULES:
1. Understand the goal: Determine whether it asks about project files/code OR real-world tools (weather/location).
2. Location & Weather Chaining:
   - For weather at current location ("where I am", "here", "today"), if coordinates or city are unknown, first call `location`, then call `weather` with the returned coordinates or city.
   - For a specific city ("weather in Hyderabad"), directly call `weather` with {"city": "Hyderabad"} without calling location.
3. Code Discovery & Grounding:
   - To inspect code, use `grep_code`, `read_file`, `list_directory`, or `run_project_tests`.
   - Never search literal colloquialisms or STT typos (e.g. if user says "Quine", search "Qwen" or "classifier"; if user says "voice recognition", search "stt", "whisper", or read assistant/config.py).
   - If `grep_code` returns 0 matches, do NOT assume the component does not exist. Recover by searching broader keywords or inspecting assistant/config.py or main.py.
4. Strict Evidence Grounding:
   - Only state what is proven by actual tool observations. If uninspected or unverified, state: "I couldn't verify that from the inspected project files."
   - Do not invent hypothetical libraries or technologies.
   - In weather responses, speak the city/region and conditions naturally; do not recite raw GPS coordinates unless requested.
5. Action format:
   - Return JSON with action "tool" and tool_args, or action "finish" and final_response.
   - Never claim a tool was used when it was not. Never invent file contents or test results.
"""


class LocalAgent:
    """Autonomous local agent using Qwen for structured planning, tool selection, and synthesis."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        registry: ToolRegistry | None = None,
        executor: ToolExecutor | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout = float(timeout)
        self.registry = registry or get_registry()
        self.executor = executor or get_executor()

    def _query_json(self, prompt: str, system: str = LOCAL_AGENT_SYSTEM_PROMPT) -> Dict[str, Any]:
        """Query Ollama with format='json' and return the parsed dictionary."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "format": "json",
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": 0.1,
            },
        }

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "JARVIS-LocalAgent"},
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_text = data.get("response", "").strip()

        # Handle markdown fences if model included them despite json mode
        clean_text = raw_text
        if clean_text.startswith("```"):
            clean_text = re.sub(r"^```[a-zA-Z]*\n?", "", clean_text)
            clean_text = re.sub(r"\n?```$", "", clean_text).strip()

        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            log.warning("[AGENT] Malformed JSON from model: %s. Raw: %r", e, raw_text[:100])
            return {"error": "malformed_json", "raw": raw_text}

    def create_plan(self, goal: str, history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        """Create a concise 1-3 step execution plan for the goal."""
        hist_context = ""
        if history:
            turns = [f"{h.get('role', 'user').capitalize()}: {h.get('message', '')}" for h in history if h.get("message")]
            if turns:
                hist_context = "Recent conversation context:\n" + "\n".join(turns[-3:]) + "\n\n"

        prompt = (
            f"{hist_context}"
            f"Goal: {goal}\n\n"
            "Create a concise execution plan (1 to 3 steps) using the available tools:\n"
            "- For weather at current location: Step 1 determine location, Step 2 fetch weather for location.\n"
            "- For weather in explicit city: Step 1 fetch weather for city.\n"
            "- For project code/architecture/models/voice questions: Step 1 search or inspect relevant files (e.g. grep or read assistant/config.py, assistant/stt, assistant/orchestration), Step 2 synthesize grounded findings.\n\n"
            "Return JSON matching:\n"
            "{\n"
            '  "plan": [\n'
            '    {"step": 1, "description": "Search or inspect..."},\n'
            '    {"step": 2, "description": "Read file or summarize..."}\n'
            "  ]\n"
            "}"
        )

        try:
            res = self._query_json(prompt)
            plan = res.get("plan", [])
            if isinstance(plan, list) and plan:
                return plan
        except Exception as e:
            log.warning("[AGENT] Plan generation failed: %s, using fallback plan", e)

        # Robust fallback plan
        return [
            {"step": 1, "description": f"Inspect project for {goal}"},
            {"step": 1, "description": f"Execute inspection or tool for {goal}"},
            {"step": 2, "description": "Verify findings and summarize"},
        ]

    def select_action(
        self,
        goal: str,
        plan: List[Dict[str, Any]],
        completed_steps: List[str],
        last_observation: Optional[str] = None,
        errors: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Select the next tool action or decide to finish based on real observations."""
        # Available tools listing from the registry
        candidate_tools = ["location", "weather", "read_file", "list_directory", "grep_code", "run_project_tests"]
        tools_desc = []
        for name in candidate_tools:
            spec = self.registry.get_tool(name)
            if spec:
                props = (spec.parameters or {}).get("properties", {})
                args = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
                tools_desc.append(f"- {name}({args}): {spec.description}")

        tools_block = "\n".join(tools_desc)

        prompt_lines = []
        if history:
            turns = [f"{h.get('role', 'user').capitalize()}: {h.get('message', '')}" for h in history if h.get("message")]
            if turns:
                prompt_lines.append("Recent conversation context:\n" + "\n".join(turns[-3:]))

        prompt_lines.extend([
            f"Goal: {goal}",
            f"Plan: {json.dumps(plan)}",
            f"Completed Steps: {json.dumps(completed_steps)}",
            f"Last Observation: {last_observation or 'None'}",
            f"Errors Encountered: {json.dumps(errors or [])}",
            "\nAvailable Tools:",
            tools_block,
            "\nDecide the next action.",
            "If you have enough information to answer the goal, choose action 'finish'.",
            "Otherwise choose action 'tool'.",
            "\nReturn JSON matching EXACTLY one of:",
            '1. {"thought": "...", "action": "tool", "tool_name": "...", "tool_args": {...}}',
            '2. {"thought": "...", "action": "finish", "final_response": "Detailed factual answer..."}',
        ])

        prompt = "\n".join(prompt_lines)

        try:
            decision = self._query_json(prompt)
            if not isinstance(decision, dict):
                return {"action": "finish", "final_response": "I could not formulate a structured step."}

            action = decision.get("action", "").lower()
            if action == "tool":
                tool_name = decision.get("tool_name", "")
                tool_args = decision.get("tool_args", {})
                if not self.registry.get_tool(tool_name):
                    return {
                        "action": "error",
                        "error": f"Tool '{tool_name}' does not exist in registry.",
                    }
                if not isinstance(tool_args, dict):
                    tool_args = {}
                return {
                    "thought": decision.get("thought", ""),
                    "action": "tool",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                }
            elif action == "finish":
                return {
                    "thought": decision.get("thought", ""),
                    "action": "finish",
                    "final_response": decision.get("final_response", ""),
                }
            else:
                return {
                    "action": "error",
                    "error": f"Unknown action: {action}",
                }
        except Exception as e:
            log.warning("[AGENT] Action selection failed: %s", e)
            return {"action": "error", "error": str(e)}

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool safely through ToolExecutor."""
        t0 = time.perf_counter()
        res = self.executor.execute(tool_name, tool_args)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log.info("[AGENT] Executed %s in %.1fms (success=%s)", tool_name, elapsed_ms, res.get("success"))
        return res

    def format_observation(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Format raw tool result into a clean, bounded observation for Qwen."""
        success = result.get("success", False)
        msg = result.get("message", "")
        data = result.get("data")
        err = result.get("error")

        if tool_name == "run_project_tests" and isinstance(data, dict):
            status = "PASSED" if success else "FAILED"
            return f"Test Results for {data.get('target')} ({status}, exit code {data.get('exit_code')}):\n{data.get('output', '')[:1200]}"

        if tool_name == "location" and isinstance(data, dict):
            status = data.get("status", "SUCCESS" if success else "ERROR")
            if not success or status != "SUCCESS":
                return f"Location check failed: {err or msg}"
            return (
                f"Location Result: {data.get('city')}, {data.get('region')}, {data.get('country')} "
                f"(latitude: {data.get('latitude')}, longitude: {data.get('longitude')}, source: {data.get('source')})"
            )

        if tool_name == "weather" and isinstance(data, dict):
            status = data.get("status", "SUCCESS" if success else "ERROR")
            if not success or status != "SUCCESS":
                return f"Weather check failed: {err or msg}"
            return (
                f"Weather Result for {data.get('location')}: {data.get('temperature_c')}°C, {data.get('condition')}, "
                f"Humidity: {data.get('humidity_percent')}%, Wind: {data.get('wind_speed_kmh')} km/h, Precipitation: {data.get('precipitation_mm')} mm"
            )

        if not success:
            return f"Tool '{tool_name}' failed: {err or msg}"

        if tool_name == "read_file" and isinstance(data, dict):
            content = data.get("content", "")
            return f"Read from {data.get('path')}:\n{content[:1200]}"

        if tool_name == "list_directory" and isinstance(data, dict):
            entries = data.get("entries", [])
            items_str = ", ".join(f"{e['name']}{'/' if e['type'] == 'dir' else ''}" for e in entries[:25])
            return f"Directory {data.get('path')} contains {len(entries)} items: {items_str}"

        if tool_name == "grep_code" and isinstance(data, dict):
            matches = data.get("matches", [])
            if not matches:
                return (
                    f"No matches found for '{data.get('query')}'. "
                    f"SUGGESTION: The search term may have an STT typo or be too specific. "
                    f"Try searching broader terms (e.g. 'stt', 'audio', 'whisper', 'model', 'classifier'), or check assistant/config.py or main.py."
                )
            match_lines = [f"{m['file']}:{m['line']} -> {m['content']}" for m in matches[:10]]
            return f"Found {len(matches)} matches:\n" + "\n".join(match_lines)

        # General payload
        return f"{msg}\nData: {str(data)[:600]}"

    def synthesize_final_response(
        self,
        goal: str,
        observations: List[str],
        candidate_response: Optional[str] = None,
    ) -> str:
        if candidate_response and len(candidate_response.strip()) > 10 and "I don't know" not in candidate_response:
            return candidate_response.strip()

        obs_text = "\n\n".join(f"Observation {i+1}:\n{obs}" for i, obs in enumerate(observations[-4:]))
        prompt = (
            f"User Task: {goal}\n\n"
            f"Factual Tool Observations:\n{obs_text}\n\n"
            "Provide a concise, direct, helpful spoken response answering the user's task "
            "based strictly on the observations above. Do not mention JSON or internal tool details."
            "Instructions:\n"
            "1. Answer based STRICTLY on the tool observations above. Never invent libraries, frameworks, or facts not proven by the observations.\n"
            "2. For project code: if the inspected files did not verify the requested component, say: 'I couldn't verify that from the inspected project files.'\n"
            "3. For weather and location: speak the location name, temperature, and conditions naturally. Do not recite raw GPS latitude/longitude coordinates unless explicitly asked.\n"
            "4. Provide a concise, direct, helpful spoken response without mentioning JSON or internal tool mechanics."
        )

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 250},
            }
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "JARVIS-LocalAgent"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data.get("response", "").strip()
                if reply:
                    return reply
        except Exception as e:
            log.warning("[AGENT] Final answer synthesis failed: %s", e)

        # Fallback to last observation if synthesis fails
        if observations:
            return f"I completed the inspection, sir. {observations[-1][:300]}"
        return "I completed the requested task, sir."
