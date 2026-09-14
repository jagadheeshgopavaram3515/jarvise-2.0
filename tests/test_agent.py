"""
Unit tests for JARVIS Phase 1 Local Agent, Inspection Tools, and Agent StateGraph.

All tests run deterministically with mocked Ollama/network where appropriate,
verifying:
1. AgentState creation and typing
2. Direct request bypass (chat/math bypasses agent)
3. Task classification (inspect/search/test tasks -> TASK)
4. Simple plan generation
5. Valid tool selection
6. Invalid tool rejection
7. Project path restriction (path traversal prevention)
8. read_file tool behavior
9. list_directory tool behavior
10. grep_code tool behavior
11. run_project_tests tool behavior
12. Tool result reaches observation in AgentState
13. Agent finishes and synthesizes output
14. Iteration limit (max_iterations) prevents infinite loop
15. Malformed Qwen output handling
16. Tool failure handling
17. No Gemini call occurs from Phase 1 agent
18. Streaming interface remains compatible with generator consumer
"""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.orchestration.agent import LocalAgent
from assistant.orchestration.events import (
    EVENT_AGENT_COMPLETED,
    EVENT_AGENT_STARTED,
    EVENT_TOOL_COMPLETED,
    EVENT_TOOL_STARTED,
    make_event,
)
from assistant.orchestration.graph import Orchestrator, build_agent_graph
from assistant.orchestration.router import QwenClassifier
from assistant.orchestration.state import AgentState
from assistant.tools import files, get_registry
from assistant.tools.executor import ToolExecutor


class TestPhase1Agent(unittest.TestCase):

    def setUp(self):
        self.registry = get_registry()
        self.executor = ToolExecutor(self.registry)

    # 1. AgentState creation
    def test_agent_state_creation(self):
        state: AgentState = {
            "user_text": "Inspect the router",
            "goal": "Inspect the router",
            "task_type": "TASK",
            "complexity": "LOW",
            "selected_agent": "local",
            "plan": [{"step": 1, "description": "Search code"}],
            "current_step": 1,
            "completed_steps": [],
            "pending_steps": ["Search code"],
            "tool_name": "grep_code",
            "tool_args": {"query": "QwenClassifier"},
            "tool_result": None,
            "observation": None,
            "errors": [],
            "retry_count": 0,
            "max_iterations": 8,
            "requires_replan": False,
            "final_response": None,
            "finished": False,
            "events": [],
        }
        self.assertEqual(state["task_type"], "TASK")
        self.assertEqual(state["max_iterations"], 8)
        self.assertFalse(state["finished"])

    # 2. Direct request bypass
    def test_direct_request_bypass(self):
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = ("LOCAL", 2.0)
        orchestrator = Orchestrator(classifier=mock_classifier)
        route, stats = orchestrator.select_route("Good morning Jarvis")
        self.assertEqual(route, "LOCAL")

    # 3. Task classification
    def test_task_classification(self):
        classifier = QwenClassifier()
        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b'{"response": "TASK"}'
            mock_resp.__enter__.return_value = mock_resp
            mock_url.return_value = mock_resp

            route, _ = classifier.classify("Inspect my JARVIS orchestration code")
            self.assertEqual(route, "TASK")

    # 4. Simple plan generation
    def test_plan_generation(self):
        agent = LocalAgent()
        with patch.object(agent, "_query_json") as mock_q:
            mock_q.return_value = {
                "plan": [
                    {"step": 1, "description": "Grep for timeout"},
                    {"step": 2, "description": "Read config.py"},
                ]
            }
            plan = agent.create_plan("Find timeout in config")
            self.assertEqual(len(plan), 2)
            self.assertEqual(plan[0]["step"], 1)

    # 5. Valid tool selection
    def test_valid_tool_selection(self):
        agent = LocalAgent(registry=self.registry, executor=self.executor)
        with patch.object(agent, "_query_json") as mock_q:
            mock_q.return_value = {
                "thought": "I will read config.py",
                "action": "tool",
                "tool_name": "read_file",
                "tool_args": {"path": "assistant/config.py", "limit": 20},
            }
            action = agent.select_action("Inspect config", [], [])
            self.assertEqual(action["action"], "tool")
            self.assertEqual(action["tool_name"], "read_file")
            self.assertIn("path", action["tool_args"])

    # 6. Invalid tool rejection
    def test_invalid_tool_rejection(self):
        agent = LocalAgent(registry=self.registry, executor=self.executor)
        with patch.object(agent, "_query_json") as mock_q:
            mock_q.return_value = {
                "thought": "Let me run bash",
                "action": "tool",
                "tool_name": "run_arbitrary_shell_command",
                "tool_args": {"cmd": "rm -rf"},
            }
            action = agent.select_action("Hack system", [], [])
            self.assertEqual(action["action"], "error")
            self.assertIn("does not exist", action["error"])

    # 7. Project path restriction
    def test_project_path_restriction(self):
        res = files.read_file("../../Windows/System32/drivers/etc/hosts")
        self.assertFalse(res["success"])
        self.assertEqual(res.get("error"), "path_traversal")

        res_dir = files.list_directory("C:/Windows")
        self.assertFalse(res_dir["success"])
        self.assertEqual(res_dir.get("error"), "path_traversal")

    # 8. read_file tool behavior
    def test_read_file(self):
        res = files.read_file("assistant/config.py", offset=1, limit=10)
        self.assertTrue(res["success"])
        self.assertIn("content", res["data"])
        self.assertLessEqual(res["data"]["limit"], 200)

    # 9. list_directory tool behavior
    def test_list_directory(self):
        res = files.list_directory("assistant/orchestration")
        self.assertTrue(res["success"])
        entries = res["data"]["entries"]
        names = [e["name"] for e in entries]
        self.assertIn("graph.py", names)
        self.assertIn("router.py", names)

    # 10. grep_code tool behavior
    def test_grep_code(self):
        res = files.grep_code("CLASSIFIER_SYSTEM_PROMPT", "assistant/orchestration")
        self.assertTrue(res["success"])
        matches = res["data"]["matches"]
        self.assertGreaterEqual(len(matches), 1)
        self.assertTrue(any("router.py" in m["file"] for m in matches))

    # 11. run_project_tests tool behavior
    def test_run_project_tests_validation(self):
        # Disallow command injection
        res = files.run_project_tests("test; rm -rf /")
        self.assertFalse(res["success"])
        self.assertEqual(res.get("error"), "invalid_target")

        # Disallow non-test targets
        res_disallowed = files.run_project_tests("assistant/config.py")
        self.assertFalse(res_disallowed["success"])
        self.assertEqual(res_disallowed.get("error"), "disallowed_target")

    # 12. Tool result reaches observation
    def test_tool_result_reaches_observation(self):
        agent = LocalAgent(registry=self.registry, executor=self.executor)
        raw_result = {"success": True, "message": "OK", "data": {"path": "a/b", "content": "class Foo:"}}
        obs = agent.format_observation("read_file", raw_result)
        self.assertIn("class Foo:", obs)
        self.assertIn("Read from a/b:", obs)

    # 13. Agent finishes
    def test_agent_finishes_and_synthesizes(self):
        agent = LocalAgent()
        with patch.object(agent, "_query_json") as mock_q:
            # First call: plan
            # Second call: finish
            mock_q.side_effect = [
                {"plan": [{"step": 1, "description": "Done"}]},
                {"action": "finish", "final_response": "I verified the routing system."},
            ]
            graph = build_agent_graph(agent)
            initial: AgentState = {
                "user_text": "Inspect system",
                "goal": "Inspect system",
                "task_type": "TASK",
                "max_iterations": 8,
            }
            final = graph.invoke(initial)
            self.assertTrue(final.get("finished"))
            self.assertIn("verified the routing system", final.get("final_response", ""))

    # 14. Iteration limit (max_iterations) prevents infinite loop
    def test_iteration_limit(self):
        agent = LocalAgent(registry=self.registry, executor=self.executor)
        with patch.object(agent, "_query_json") as mock_q:
            # Plan, then keep returning a tool action indefinitely
            mock_q.return_value = {
                "thought": "Keep looping",
                "action": "tool",
                "tool_name": "list_directory",
                "tool_args": {"path": "assistant"},
            }
            graph = build_agent_graph(agent)
            initial: AgentState = {
                "user_text": "Loop test",
                "goal": "Loop test",
                "task_type": "TASK",
                "max_iterations": 3,  # tight limit for test speed
            }
            final = graph.invoke(initial)
            self.assertTrue(final.get("finished"))
            self.assertGreaterEqual(final.get("retry_count", 0), 3)

    # 15. Malformed Qwen output
    def test_malformed_qwen_output(self):
        agent = LocalAgent()
        with patch.object(agent, "_query_json") as mock_q:
            mock_q.return_value = {"error": "malformed_json"}
            action = agent.select_action("Goal", [], [])
            self.assertEqual(action["action"], "error")

    # 16. Tool failure
    def test_tool_failure_in_observation(self):
        agent = LocalAgent(registry=self.registry, executor=self.executor)
        failed_res = {"success": False, "message": "File not found", "error": "not_found"}
        obs = agent.format_observation("read_file", failed_res)
        self.assertIn("failed", obs)
        self.assertIn("not_found", obs)

    # 17. No Gemini call from Phase 1 agent
    @patch("assistant.llm.gemini.GeminiClient.stream")
    def test_no_gemini_call_from_phase1_agent(self, mock_gemini_stream):
        agent = LocalAgent()
        with patch.object(agent, "_query_json") as mock_q:
            mock_q.side_effect = [
                {"plan": [{"step": 1, "description": "Done"}]},
                {"action": "finish", "final_response": "Done locally."},
            ]
            graph = build_agent_graph(agent)
            graph.invoke({"user_text": "Local task", "goal": "Local task"})
            mock_gemini_stream.assert_not_called()

    # 18. Streaming interface compatibility
    def test_streaming_interface_compatibility(self):
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = ("TASK", 2.0)
        mock_agent = MagicMock()
        mock_agent.create_plan.return_value = [{"step": 1, "description": "Inspect"}]
        mock_agent.select_action.return_value = {"action": "finish", "final_response": "The routing system is active."}
        mock_agent.synthesize_final_response.return_value = "The routing system is active."

        orchestrator = Orchestrator(classifier=mock_classifier, agent=mock_agent)
        chunks = list(orchestrator.stream("Inspect the routing"))
        self.assertIn("I'll inspect that for you, sir.", chunks)
        self.assertTrue(any("routing system" in c for c in chunks))

    # 19. Location tool registration
    def test_location_tool_registered(self):
        spec = self.registry.get_tool("location")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.name, "location")
        self.assertTrue(callable(spec.handler))

    # 20. Weather tool registration
    def test_weather_tool_registered(self):
        spec = self.registry.get_tool("weather")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.name, "weather")
        self.assertTrue(callable(spec.handler))

    # 21. Location tool execution
    def test_location_tool_execution(self):
        from assistant.tools.location import LocationProvider, set_location_provider
        class MockLocationProvider(LocationProvider):
            def get_location(self, purpose: str = ""):
                return {
                    "status": "SUCCESS",
                    "city": "Hyderabad",
                    "region": "Telangana",
                    "country": "India",
                    "latitude": 17.3850,
                    "longitude": 78.4867,
                    "accuracy": "city",
                    "source": "mock",
                }
        set_location_provider(MockLocationProvider())
        res = self.executor.execute("location", {})
        self.assertTrue(res["success"])
        self.assertIn("Hyderabad", res["message"])
        self.assertEqual(res["data"]["latitude"], 17.3850)

    # 22. Location failure handling
    def test_location_failure(self):
        from assistant.tools.location import LocationProvider, set_location_provider
        class FailingLocationProvider(LocationProvider):
            def get_location(self, purpose: str = ""):
                return {"status": "UNAVAILABLE", "error": "Network down"}
        set_location_provider(FailingLocationProvider())
        res = self.executor.execute("location", {})
        self.assertFalse(res["success"])
        self.assertEqual(res["error"], "location_unavailable")

    # 23. Weather tool execution with coordinates
    def test_weather_tool_coordinates(self):
        from assistant.tools.weather import WeatherProvider, set_weather_provider
        class MockWeatherProvider(WeatherProvider):
            def get_weather(self, latitude=None, longitude=None, city=""):
                return {
                    "status": "SUCCESS",
                    "location": "Hyderabad, Telangana",
                    "temperature_c": 29.5,
                    "condition": "Partly cloudy",
                    "humidity_percent": 60,
                    "wind_speed_kmh": 10.0,
                    "precipitation_mm": 0.0,
                    "source": "mock",
                }
        set_weather_provider(MockWeatherProvider())
        res = self.executor.execute("weather", {"latitude": 17.3850, "longitude": 78.4867})
        self.assertTrue(res["success"])
        self.assertIn("29.5°C", res["message"])
        self.assertIn("Partly cloudy", res["message"])

    # 24. Weather tool execution with explicit city
    def test_weather_tool_city(self):
        from assistant.tools.weather import WeatherProvider, set_weather_provider
        class MockWeatherProvider(WeatherProvider):
            def get_weather(self, latitude=None, longitude=None, city=""):
                return {
                    "status": "SUCCESS",
                    "location": f"{city}, India",
                    "temperature_c": 22.0,
                    "condition": "Clear sky",
                    "humidity_percent": 50,
                    "wind_speed_kmh": 8.0,
                    "precipitation_mm": 0.0,
                    "source": "mock",
                }
        set_weather_provider(MockWeatherProvider())
        res = self.executor.execute("weather", {"city": "Bengaluru"})
        self.assertTrue(res["success"])
        self.assertIn("22.0°C", res["message"])
        self.assertIn("Bengaluru", res["message"])

    # 25. Weather failure handling
    def test_weather_failure(self):
        from assistant.tools.weather import WeatherProvider, set_weather_provider
        class FailingWeatherProvider(WeatherProvider):
            def get_weather(self, latitude=None, longitude=None, city=""):
                return {"status": "ERROR", "error": "Service timeout"}
        set_weather_provider(FailingWeatherProvider())
        res = self.executor.execute("weather", {"city": "InvalidCity"})
        self.assertFalse(res["success"])
        self.assertEqual(res["error"], "Service timeout")

    # 26. Multi-tool dependency: location -> weather chaining
    def test_weather_location_chaining(self):
        from assistant.tools.location import LocationProvider, set_location_provider
        from assistant.tools.weather import WeatherProvider, set_weather_provider

        class MockLocation(LocationProvider):
            def get_location(self, purpose=""):
                return {"status": "SUCCESS", "city": "Hyderabad", "latitude": 17.385, "longitude": 78.486, "region": "TG", "country": "IN", "source": "mock"}

        class MockWeather(WeatherProvider):
            def get_weather(self, latitude=None, longitude=None, city=""):
                return {"status": "SUCCESS", "location": "Hyderabad", "temperature_c": 28.0, "condition": "Sunny"}

        set_location_provider(MockLocation())
        set_weather_provider(MockWeather())

        agent = LocalAgent(registry=self.registry, executor=self.executor)
        with patch.object(agent, "_query_json") as mock_q:
            mock_q.side_effect = [
                # 1. Plan
                {"plan": [{"step": 1, "description": "Get location"}, {"step": 2, "description": "Fetch weather"}]},
                # 2. Select location
                {"action": "tool", "tool_name": "location", "tool_args": {"purpose": "weather"}},
                # 3. Select weather using coordinates
                {"action": "tool", "tool_name": "weather", "tool_args": {"latitude": 17.385, "longitude": 78.486}},
                # 4. Finish
                {"action": "finish", "final_response": "The weather in Hyderabad is 28°C and sunny."},
            ]
            graph = build_agent_graph(agent)
            final = graph.invoke({"user_text": "What's the weather where I am?", "goal": "What's the weather where I am?"})
            self.assertTrue(final.get("finished"))
            self.assertEqual(len(final.get("completed_steps", [])), 2)
            self.assertIn("Ran location", final["completed_steps"][0])
            self.assertIn("Ran weather", final["completed_steps"][1])

    # 27. Weather with explicit city does not require location
    def test_explicit_city_weather_no_location(self):
        from assistant.tools.weather import WeatherProvider, set_weather_provider
        class MockWeather(WeatherProvider):
            def get_weather(self, latitude=None, longitude=None, city=""):
                return {"status": "SUCCESS", "location": city, "temperature_c": 18.0, "condition": "Cloudy"}
        set_weather_provider(MockWeather())

        agent = LocalAgent(registry=self.registry, executor=self.executor)
        with patch.object(agent, "_query_json") as mock_q:
            mock_q.side_effect = [
                {"plan": [{"step": 1, "description": "Get weather for London"}]},
                {"action": "tool", "tool_name": "weather", "tool_args": {"city": "London"}},
                {"action": "finish", "final_response": "The weather in London is 18°C and cloudy."},
            ]
            graph = build_agent_graph(agent)
            final = graph.invoke({"user_text": "What's the weather in London?", "goal": "What's the weather in London?"})
            self.assertTrue(final.get("finished"))
            self.assertEqual(len(final.get("completed_steps", [])), 1)
            self.assertIn("Ran weather", final["completed_steps"][0])

    # 28. Zero-result search recovery guidance in observation
    def test_zero_result_recovery_observation(self):
        agent = LocalAgent(registry=self.registry, executor=self.executor)
        zero_res = {"success": True, "message": "Found 0 matches for 'Quine classifier'", "data": {"query": "Quine classifier", "matches": []}}
        obs = agent.format_observation("grep_code", zero_res)
        self.assertIn("No matches found", obs)
        self.assertIn("SUGGESTION", obs)
        self.assertIn("typo", obs)

    # 29. Evidence grounding in synthesis
    def test_evidence_grounding_synthesis(self):
        agent = LocalAgent(registry=self.registry, executor=self.executor)
        with patch.object(agent, "_query_json") as mock_q:
            mock_resp = "I couldn't verify that from the inspected project files."
            with patch("urllib.request.urlopen") as mock_url:
                mock_http = MagicMock()
                mock_http.read.return_value = json.dumps({"response": mock_resp}).encode("utf-8")
                mock_http.__enter__.return_value = mock_http
                mock_url.return_value = mock_http

                resp = agent.synthesize_final_response("What database do we use?", ["No matches found for 'database'."])
                self.assertEqual(resp, mock_resp)

    # 30. Natural project questions routing to TASK
    def test_natural_project_questions_route_to_task(self):
        classifier = QwenClassifier()
        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b'{"response": "TASK"}'
            mock_resp.__enter__.return_value = mock_resp
            mock_url.return_value = mock_resp

            queries = [
                "How does my voice reach you?",
                "What components are handling my voice?",
                "What models am I using?",
                "Where is the Qwen classifier?",
                "What's the weather where I am?",
            ]
            for q in queries:
                route, _ = classifier.classify(q)
                self.assertEqual(route, "TASK", f"Query '{q}' should route to TASK")


if __name__ == "__main__":
    unittest.main()

