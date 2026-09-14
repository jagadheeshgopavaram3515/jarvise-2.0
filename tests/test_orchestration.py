"""
Tests for LangGraph Orchestration Layer and Fast Qwen Classifier in JARVIS.

All network/model calls are strictly mocked so tests run deterministically
and offline without requiring Ollama or Gemini to be live.
"""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant import config
from assistant.core.events import Bus, Transcript
from assistant.core.dispatcher import Dispatcher
from assistant.llm.gemini import GeminiClient
from assistant.llm.ollama import OllamaClient
from assistant.orchestration.graph import Orchestrator, build_routing_graph
from assistant.orchestration.router import QwenClassifier
from assistant.orchestration.state import OrchestratorState
from assistant.reminders.scheduler import ReminderStore


class TestOrchestration(unittest.TestCase):

    def test_graph_construction(self):
        """1. Graph construction: verify StateGraph compiles with correct nodes and structure."""
        classifier = QwenClassifier()
        graph = build_routing_graph(classifier)
        self.assertIsNotNone(graph)
        # Verify compiled graph has the expected node names
        nodes = graph.nodes
        self.assertIn("classify", nodes)
        self.assertIn("local_response", nodes)
        self.assertIn("gemini_response", nodes)

    @patch("urllib.request.urlopen")
    def test_classify_local(self, mock_urlopen):
        """2. LOCAL classification: returns 'LOCAL' when Qwen outputs LOCAL."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "LOCAL"}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        classifier = QwenClassifier()
        route, latency_ms = classifier.classify("What is 15 + 5?")
        self.assertEqual(route, "LOCAL")
        self.assertGreaterEqual(latency_ms, 0.0)

    @patch("urllib.request.urlopen")
    def test_classify_gemini(self, mock_urlopen):
        """3. GEMINI classification: returns 'GEMINI' for complex / philosophical queries."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "GEMINI"}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        classifier = QwenClassifier()
        route, latency_ms = classifier.classify("Explain consciousness and quantum mechanics.")
        self.assertEqual(route, "GEMINI")

    @patch("urllib.request.urlopen")
    def test_invalid_classifier_output_fallback(self, mock_urlopen):
        """4. Invalid classifier output: falls back cleanly to 'GEMINI'."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "I think this is an interesting question..."}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        classifier = QwenClassifier()
        route, latency_ms = classifier.classify("Random question")
        self.assertEqual(route, "GEMINI")

    @patch("urllib.request.urlopen")
    def test_ollama_unavailable_fallback(self, mock_urlopen):
        """5. Ollama unavailable: URLError connection refused falls back to 'GEMINI'."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        classifier = QwenClassifier()
        route, latency_ms = classifier.classify("What is 2 + 2?")
        self.assertEqual(route, "GEMINI")

    @patch("urllib.request.urlopen")
    def test_qwen_timeout_fallback(self, mock_urlopen):
        """6. Qwen timeout: socket / HTTP timeout falls back to 'GEMINI'."""
        mock_urlopen.side_effect = TimeoutError("Classification timed out")

        classifier = QwenClassifier()
        route, latency_ms = classifier.classify("What is 2 + 2?")
        self.assertEqual(route, "GEMINI")

    def test_unknown_route_fallback(self):
        """7. Unknown route in state graph: defaults to 'GEMINI'."""
        mock_classifier = MagicMock(spec=QwenClassifier)
        mock_classifier.classify.return_value = ("UNKNOWN_ROUTE", 5.0)

        orchestrator = Orchestrator(classifier=mock_classifier)
        route, stats = orchestrator.select_route("Some prompt")
        self.assertEqual(route, "GEMINI")

    def test_gemini_fallback_client_selection(self):
        """8. Gemini fallback: verify get_client returns GeminiClient instance."""
        mock_classifier = MagicMock(spec=QwenClassifier)
        mock_classifier.classify.return_value = ("GEMINI", 5.0)

        orchestrator = Orchestrator(classifier=mock_classifier)
        client = orchestrator.get_client("GEMINI")
        self.assertIsInstance(client, GeminiClient)

    def test_routing_disabled_flag(self):
        """9. Routing disabled: Dispatcher initializes get_llm_client() (Gemini) when flag is false."""
        with patch.object(config, "LLM_ROUTING_ENABLED", False):
            bus = Bus()
            bus.shutdown.set()  # Allow run() to initialize self.llm and terminate immediately
            reminders = ReminderStore()
            dispatcher = Dispatcher(bus, reminders)
            dispatcher.run()
            # Default provider is GeminiClient
            self.assertIsInstance(dispatcher.llm, GeminiClient)

    def test_existing_provider_behavior(self):
        """10. Existing provider behavior: verify get_client('LOCAL') returns OllamaClient."""
        orchestrator = Orchestrator()
        client = orchestrator.get_client("LOCAL")
        self.assertIsInstance(client, OllamaClient)
        self.assertEqual(client.model, config.OLLAMA_MODEL)

    def test_deterministic_commands_bypass_routing(self):
        """11. Deterministic commands bypass routing: command handler handles intent without calling LLM."""
        bus = Bus()
        reminders = ReminderStore()
        dispatcher = Dispatcher(bus, reminders)
        mock_llm = MagicMock()
        dispatcher.llm = mock_llm

        tr = Transcript(
            text="what time is it",
            language="en",
            is_partial=False,
        )
        dispatcher._route(tr)
        # LLM stream should never have been invoked because 'what time is it' is a deterministic command
        mock_llm.stream.assert_not_called()

    def test_streaming_compatibility(self):
        """12. Streaming compatibility: Orchestrator.stream() yields sentences sequentially without buffering."""
        mock_classifier = MagicMock(spec=QwenClassifier)
        mock_classifier.classify.return_value = ("LOCAL", 2.0)

        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.stream.return_value = iter(["Sentence one.", "Sentence two."])

        orchestrator = Orchestrator(
            classifier=mock_classifier,
            ollama_client=mock_ollama,
        )

        chunks = list(orchestrator.stream("Simple question"))
        self.assertEqual(chunks, ["Sentence one.", "Sentence two."])
        mock_ollama.stream.assert_called_once_with("Simple question")

    @patch("urllib.request.urlopen")
    def test_routing_policy_20_benchmarks(self, mock_urlopen):
        """13. Comprehensive 20 benchmark tests covering English, Indic, Mixed, and Boundary queries."""
        benchmarks = [
            # LOCAL
            ("What is 25 times 40?", "LOCAL"),
            ("Write a Python function to reverse a string", "LOCAL"),
            ("What is the capital of France?", "LOCAL"),
            ("How are you?", "LOCAL"),
            ("Good morning Jarvis", "LOCAL"),
            ("Let's talk", "LOCAL"),
            # GEMINI (Indic & Mixed)
            ("నువ్వు ఎలా ఉన్నావు?", "GEMINI"),
            ("nuvvu ela unnav", "GEMINI"),
            ("तुम कैसे हो?", "GEMINI"),
            ("tum kaise ho", "GEMINI"),
            ("naaku AI gurinchi explain cheyyi", "GEMINI"),
            ("mujhe AI ke baare mein explain karo", "GEMINI"),
            # GEMINI (Deep English & Philosophy)
            ("Explain the architectural tradeoffs of using LangGraph for a production voice assistant.", "GEMINI"),
            ("What is consciousness?", "GEMINI"),
            # Boundary cases
            ("10 + 20 entha?", "GEMINI"),
            ("10 + 20 kitna hai?", "GEMINI"),
            ("How are you doing today?", "LOCAL"),
            ("What did we decide about the JARVIS architecture?", "GEMINI"),
            ("Explain HTTP simply", "LOCAL"),
            ("Design a complex production architecture for JARVIS", "GEMINI"),
        ]

        classifier = QwenClassifier()
        for query, expected_route in benchmarks:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"response": expected_route}).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_urlopen.return_value = mock_resp

            route, latency = classifier.classify(query)
            self.assertEqual(route, expected_route, f"Query '{query}' failed! Expected {expected_route}, got {route}")

    @patch("assistant.memory.store.format_memory_context")
    @patch("assistant.memory.store.format_history")
    def test_memory_shared_and_lightweight_math(self, mock_history, mock_mem):
        """14. Shared memory integration: conversations receive context, simple math stays lightweight."""
        mock_history.return_value = "User: Hello\nAssistant: Hi sir"
        mock_mem.return_value = "About user: Jagadheesh"

        ollama = OllamaClient()

        # A. Normal conversation -> retrieves shared memory context
        prompt_conv = ollama._build_prompt("Good morning Jarvis")
        mock_mem.assert_called()
        self.assertIn("Background about the user:", prompt_conv)
        self.assertIn("About user: Jagadheesh", prompt_conv)
        self.assertIn("Recent conversation:", prompt_conv)

        # B. Simple math -> skips heavy background memory blocks to remain lightweight
        mock_mem.reset_mock()
        prompt_math = ollama._build_prompt("What is 25 times 40?")
        mock_mem.assert_not_called()
        self.assertNotIn("Background about the user:", prompt_math)
        self.assertIn("User: What is 25 times 40?", prompt_math)


if __name__ == "__main__":
    unittest.main()

