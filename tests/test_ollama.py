"""
Unit tests for the Ollama / Qwen integration in JARVIS.
Verifies client configuration, factory defaults, streaming, error handling,
and non-breaking compatibility.
"""
import io
import json
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from assistant import config
from assistant.llm import get_llm_client, GeminiClient, OllamaClient


class TestOllamaIntegration(unittest.TestCase):

    def test_default_llm_provider_is_gemini(self):
        """Verify that get_llm_client() defaults to GeminiClient without breaking existing pipeline."""
        client = get_llm_client()
        self.assertIsInstance(client, GeminiClient)

    def test_explicit_ollama_provider_factory(self):
        """Verify that get_llm_client('ollama') produces an OllamaClient."""
        client = get_llm_client("ollama")
        self.assertIsInstance(client, OllamaClient)
        self.assertEqual(client.model, config.OLLAMA_MODEL)
        self.assertEqual(client.base_url, config.OLLAMA_BASE_URL)

    def test_ollama_client_init_defaults(self):
        """Verify default parameters on OllamaClient."""
        client = OllamaClient()
        self.assertEqual(client.model, "qwen2.5-coder:3b")
        self.assertEqual(client.base_url, "http://localhost:11434")
        self.assertEqual(client.timeout, 45.0)

    @patch("urllib.request.urlopen")
    def test_ollama_is_available_true(self, mock_urlopen):
        """is_available returns True when Ollama /api/tags succeeds."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"models": [{"name": "qwen2.5-coder:3b"}]}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        client = OllamaClient()
        self.assertTrue(client.is_available())

    @patch("urllib.request.urlopen")
    def test_ollama_is_available_false_on_connection_error(self, mock_urlopen):
        """is_available returns False when Ollama server is unreachable."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        client = OllamaClient()
        self.assertFalse(client.is_available())

    @patch("assistant.memory.store.add_exchange")
    @patch("assistant.memory.store.remember_topic")
    @patch("urllib.request.urlopen")
    def test_ollama_streaming_sentences(self, mock_urlopen, mock_topic, mock_exchange):
        """Test streaming JSON responses from Ollama API and chunking by sentence."""
        lines = [
            json.dumps({"response": "Hello world! "}).encode("utf-8") + b"\n",
            json.dumps({"response": "How are you today?"}).encode("utf-8") + b"\n",
            json.dumps({"response": "", "done": True}).encode("utf-8") + b"\n",
        ]
        mock_response = MagicMock()
        mock_response.__iter__.return_value = iter(lines)
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        client = OllamaClient()

        chunks = list(client.stream("Hi there"))
        self.assertGreater(len(chunks), 0)
        full_text = " ".join(chunks)
        self.assertIn("Hello world!", full_text)
        self.assertIn("How are you today?", full_text)

        # Verify store recording was called
        mock_exchange.assert_called_once_with("Hi there", "Hello world! How are you today?")

    @patch("urllib.request.urlopen")
    def test_ollama_stream_connection_failure_yields_clean_error(self, mock_urlopen):
        """When Ollama fails or disconnects, stream yields an error message without crashing."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        client = OllamaClient()

        chunks = list(client.stream("Test prompt"))
        self.assertEqual(len(chunks), 1)
        self.assertIn("trouble communicating with my local reasoning model", chunks[0])

    @patch("urllib.request.urlopen")
    def test_ollama_warmup_success(self, mock_urlopen):
        """Warmup succeeds when /api/generate returns 200."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = OllamaClient()
        self.assertTrue(client.warmup(timeout=5.0))

    @patch("urllib.request.urlopen")
    def test_ollama_warmup_failure_non_fatal(self, mock_urlopen):
        """Warmup returns False on network error without raising."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        client = OllamaClient()
        self.assertFalse(client.warmup(timeout=5.0))


if __name__ == "__main__":
    unittest.main()
