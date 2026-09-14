"""
Fast Qwen Classifier for LangGraph Orchestration.

Uses the locally resident Ollama model to classify incoming requests into
either 'LOCAL' (Ollama) or 'GEMINI' (cloud Gemini), with minimal token generation
and ultra-low latency.
"""
import json
import time
import urllib.error
import urllib.request
from typing import Tuple

from assistant import config
from assistant.core.log import get

log = get("router")

CLASSIFIER_SYSTEM_PROMPT = (
    "You are a strict query router. Classify the user query into exactly one category: TASK, LOCAL, or GEMINI.\n\n"
    "CRITICAL RULES (Strict Priority):\n"
    "1. TASK: ANY query about weather, rain, temperature, forecast, location, coordinates, or inspecting/explaining the JARVIS project and codebase.\n"
    "   Examples of TASK:\n"
    "   - 'What's the weather where I am?' -> TASK\n"
    "   - 'What is the weather in Hyderabad?' -> TASK\n"
    "   - 'Will it rain where I am?' -> TASK\n"
    "   - 'What is the temperature in London?' -> TASK\n"
    "   - 'Where am I located?' -> TASK\n"
    "   - 'How does my voice reach you?' -> TASK\n"
    "   - 'What components are handling my voice?' -> TASK\n"
    "   - 'Where is the Qwen classifier?' -> TASK\n"
    "   - 'What models am I using in this project?' -> TASK\n"
    "   - 'Inspect my JARVIS orchestration code' -> TASK\n"
    "   - 'Run the tests' -> TASK\n\n"
    "2. GEMINI: The query contains Telugu (script or Romanised e.g., 'nuvvu', 'entha', 'ela'), Hindi (script or Romanised e.g., 'tum', 'kaise', 'kya'), code-switching, or deep philosophy ('What is consciousness?'). Language priority overrides simplicity.\n\n"
    "3. LOCAL: Pure English conversational small-talk ('Hello Chavez, can you hear me?', 'Good morning Jarvis', 'How are you?', 'Let\'s talk'), simple math ('What is 25 times 40?'), or generic factual QA completely unrelated to weather, location, or this project ('What is HTTP?', 'What is the capital of France?').\n\n"
    "Respond with ONLY one word: TASK, LOCAL, or GEMINI."
)


class QwenClassifier:
    """Lightweight classifier invoking local Ollama Qwen with strict token limits."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.CLASSIFIER_MODEL
        self.timeout_s = float(timeout_s or config.CLASSIFIER_TIMEOUT_S)
        self.keep_alive = config.OLLAMA_KEEP_ALIVE

    def classify(self, user_text: str) -> Tuple[str, float]:
        """Classify user_text into 'TASK', 'LOCAL', or 'GEMINI'.

        Returns:
            (route, latency_ms): Selected route ('TASK', 'LOCAL', or 'GEMINI') and execution time in ms.
        """
        t0 = time.perf_counter()

        prompt = (
            f"{CLASSIFIER_SYSTEM_PROMPT}\n\n"
            f"Query: {user_text.strip()}\n"
            f"Route:"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0.0,
                "num_predict": 5,
                "stop": ["\n", " ", ".", ","],
            },
        }

        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "JARVIS-Router"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_route = data.get("response", "").strip().upper()

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Normalize and validate route
            if "TASK" in raw_route:
                route = "TASK"
            elif "LOCAL" in raw_route:
                route = "LOCAL"
            elif "GEMINI" in raw_route:
                route = "GEMINI"
            else:
                log.warning(
                    "[ROUTER] Invalid classifier output '%s' for '%s', defaulting to GEMINI",
                    raw_route, user_text[:50]
                )
                route = "GEMINI"

            log.info("[ROUTER] Classified in %.1fms -> %s (query: '%s')", elapsed_ms, route, user_text[:40])
            return route, elapsed_ms

        except urllib.error.URLError as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            log.warning("[ROUTER] Ollama unreachable (%s) in %.1fms, falling back to GEMINI", e, elapsed_ms)
            return "GEMINI", elapsed_ms

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            log.warning("[ROUTER] Classification error (%s) in %.1fms, falling back to GEMINI", e, elapsed_ms)
            return "GEMINI", elapsed_ms

