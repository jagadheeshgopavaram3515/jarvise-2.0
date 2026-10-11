"""
Fast Qwen Classifier for LangGraph Orchestration.

Uses the locally resident Ollama model to classify incoming requests into
either 'LOCAL' (Ollama) or 'GEMINI' (cloud Gemini), with minimal token generation
and ultra-low latency.
"""
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Tuple

import re
from typing import Tuple

from assistant import config
from assistant.core.log import get

log = get("router")


def _is_timeout_error(error: BaseException) -> bool:
    """Identify timeout failures without reclassifying other URL errors."""
    if isinstance(error, (TimeoutError, socket.timeout)):
        return True
    if isinstance(error, urllib.error.URLError):
        reason = error.reason
        return isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower()
    return False

# Regex pattern for native Indic scripts (Telugu, Devanagari/Hindi, Tamil)
_INDIC_SCRIPT_RE = re.compile(r"[\u0C00-\u0C7F\u0900-\u097F\u0B80-\u0BFF]")

# Romanised words indicative of conversational Telugu / Hindi code-switching
_ROMANISED_INDIC_RE = re.compile(
    r"\b(nuvvu|ela|unnav|unnaru|entha|eppudu|ekkada|cheyyi|cheppu|kavali|undi|ledu|"
    r"tum|aap|kaise|ho|kya|kyun|kab|kahan|batao|chalo|hai|hain|nahi|theek|karo)\b",
    re.I,
)

# TASK indicators: weather, location, codebase inspection, tests
_TASK_KEYWORD_RE = re.compile(
    r"\b(weather|rain|raining|temperature|forecast|location|coordinates|where am i|"
    r"codebase|code base|inspect|orchestration|classifier|models am i using|"
    r"run.*tests|unit.*tests|test.*suite)\b",
    re.I,
)

# Common simple English greetings / conversational small-talk / simple math
_COMMON_ENGLISH_LOCAL_RE = re.compile(
    r"^(hello|hi|hey|good morning|good afternoon|good evening|how are you|"
    r"how are you doing|let'?s talk|can you hear me|who are you|what is your name|"
    r"what is \d+[\s\+\-\*\/]+\d+\??)$",
    re.I,
)


def fast_classify(user_text: str) -> str | None:
    """Deterministic fast-path router. Returns route if confident, or None to fall back to Qwen LLM."""
    clean = user_text.strip()
    if not clean:
        return None

    # 1. TASK priority: weather, location, codebase, tests
    if _TASK_KEYWORD_RE.search(clean):
        return "TASK"

    # 2. Multilingual priority: Telugu, Hindi, Tamil script or Romanised code-switching -> GEMINI
    if _INDIC_SCRIPT_RE.search(clean) or _ROMANISED_INDIC_RE.search(clean):
        return "GEMINI"

    # For general English queries, let Qwen decide (or fallback if Ollama offline)
    return None

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

        Fast-path evaluation runs first in < 0.5ms. If unambiguous, it bypasses
        Ollama. If ambiguous or complex, it falls back to local Qwen.

        Returns:
            (route, latency_ms): Selected route ('TASK', 'LOCAL', or 'GEMINI') and execution time in ms.
        """
        t0 = time.perf_counter()

        # 1. Fast deterministic route
        fast = fast_classify(user_text)
        if fast is not None:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            log.info("[ROUTER] Fast-path route=%s total_ms=%.2f ollama_called=false", fast, elapsed_ms)
            return fast, elapsed_ms

        # 2. Semantic agent / Qwen route via local Ollama
        preparation_started = time.perf_counter()
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
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "JARVIS-Router"},
        )
        preparation_ms = (time.perf_counter() - preparation_started) * 1000.0

        try:
            http_started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                response_bytes = resp.read()
            http_ms = (time.perf_counter() - http_started) * 1000.0

            parsing_started = time.perf_counter()
            data = json.loads(response_bytes.decode("utf-8"))
            raw_route = data.get("response", "").strip().upper()

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
            parsing_ms = (time.perf_counter() - parsing_started) * 1000.0
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            log.info(
                "[ROUTER] Classifier success route=%s total_ms=%.1f preparation_ms=%.1f "
                "http_read_ms=%.1f parsing_ms=%.1f ollama_called=true",
                route, elapsed_ms, preparation_ms, http_ms, parsing_ms,
            )
            return route, elapsed_ms

        except urllib.error.URLError as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            category = "timeout" if _is_timeout_error(e) else "connection_error"
            log.warning(
                "[ROUTER] Classifier failure category=%s total_ms=%.1f preparation_ms=%.1f "
                "fallback=GEMINI",
                category, elapsed_ms, preparation_ms,
            )
            return "GEMINI", elapsed_ms

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            category = "timeout" if _is_timeout_error(e) else "response_or_parse_error"
            log.warning(
                "[ROUTER] Classifier failure category=%s total_ms=%.1f preparation_ms=%.1f "
                "fallback=GEMINI",
                category, elapsed_ms, preparation_ms,
            )
            return "GEMINI", elapsed_ms

