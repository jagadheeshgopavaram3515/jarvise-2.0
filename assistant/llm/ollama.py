"""
OllamaClient — local streaming LLM client for Ollama (e.g. Qwen).

Provides the exact same interface as GeminiClient:
    stream(user_text: str) -> Iterator[str]

Sentence-by-sentence streaming ensures the TTS engine starts playback
immediately while the rest of the generation continues in the background.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Iterator

from assistant import config
from assistant.core.log import get
from assistant.memory import store
from assistant.realtime import RealTimeManager

log = get("ollama")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+|\n+")
_RECALL_INTENT = re.compile(
    r"\b(do you remember|you remember|remember when|remember that|last time|"
    r"we (?:talked|spoke|discussed|chatted)|you (?:said|told|mentioned)|recall|"
    r"what did i (?:say|tell)|as i (?:said|mentioned)|earlier i)\b", re.I)

SYSTEM_PROMPT = (
    "You are {name}, a warm, friendly British AI companion in the spirit of Iron Man's "
    "JARVIS — gently witty, relaxed and genuinely caring, like a close friend who happens "
    "to be a brilliant butler. Not stiff or formal. Address the user as 'sir'. "
    "Keep everyday replies short and conversational — one or two sentences — so the chat "
    "flows naturally. Only when the user asks for a story, explanation, or code do you "
    "give the full, complete answer. "
    "All text is spoken aloud: plain spoken language, NO markdown, asterisks, headings, "
    "bullet points or emojis. Be warm, friendly and human.\n"
    "Respond to the user's CURRENT message below warmly and to the point.\n"
)


_SIMPLE_MATH_OR_QA = re.compile(
    r"(\b(?:calculate|convert)\b|\d+\s*(?:times|plus|minus|divided by|[\+\-\*\/\%x×])\s*\d+|"
    r"what is (?:the )?(?:capital|definition|meaning|http|cpu|api|langgraph)\b)",
    re.I,
)


class OllamaClient:
    """Local LLM client communicating with an Ollama HTTP service."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout = float(timeout or config.OLLAMA_TIMEOUT_S)
        self.realtime = RealTimeManager()

    def is_available(self, check_timeout: float = 2.0) -> bool:
        """Quick health check to determine if the local Ollama instance is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", headers={"User-Agent": "JARVIS"})
            with urllib.request.urlopen(req, timeout=check_timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", [])]
                    # True if reachable and optionally contains requested model
                    return bool(models)
        except Exception:
            return False
        return False

    def warmup(self, timeout: float = 35.0) -> bool:
        """Pre-warm the local Ollama model with a minimal generation request.

        Ensures the model is resident in GPU VRAM before voice interaction begins.
        """
        payload = {
            "model": self.model,
            "prompt": "hi",
            "stream": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {
                "num_predict": 1,
            },
        }
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "JARVIS-Warmup"},
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    took_ms = (time.perf_counter() - t0) * 1000.0
                    log.info("[WARMUP] Local model '%s' warmed in %.1fms.", self.model, took_ms)
                    return True
        except Exception as e:
            log.warning("[WARMUP] Local model '%s' warmup failed: %s", self.model, e)
            return False
        return False

    def _build_prompt(self, user_text: str, realtime_context: str = "") -> str:
        system = SYSTEM_PROMPT.format(name=config.ASSISTANT_NAME)
        recall = bool(_RECALL_INTENT.search(user_text))
        history = store.format_history(store.load_history()[-config.GEMINI_MAX_HISTORY:])

        # Keep simple math & definitions lightweight; supply shared memory for conversations
        is_simple = bool(_SIMPLE_MATH_OR_QA.search(user_text))
        if is_simple and not recall:
            memory = ""
        else:
            memory = store.format_memory_context(user_text, recall=recall)

        now = datetime.now()

        parts = [
            system,
            f"For reference, right now it is {now:%I:%M %p} on {now:%A, %d %B %Y}.",
        ]
        if memory:
            parts.append("Background about the user:\n" + memory)
        if history:
            parts.append("Recent conversation:\n" + history)
        if realtime_context:
            parts.append("Real-time web briefing:\n" + realtime_context)

        parts.append(f"User: {user_text}\nAssistant:")
        return "\n\n".join(parts)

    @staticmethod
    def _ready_chunks(buffer: str, force: bool = False) -> tuple[list[str], str]:
        chunks: list[str] = []
        parts = _SENTENCE_END.split(buffer)
        if len(parts) > 1:
            *complete, buffer = parts
            for part in complete:
                s = part.strip()
                if s:
                    chunks.append(s)

        words = buffer.split()
        while len(words) >= config.STREAM_MAX_WORDS:
            chunks.append(" ".join(words[:config.STREAM_MAX_WORDS]))
            words = words[config.STREAM_MAX_WORDS:]
            buffer = " ".join(words)

        if force and buffer.strip():
            chunks.append(buffer.strip())
            buffer = ""
        elif len(words) >= config.STREAM_MIN_WORDS and buffer.rstrip().endswith((",", ";", ":")):
            chunks.append(buffer.strip())
            buffer = ""
        return chunks, buffer

    def stream(self, user_text: str) -> Iterator[str]:
        """Stream reply sentences from the local Ollama Qwen model."""
        realtime_context = ""
        if self.realtime.needs_realtime(user_text):
            try:
                realtime_context = self.realtime.get_realtime_context(user_text)
            except Exception as e:
                log.warning("[REALTIME] context fetch failed: %s", e)

        prompt = self._build_prompt(user_text, realtime_context=realtime_context)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.3,
            },
        }

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "JARVIS"},
        )

        buffer = ""
        full: list[str] = []
        yielded = False

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    if not raw_line:
                        continue
                    line_str = raw_line.decode("utf-8").strip()
                    if not line_str:
                        continue
                    try:
                        chunk = json.loads(line_str)
                    except json.JSONDecodeError:
                        continue

                    piece = chunk.get("response", "")
                    if piece:
                        buffer += piece
                        ready, buffer = self._ready_chunks(buffer)
                        for clause in ready:
                            full.append(clause)
                            yielded = True
                            yield clause

                    if chunk.get("done", False):
                        break

            # Flush any remaining buffer text
            ready, buffer = self._ready_chunks(buffer, force=True)
            for tail in ready:
                full.append(tail)
                yielded = True
                yield tail

        except urllib.error.URLError as e:
            log.warning("Ollama connection failed: %s", e)
            if not yielded:
                yield self._fallback(user_text, e)
            return
        except Exception as e:
            log.warning("Ollama stream error: %s", e)
            if not yielded:
                yield self._fallback(user_text, e)
            return

        if full:
            store.remember_topic(user_text)
            store.add_exchange(user_text, " ".join(full))
        elif not yielded:
            yield self._fallback(user_text, RuntimeError("empty_response"))

    @staticmethod
    def _fallback(prompt: str, error: Exception) -> str:
        words = set(prompt.lower().split())
        if words & {"hello", "hi", "hey", "namaste"}:
            return "Hello! How can I assist you today, sir?"
        if words & {"thanks", "thank"}:
            return "You're welcome, sir."
        now = datetime.now()
        if "time" in words:
            return f"The time is {now:%I:%M %p}, sir."
        if {"date", "today"} & words:
            return f"Today is {now:%A, %B %d, %Y}, sir."
        return "I apologize, sir, but I am having trouble communicating with my local reasoning model."

