"""
Tool-calling layer (Phase 5) — selects a tool, runs it, speaks the result.

Flow (additive; the main conversational LLM path is untouched):

    User ─▶ Dispatcher ─▶ commands.handlers.handle()
                              │  (fast deterministic commands first)
                              ▼
                        router.try_tools(text, bus)
                              │  1. deterministic intent match  (no API call)
                              │  2. else Gemini tool selection  → {"tool","args"}
                              ▼
                        ToolExecutor.execute(tool, args)   (allow-list + timeout)
                              │
                              ▼
                        Gemini natural response  ─▶ bus.speak(...)  ─▶ TTS

``try_tools`` returns True iff a tool handled the request — in which case the
Dispatcher's command fast-path returns True and the main LLM is skipped. If no
tool matches, it returns False and everything proceeds exactly as before
(backward compatible).

This layer owns its OWN lightweight genai client for selection/response — it
never touches the protected ``llm.gemini.GeminiClient``. Selection/response use
a small, fast model so they don't compete with the main turn model's quota.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Any, Dict, Optional, Tuple

from assistant import config
from assistant.core.log import get
from assistant.tools import get_executor, get_registry

log = get("tools")

# A request that is clearly conversational (a question, chit-chat) should never
# be hijacked by a tool. Tool intents are imperative: open/close/find/search/read.
_TOOL_TRIGGER = re.compile(
    r"\b(open|launch|start|close|quit|kill|find|search|look\s+for|locate|"
    r"show|list|read|go\s+to|browse|play)\b", re.I)


# --------------------------------------------------------------------------- #
# Deterministic intent matcher — fast path, no API call. Covers the common
# voice commands in the success criteria precisely; anything else falls through
# to Gemini selection.
# --------------------------------------------------------------------------- #
_APP_WORDS = {
    "chrome": "chrome", "google chrome": "chrome",
    "edge": "edge", "microsoft edge": "edge",
    "vs code": "vs code", "vscode": "vs code", "visual studio code": "vs code",
    "code": "vs code",
    "notepad": "notepad",
    "calculator": "calculator", "calc": "calculator",
    "explorer": "explorer", "file explorer": "explorer",
}


def _match_app(text: str) -> Optional[str]:
    low = text.lower()
    # Prefer the longest alias so "google chrome" wins over "chrome", etc.
    for alias in sorted(_APP_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", low):
            return _APP_WORDS[alias]
    return None


def _deterministic(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return (tool, args) for an unambiguous phrase, else None."""
    t = text.strip()
    low = t.lower()

    # --- YouTube search: "search youtube for X" / "youtube X" ---------------
    m = re.search(r"(?:search|find|look up|play)\s+(?:on\s+)?youtube\s+(?:for\s+)?(.+)", low)
    if m and m.group(1).strip():
        return "youtube_search", {"query": m.group(1).strip()}
    m = re.search(r"youtube\s+(?:search\s+)?(?:for\s+)?(.+)", low)
    if m and m.group(1).strip() and "open" not in low:
        return "youtube_search", {"query": m.group(1).strip()}

    # --- Open a folder: "open my X folder" / "open folder X" ----------------
    m = re.search(r"open\s+(?:my\s+|the\s+)?(.+?)\s+folder", low)
    if m and m.group(1).strip():
        return "open_folder", {"path": m.group(1).strip()}
    m = re.search(r"open\s+folder\s+(.+)", low)
    if m and m.group(1).strip():
        return "open_folder", {"path": m.group(1).strip()}

    # --- Open / close an app ------------------------------------------------
    if re.search(r"\b(open|launch|start)\b", low):
        app = _match_app(low)
        if app:
            return "open_app", {"app": app}
    if re.search(r"\b(close|quit|kill|exit)\b", low):
        app = _match_app(low)
        if app:
            return "close_app", {"app": app}

    # --- Find a file: "find my resume" / "find file X" / "search files X" ---
    m = (re.search(r"(?:find|locate)\s+(?:a\s+|the\s+|my\s+)?(.+)", low)
         or re.search(r"search\s+(?:for\s+)?(?:files?\s+(?:for\s+)?)?"
                      r"(?:a\s+|the\s+|my\s+)?(.+)", low))
    if m and m.group(1).strip() and "youtube" not in low and "google" not in low:
        target = m.group(1).strip()
        # Strip leftover possessive and a trailing "file"/"files" noise word.
        target = re.sub(r"^(?:my|the|a)\s+", "", target)
        target = re.sub(r"\b(file|files)\b$", "", target).strip()
        if target:
            return "find_file", {"filename": target}

    # --- Read page title / page text (needs an explicit URL) ----------------
    url = _extract_url(low)
    if url and re.search(r"\b(title|titled)\b", low):
        return "get_page_title", {"url": url}
    if url and re.search(r"\bread\b", low):
        return "read_page", {"url": url}
    if url and re.search(r"\b(open|go to|browse)\b", low):
        return "open_url", {"url": url}

    # --- Google search: "google X" / "search google for X" ------------------
    m = re.search(r"(?:search\s+(?:on\s+)?google\s+(?:for\s+)?|google\s+(?:for\s+)?)(.+)", low)
    if m and m.group(1).strip() and "youtube" not in low:
        return "google_search", {"query": m.group(1).strip()}

    return None


_URL_RE = re.compile(r"(https?://[^\s]+|[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?)", re.I)


def _extract_url(text: str) -> Optional[str]:
    m = _URL_RE.search(text)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Gemini tool selection + natural-response generation.
# --------------------------------------------------------------------------- #
class _Selector:
    """Owns a lazily-created genai client for tool selection and phrasing."""

    def __init__(self) -> None:
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                try:
                    from google import genai
                    self._client = genai.Client(api_key=config.GOOGLE_API_KEY)
                except Exception as e:
                    log.warning("[TOOLS] selection client unavailable: %s", e)
                    self._client = None
        return self._client

    # ---- tool selection ----
    def select(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        client = self._get_client()
        if client is None:
            return None
        registry = get_registry()
        prompt = (
            "You route a voice assistant's request to ONE desktop tool, or to no "
            "tool if it's just conversation.\n"
            "Available tools:\n" + registry.describe() + "\n\n"
            "Reply with ONLY compact JSON. If a tool fits, output "
            '{\"tool\": \"<name>\", \"args\": { ... }}. '
            "If NO tool fits (it's a question, chit-chat, or needs the "
            'conversational AI), output {\"tool\": null}.\n'
            "Never invent tools or arguments. Fill args exactly as the tool "
            "expects.\n\n"
            f"Request: {text}\nJSON:"
        )
        try:
            resp = _with_timeout(
                lambda: client.models.generate_content(
                    model=config.TOOLS_SELECTION_MODEL, contents=prompt),
                config.TOOLS_SELECTION_TIMEOUT_S)
            raw = getattr(resp, "text", "") if resp else ""
            data = _parse_json(raw)
            if not data:
                return None
            tool = data.get("tool")
            if not tool:
                return None
            if tool not in registry:
                log.info("[TOOLS] Gemini picked unknown tool %r — ignoring", tool)
                return None
            args = data.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            return tool, args
        except Exception as e:
            log.warning("[TOOLS] selection failed: %s", e)
            return None

    # ---- natural response ----
    def phrase(self, user_text: str, tool: str, result: Dict[str, Any]) -> Optional[str]:
        if not config.TOOLS_NATURAL_RESPONSE:
            return None
        client = self._get_client()
        if client is None:
            return None
        prompt = (
            "You are Jarvis, a warm, concise British AI butler. Address the user "
            "as 'sir'. The user gave a command and a desktop tool just ran. In ONE "
            "short spoken sentence (no markdown, no emoji), tell the user the "
            "outcome naturally.\n"
            f"User said: {user_text}\n"
            f"Tool: {tool}\n"
            f"Tool result: {json.dumps({'success': result.get('success'), 'message': result.get('message')}, ensure_ascii=False)}\n"
            "Reply:"
        )
        try:
            resp = _with_timeout(
                lambda: client.models.generate_content(
                    model=config.TOOLS_RESPONSE_MODEL, contents=prompt),
                config.TOOLS_RESPONSE_TIMEOUT_S)
            text = (getattr(resp, "text", "") or "").strip()
            return text or None
        except Exception as e:
            log.warning("[TOOLS] phrasing failed: %s", e)
            return None


def _with_timeout(fn, timeout: float):
    """Run ``fn`` with a hard timeout so a slow API call can't stall dispatch."""
    box: Dict[str, Any] = {}

    def _run():
        try:
            box["result"] = fn()
        except Exception as e:  # captured, re-raised on the caller thread
            box["error"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError(f"call exceeded {timeout}s")
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _parse_json(raw: str) -> Optional[dict]:
    """Extract a JSON object from a model reply (tolerates ``` fences)."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s).rstrip("`").strip()
    # Grab the first {...} block.
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None


_selector = _Selector()


# --------------------------------------------------------------------------- #
# Public entry point used by commands.handlers (the single integration hook).
# --------------------------------------------------------------------------- #
def try_tools(text: str, bus) -> bool:
    """Attempt to satisfy ``text`` with a desktop tool.

    Returns True if a tool ran (Dispatcher skips the main LLM); False otherwise
    (the conversation continues to the LLM exactly as before).
    """
    if not config.TOOLS_ENABLED:
        return False
    text = (text or "").strip()
    if not text:
        return False

    # 1) Fast deterministic match — no API call.
    picked = _deterministic(text)

    # 2) Gemini selection fallback (only for plausibly-imperative requests, so
    #    ordinary conversation never pays an extra API round-trip).
    if picked is None and config.TOOLS_GEMINI_SELECTION and _TOOL_TRIGGER.search(text):
        picked = _selector.select(text)

    if picked is None:
        return False

    tool, args = picked
    log.info("[TOOLS] dispatch %s args=%s (text=%r)", tool, args, text[:80])

    executor = get_executor()
    result = executor.execute(tool, args)

    # 3) Speak the outcome — naturally via Gemini, or the tool's own message.
    spoken = None
    if result.get("success") or config.TOOLS_NATURAL_RESPONSE:
        spoken = _selector.phrase(text, tool, result)
    bus.speak(spoken or result.get("message") or "Done, sir.")
    return True
