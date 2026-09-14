"""
Memory extractor + importance scorer.

Two paths, both off the hot conversation path (run in the background worker):

  * score_exchange()  — CHEAP heuristic importance (no LLM). Decides whether a
    single exchange is even worth deep processing, so we don't spend a model
    call on "what time is it".
  * consolidate()     — LLM pass over a window of recent turns (every N turns).
    Returns structured episodes, profile updates, project updates, an emotion
    reading and a session summary. Falls back to a heuristic extractor when the
    LLM is unavailable/offline, so memory still grows without the network.

The LLM here is the small/fast model (gemini-2.5-flash-lite by default) — NEVER
the main turn model, so it can't slow a reply or burn primary quota.
"""
from __future__ import annotations

import json
import re

from assistant import config
from assistant.core.log import get

log = get("memory.extract")

# Phrases that signal a durable, important life/project fact worth storing.
_HIGH_SIGNALS = (
    "i got", "i landed", "new job", "interview", "i started", "i began",
    "i built", "i'm building", "im building", "i am building", "i launched",
    "i learned", "i finished", "i completed", "my goal", "my dream",
    "i want to become", "i'm preparing", "im preparing", "preparing for",
    "i decided", "i moved", "i joined", "i quit", "i failed", "i passed",
    "i bought", "my project", "working on", "i love", "i hate", "my name is",
    "call me", "i live in", "i'm studying", "im studying",
)
_LOW_SIGNALS = (
    "what time", "open ", "close ", "play ", "search for", "hello", "hi ",
    "thanks", "thank you", "stop", "the time", "the date", "what's the date",
)
_EMO = {
    "anxious": ("anxious", "nervous", "worried", "scared", "afraid", "stressed", "stress"),
    "frustrated": ("frustrated", "stuck", "annoyed", "angry", "bug", "not working", "failing"),
    "sad": ("sad", "low", "down", "depressed", "lonely", "tired", "exhausted", "hopeless"),
    "motivated": ("motivated", "ready", "determined", "focused", "let's go", "grinding"),
    "excited": ("excited", "thrilled", "can't wait", "cannot wait", "pumped", "amazing", "awesome"),
    "happy": ("happy", "glad", "great", "good", "relieved", "proud"),
}


def _has_marker(text: str, marker: str) -> bool:
    """Match emotion words/phrases, not fragments such as 'down payment'."""
    suffix = r"(?!\s+payment\b)" if marker == "down" else ""
    return re.search(rf"(?<!\w){re.escape(marker)}(?!\w){suffix}", text) is not None

_EXTRACT_SYSTEM = (
    "You extract durable memory from a short conversation between a user and "
    "their AI assistant. Return STRICT JSON only — no prose, no code fences.\n"
    "Schema:\n"
    "{\n"
    '  "session_summary": "2-3 sentence summary of what was discussed",\n'
    '  "episodes": [{"summary": "one meaningful event in the user\'s life/work", '
    '"importance": 0-10, "tags": ["..."], "emotion": ""}],\n'
    '  "profile_updates": {"name":"", "location":"", "role":"", "education":"", '
    '"skills":[], "goals":[], "dreams":[], "interests":[], "likes":[], "dislikes":[], '
    '"favorite_anime":[], "favorite_movies":[], "favorite_books":[], '
    '"career_aspirations":[], "routines":[], "habits":[], "languages":[], "relationships":[]},\n'
    '  "projects": [{"project_name":"", "status":"", "goals":[], "blockers":[], "next_steps":[]}],\n'
    '  "emotion": {"emotion":"", "intensity":"low|medium|high", "trigger":""}\n'
    "}\n"
    "Rules: Only include FACTS the user actually stated about THEMSELVES. Omit any "
    "field you have nothing for (use [] or \"\"). importance>=6 means a real life "
    "event (new job, started a project, a dream, an interview, a relationship, a "
    "purchase). Small talk and commands are importance<=3. Do NOT invent anything."
)


def score_exchange(user_text: str, assistant_text: str = "") -> int:
    """Cheap 0-10 importance for a single exchange (no LLM)."""
    low = (user_text or "").lower()
    if not low.strip():
        return 0
    if any(s in low for s in _LOW_SIGNALS) and len(low.split()) <= 6:
        return 1
    score = 3
    if any(s in low for s in _HIGH_SIGNALS):
        score = 7
    # Longer, self-referential statements tend to carry more.
    if low.startswith(("i ", "my ", "i'm", "im ")) and len(low.split()) >= 6:
        score = max(score, 6)
    return score


def detect_emotion(text: str) -> tuple[str, str] | None:
    low = (text or "").lower()
    for emo, markers in _EMO.items():
        if any(_has_marker(low, m) for m in markers):
            intensity = "high" if any(c in text for c in "!") or "very" in low else "medium"
            return emo, intensity
    return None


def extract_user_name(text: str) -> str | None:
    """Detect immediate name declarations in user speech."""
    t = (text or "").strip()
    m = re.search(
        r"\b(?:my\s+name\s+is|call\s+me|change\s+my\s+name\s+to|i\s+am\s+called)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b",
        t, re.IGNORECASE
    )
    if not m:
        return None
    raw = m.group(1).strip()
    first_token = raw.split()[0].strip().lower()
    if first_token in {"not", "a", "an", "the", "here", "sorry", "tired", "back"}:
        return None
    words = [w for w in raw.split() if w.lower() not in {"not", "and", "or", "but"}]
    if not words:
        return None
    return " ".join(words).title()


def _strip_json(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    # Grab the outermost JSON object if the model added stray text.
    start, end = t.find("{"), t.rfind("}")
    return t[start:end + 1] if start != -1 and end != -1 else t


class _ExtractorLLM:
    def __init__(self):
        from google import genai
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)
        try:
            from google.genai import types
            self._types = types
        except Exception:
            self._types = None
        self.model = config.MEMORY_EXTRACTION_MODEL

    def consolidate(self, transcript: str) -> dict | None:
        try:
            if self._types is not None:
                cfg = self._types.GenerateContentConfig(
                    system_instruction=_EXTRACT_SYSTEM,
                    temperature=0.2,
                    max_output_tokens=900,
                    response_mime_type="application/json",
                )
                resp = self._client.models.generate_content(
                    model=self.model, contents=transcript, config=cfg)
            else:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=f"{_EXTRACT_SYSTEM}\n\nConversation:\n{transcript}")
            return json.loads(_strip_json(getattr(resp, "text", "") or ""))
        except Exception:
            log.debug("LLM consolidation failed", exc_info=True)
            return None


_llm: _ExtractorLLM | None = None
_llm_failed = False


def _get_llm():
    global _llm, _llm_failed
    if _llm is None and not _llm_failed and config.MEMORY_EXTRACTION:
        try:
            _llm = _ExtractorLLM()
        except Exception:
            _llm_failed = True
            log.info("memory extractor LLM unavailable; heuristic only", exc_info=True)
    return _llm


def consolidate(turns: list[dict]) -> dict:
    """Return a structured memory delta for a window of recent turns."""
    transcript = "\n".join(
        f"{t.get('role', 'user').capitalize()}: {t.get('message', '')}" for t in turns)
    llm = _get_llm()
    if llm is not None:
        result = llm.consolidate(transcript)
        if result:
            return result
    return _heuristic_consolidate(turns)


def _heuristic_consolidate(turns: list[dict]) -> dict:
    """Offline fallback: marker-based facts + first-line session summary."""
    user_msgs = [t["message"] for t in turns if t.get("role") == "user"]
    episodes, emotion = [], None
    for m in user_msgs:
        if score_exchange(m) >= 6:
            episodes.append({"summary": m.strip()[:160], "importance": 6,
                             "tags": [], "emotion": ""})
        e = detect_emotion(m)
        if e:
            emotion = {"emotion": e[0], "intensity": e[1], "trigger": ""}
    summary = " ".join(user_msgs[-3:])[:300]
    return {"session_summary": summary, "episodes": episodes,
            "profile_updates": {}, "projects": [], "emotion": emotion or {}}
