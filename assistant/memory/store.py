"""
Backward-compatible facade over the multi-layer MemoryEngine.

This module's PUBLIC API is byte-for-byte the same surface the rest of the app
already calls — gemini.py, dispatcher.py and pipeline.py are UNCHANGED:

    get_user_name / set_user_name
    load_memory / save_memory
    detect_mood / remember_mood / recent_mood
    remember_topic
    load_history / save_history / add_exchange / format_history
    load_session_summary
    retrieve_memories
    format_memory_context

Behind these calls, the shallow keyword/bag-of-words system is replaced by the
seven-layer engine (working, episodic, profile, relationship, project, emotional,
semantic). The engine is built lazily on first use so importing this module has
no side effects.

The old flat files (memory.json, conversation_history.json, vector_memory.json,
mood_state.json) are imported once on startup when no v2 memory exists yet.
"""
from __future__ import annotations

import os

from assistant import config
from assistant.memory import extractor
from assistant.memory.engine import get_engine
from assistant.memory.profile import LIST_FIELDS, SCALAR_FIELDS


def ensure_legacy_migrated() -> None:
    """Import legacy memory only for a fresh v2 store; never duplicate live data."""
    marker = os.path.join(config.MEMORY_DIR, "_migrated_v2.flag")
    if os.path.exists(marker):
        return
    active = ("working.json", "profile.json", "projects.json", "episodes.json",
              "emotions.json", "semantic.meta.json")
    if any(os.path.exists(os.path.join(config.MEMORY_DIR, name)) for name in active):
        return
    legacy = (config.MEMORY_FILE, config.CONVERSATION_HISTORY_FILE,
              config.VECTOR_MEMORY_FILE, config.MOOD_FILE)
    if not any(os.path.exists(path) for path in legacy):
        return
    from assistant.memory.migrate import migrate
    migrate(force=False)


# ----------------------------------------------------------------- name
def get_user_name(default: str = "sir") -> str:
    name = get_engine().profile.get().get("name", "").strip()
    if name:
        return name
    # Legacy fallback so an existing user_name.txt still greets correctly.
    if os.path.exists(config.NAME_FILE):
        try:
            txt = open(config.NAME_FILE, encoding="utf-8").read().strip()
            if txt:
                return txt
        except Exception:
            pass
    return default


def set_user_name(name: str) -> None:
    name = (name or "").strip()
    if not name:
        return
    get_engine().profile.set_scalar("name", name)
    try:
        with open(config.NAME_FILE, "w", encoding="utf-8") as f:
            f.write(name)
    except Exception:
        pass


# ----------------------------------------------------------------- legacy buckets
def load_memory() -> dict:
    """Reconstruct the old bucket dict from the profile (read compatibility)."""
    p = get_engine().profile.get()
    return {
        "learned_topics": p.get("interests", []),
        "preferences": p.get("likes", []),
        "dislikes": p.get("dislikes", []),
        "names": [p["name"]] if p.get("name") else [],
        "projects": [r["project_name"] for r in get_engine().projects.all()],
        "goals": p.get("goals", []),
        "routines": p.get("routines", []),
        "mood": [e["emotion"] for e in get_engine().emotional.recent(5)],
    }


def save_memory(memory: dict) -> None:
    """Best-effort: route legacy bucket writes into the profile."""
    prof = get_engine().profile
    mapping = {"preferences": "likes", "dislikes": "dislikes",
               "goals": "goals", "routines": "routines",
               "learned_topics": "interests"}
    for bucket, field in mapping.items():
        if memory.get(bucket):
            prof.merge_list(field, memory[bucket])


# ----------------------------------------------------------------- mood
def detect_mood(text: str):
    e = extractor.detect_emotion(text)
    return e[0] if e else None


def remember_mood(text: str):
    e = extractor.detect_emotion(text)
    if not e:
        return None
    get_engine().emotional.add(e[0], intensity=e[1], trigger=(text or "")[:80])
    get_engine().working.set_state(emotion=e[0])
    return e[0]


def recent_mood(max_age_hours: float = 18.0):
    rec = get_engine().emotional.last(max_age_hours=max_age_hours)
    return rec.get("emotion") if rec else None


def remember_topic(text: str) -> None:
    words = (text or "").lower().split()
    if "about" in words:
        i = words.index("about")
        if i + 1 < len(words):
            get_engine().profile.merge_list("interests", [words[i + 1]])


# ----------------------------------------------------------------- history
def load_history() -> list[dict]:
    """Recent turns as the old [{'role','message'}] shape (working memory)."""
    return [{"role": t.get("role", "user"), "message": t.get("message", "")}
            for t in get_engine().working.turns()]


def save_history(history: list[dict]) -> None:
    """No-op: working memory is the source of truth now (kept for API parity)."""
    return None


def add_exchange(user_msg: str, assistant_reply: str) -> None:
    get_engine().record_exchange(user_msg, assistant_reply)


def flush(timeout: float = 2.0) -> None:
    """Best-effort drain of background memory work during application exit."""
    get_engine().flush(timeout=timeout)


def format_history(history: list[dict]) -> str:
    return "\n".join(f"{h['role'].capitalize()}: {h['message']}" for h in history)


# ------------------------------------------------------------- tiered memory
def load_session_summary() -> dict:
    eng = get_engine()
    return {"turns_seen": eng.relationship.stats().get("total_turns", 0),
            "summary": eng.working.state().get("topic", "")}


def retrieve_memories(query: str, top_k: int | None = None,
                      min_similarity: float | None = None) -> list[str]:
    """Semantic retrieval (replaces token-cosine). Returns memory texts."""
    top_k = top_k or config.MEMORY_TOP_K
    idx = get_engine()._index_if_ready()   # never build on a read/hot path
    if idx is None:
        return []
    try:
        hits = idx.search(query, top_k=top_k)
        return [h.get("text", "") for h in hits if h.get("text")]
    except Exception:
        return []


def format_memory_context(query: str, recall: bool = False) -> str:
    """The single string injected into the Gemini prompt — now multi-layer."""
    return get_engine().build_context(query, recall=recall)
