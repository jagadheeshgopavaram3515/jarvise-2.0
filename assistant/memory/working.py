"""
Layer 1 — Working Memory.

The live conversation: the last N turns plus the current topic / goal / task /
emotional state / recent entities. This is the cheap, synchronous, always-injected
layer. add_turn() is called on the dispatcher thread and must stay fast (no LLM,
no embedding) — those happen later in the background worker.
"""
from __future__ import annotations

from assistant import config
from assistant.memory.schema import JSONStore, now_ts

_DEFAULT = {
    "topic": "", "goal": "", "task": "", "emotion": "",
    "entities": [], "turns": [],
}


class WorkingMemory:
    def __init__(self, path: str):
        self._store = JSONStore(path, _DEFAULT)
        self._cap = max(4, config.MEMORY_WORKING_TURNS)

    def add_turn(self, role: str, message: str) -> None:
        msg = (message or "").strip()
        if not msg:
            return

        def _mut(d):
            d.setdefault("turns", []).append(
                {"role": role, "message": msg, "ts": now_ts()})
            d["turns"] = d["turns"][-self._cap:]
        self._store.update(_mut)

    def set_state(self, *, topic=None, goal=None, task=None,
                  emotion=None, entities=None) -> None:
        def _mut(d):
            if topic is not None:
                d["topic"] = topic
            if goal is not None:
                d["goal"] = goal
            if task is not None:
                d["task"] = task
            if emotion is not None:
                d["emotion"] = emotion
            if entities is not None:
                merged = list(dict.fromkeys((d.get("entities") or []) + list(entities)))
                d["entities"] = merged[-12:]
        self._store.update(_mut)

    def turns(self, limit: int | None = None) -> list[dict]:
        t = self._store.load().get("turns", [])
        return t[-limit:] if limit else t

    def state(self) -> dict:
        d = self._store.load()
        return {k: d.get(k) for k in ("topic", "goal", "task", "emotion", "entities")}

    def render(self) -> str:
        """A compact working-memory header for the prompt (state only — the
        verbatim recent turns are rendered separately by the facade)."""
        s = self.state()
        bits = []
        if s.get("topic"):
            bits.append(f"current topic: {s['topic']}")
        if s.get("goal"):
            bits.append(f"current goal: {s['goal']}")
        if s.get("task"):
            bits.append(f"current task: {s['task']}")
        if s.get("emotion"):
            bits.append(f"user mood: {s['emotion']}")
        if s.get("entities"):
            bits.append("recent topics: " + ", ".join(s["entities"][-6:]))
        return "; ".join(bits)
