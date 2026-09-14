"""
Layer 4 — Relationship Memory.

How long Jarvis has known the user, how many conversations they've had, and the
milestones along the way. This is what powers natural lines like "You have been
learning SAP for 3 months, sir" — grounded in real recorded dates, never
fabricated.
"""
from __future__ import annotations

from assistant.memory.schema import JSONStore, human_age, iso, now_ts

_DEFAULT = {"first_seen": 0.0, "total_turns": 0, "milestones": []}


class RelationshipMemory:
    def __init__(self, path: str):
        self._store = JSONStore(path, _DEFAULT)
        # Stamp first_seen on first ever use.
        d = self._store.load()
        if not d.get("first_seen"):
            d["first_seen"] = now_ts()
            self._store.save(d)

    def note_turn(self) -> None:
        self._store.update(lambda d: d.__setitem__(
            "total_turns", int(d.get("total_turns", 0)) + 1))

    def add_milestone(self, summary: str, ts: float | None = None) -> None:
        summary = (summary or "").strip()
        if not summary:
            return

        def _mut(d):
            ms = d.setdefault("milestones", [])
            ms.append({"ts": ts or now_ts(), "summary": summary})
            d["milestones"] = ms[-50:]
        self._store.update(_mut)

    def stats(self) -> dict:
        return self._store.load()

    def render(self) -> str:
        d = self._store.load()
        first = d.get("first_seen") or now_ts()
        parts = [f"You've known the user since {iso(first)[:10]} "
                 f"({human_age(first)}), across {d.get('total_turns', 0)} exchanges."]
        ms = d.get("milestones", [])
        if ms:
            recent = ms[-3:]
            parts.append("Milestones: " + "; ".join(
                f"{m['summary']} ({human_age(m['ts'])})" for m in recent))
        return " ".join(parts)
