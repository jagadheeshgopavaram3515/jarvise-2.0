"""
Layer 5 — Project Memory.

Dedicated tracking for the user's ongoing work (Bluye, Jarvis, SAP Datasphere,
MediBot, SmartPaisa...). Keyed by a normalized project name so "continue Bluye"
resolves to the right record. Each record carries status, goals, blockers and
next steps so Jarvis can resume context.
"""
from __future__ import annotations

from assistant.memory.schema import JSONStore, iso, now_ts

_DEFAULT: dict = {}   # {key: record}


def _key(name: str) -> str:
    return (name or "").strip().lower()


def _blank(name: str) -> dict:
    return {
        "project_name": name.strip(),
        "status": "active",
        "last_discussed": now_ts(),
        "goals": [],
        "blockers": [],
        "next_steps": [],
        "notes": [],
    }


def _merge_list(dst: list, vals) -> list:
    seen = {x.lower() for x in dst}
    for v in vals or []:
        v = str(v).strip()
        if v and v.lower() not in seen:
            dst.append(v)
            seen.add(v.lower())
    return dst[-12:]


class ProjectMemory:
    def __init__(self, path: str):
        self._store = JSONStore(path, _DEFAULT)

    def upsert(self, name: str, *, status=None, goals=None, blockers=None,
               next_steps=None, note=None) -> None:
        k = _key(name)
        if not k:
            return

        def _mut(d):
            rec = d.get(k) or _blank(name)
            rec["last_discussed"] = now_ts()
            if status:
                rec["status"] = status
            if goals:
                rec["goals"] = _merge_list(rec.get("goals", []), goals)
            if blockers:
                rec["blockers"] = _merge_list(rec.get("blockers", []), blockers)
            if next_steps:
                rec["next_steps"] = _merge_list(rec.get("next_steps", []), next_steps)
            if note:
                rec["notes"] = _merge_list(rec.get("notes", []), [note])
            d[k] = rec
        self._store.update(_mut)

    def get(self, name: str) -> dict | None:
        return self._store.load().get(_key(name))

    def all(self) -> list[dict]:
        return list(self._store.load().values())

    def active(self, limit: int = 4) -> list[dict]:
        recs = [r for r in self.all() if r.get("status") != "done"]
        recs.sort(key=lambda r: r.get("last_discussed", 0), reverse=True)
        return recs[:limit]

    def find_mentioned(self, text: str) -> list[dict]:
        low = (text or "").lower()
        return [r for r in self.all() if r.get("project_name", "").lower() in low]

    def render_one(self, rec: dict) -> str:
        parts = [f"{rec['project_name']} (status: {rec.get('status', 'active')}, "
                 f"last discussed {iso(rec.get('last_discussed'))[:10]})"]
        if rec.get("next_steps"):
            parts.append("  next steps: " + "; ".join(rec["next_steps"][-3:]))
        if rec.get("blockers"):
            parts.append("  blockers: " + "; ".join(rec["blockers"][-2:]))
        if rec.get("goals"):
            parts.append("  goals: " + "; ".join(rec["goals"][-2:]))
        return "\n".join(parts)
