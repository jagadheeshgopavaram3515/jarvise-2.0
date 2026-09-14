"""
Layer 6 — Emotional Memory.

Remembers emotional *patterns* over time (anxious about career, motivated after
progress, frustrated debugging, excited about motorcycles). It records and
summarizes — it does NOT diagnose or act like a therapist.
"""
from __future__ import annotations

from collections import Counter
import re

from assistant.memory.schema import EmotionRecord, JSONStore, human_age, now_ts

_DEFAULT: list = []


class EmotionalMemory:
    def __init__(self, path: str):
        self._store = JSONStore(path, _DEFAULT)

    def add(self, emotion: str, *, intensity: str = "medium",
            trigger: str = "", cap: int = 300) -> None:
        if not emotion:
            return
        rec = EmotionRecord(emotion=emotion, intensity=intensity, trigger=trigger).to_dict()

        def _mut(d):
            # Dispatcher and background extraction can observe the same turn.
            if d:
                prev = d[-1]
                same = (prev.get("emotion") == rec["emotion"]
                        and prev.get("trigger", "") == rec["trigger"])
                if same and rec["timestamp"] - prev.get("timestamp", 0) < 10.0:
                    return
            d.append(rec)
            if len(d) > cap:
                del d[:len(d) - cap]
        self._store.update(_mut)

    def recent(self, limit: int = 10) -> list[dict]:
        return self._effective_rows()[-limit:]

    def _effective_rows(self) -> list[dict]:
        """Ignore known legacy false positives/duplicates without rewriting data."""
        out = []
        for rec in self._store.load():
            trigger = rec.get("trigger", "")
            if (rec.get("emotion") == "sad"
                    and re.search(r"\bdown payment\b", trigger, re.I)):
                continue
            if out:
                prev = out[-1]
                same = (prev.get("emotion") == rec.get("emotion")
                        and prev.get("trigger", "") == trigger)
                if same and rec.get("timestamp", 0) - prev.get("timestamp", 0) < 10.0:
                    continue
            out.append(rec)
        return out

    def last(self, max_age_hours: float = 24.0) -> dict | None:
        rows = self._effective_rows()
        if not rows:
            return None
        rec = rows[-1]
        if now_ts() - rec.get("timestamp", 0) > max_age_hours * 3600:
            return None
        return rec

    def render(self) -> str:
        """A brief, non-clinical pattern summary for the prompt."""
        rows = self._effective_rows()[-30:]
        if not rows:
            return ""
        common = Counter(r["emotion"] for r in rows if r.get("emotion")).most_common(2)
        if not common:
            return ""
        latest = rows[-1]
        pat = ", ".join(f"{emo}" for emo, _ in common)
        trig = f" (recently around: {latest.get('trigger')})" if latest.get("trigger") else ""
        return (f"Recent emotional pattern: {pat}. "
                f"Last noted {human_age(latest.get('timestamp', 0))}{trig}.")
