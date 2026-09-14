"""
Layer 2 — Episodic Memory.

Stores meaningful life events as structured Episode records (the source of
truth). Semantic recall over episode summaries is handled by the shared vector
index (engine indexes each episode's summary with type="episode"); this store
holds the full records, keyed by id, for rendering and importance filtering.
"""
from __future__ import annotations

from assistant.memory.schema import Episode, JSONStore, human_age

_DEFAULT: list = []


class EpisodicMemory:
    def __init__(self, path: str):
        self._store = JSONStore(path, _DEFAULT)

    def add(self, ep: Episode, cap: int = 500) -> dict:
        rec = ep.to_dict()

        def _mut(d):
            d.append(rec)
            if len(d) > cap:
                del d[:len(d) - cap]
        self._store.update(_mut)
        return rec

    def all(self) -> list[dict]:
        return self._store.load()

    def by_ids(self, ids) -> list[dict]:
        want = set(ids or [])
        return [e for e in self.all() if e.get("id") in want]

    def recent(self, limit: int = 5, min_importance: int = 0) -> list[dict]:
        eps = [e for e in self.all() if e.get("importance", 0) >= min_importance]
        eps.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        return eps[:limit]

    @staticmethod
    def render(ep: dict) -> str:
        when = human_age(ep.get("timestamp", 0))
        return f"- ({when}) {ep.get('summary', '').strip()}"
