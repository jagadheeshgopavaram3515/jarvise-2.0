"""
Layer 3 — User Profile Memory.

A living document of who the user is. Scalar fields (name, age, location...) are
set/overwritten; list fields (projects, skills, goals, likes...) are merged and
de-duplicated. Updated automatically by the extractor; never blocks a turn.
"""
from __future__ import annotations

from assistant.memory.schema import JSONStore

# Canonical fields. List fields accumulate; the rest are scalars.
LIST_FIELDS = {
    "projects", "skills", "goals", "dreams", "interests", "likes", "dislikes",
    "favorite_movies", "favorite_anime", "favorite_books",
    "career_aspirations", "routines", "habits", "languages", "relationships",
}
SCALAR_FIELDS = {"name", "age", "location", "education", "role"}

_DEFAULT = {f: [] for f in LIST_FIELDS}
_DEFAULT.update({f: "" for f in SCALAR_FIELDS})

_CAP = 25  # max items kept per list field


class UserProfile:
    def __init__(self, path: str):
        self._store = JSONStore(path, _DEFAULT)

    def get(self) -> dict:
        return self._store.load()

    def set_scalar(self, field: str, value: str) -> None:
        if field not in SCALAR_FIELDS or not value:
            return
        self._store.update(lambda d: d.__setitem__(field, str(value).strip()))

    def merge_list(self, field: str, values) -> None:
        if field not in LIST_FIELDS:
            return
        vals = [str(v).strip() for v in values if str(v).strip()]
        if not vals:
            return

        def _mut(d):
            cur = d.setdefault(field, [])
            # Case-insensitive dedupe, newest kept.
            seen = {x.lower() for x in cur}
            for v in vals:
                if v.lower() not in seen:
                    cur.append(v)
                    seen.add(v.lower())
            d[field] = cur[-_CAP:]
        self._store.update(_mut)

    def apply_updates(self, updates: dict) -> None:
        """Apply a dict of {field: value|list} from the extractor."""
        for field, value in (updates or {}).items():
            if field in SCALAR_FIELDS:
                self.set_scalar(field, value)
            elif field in LIST_FIELDS:
                self.merge_list(field, value if isinstance(value, list) else [value])

    def render(self, max_chars: int = 700) -> str:
        d = self.get()
        lines = []
        for f in ("name", "role", "location", "education", "age"):
            if d.get(f):
                lines.append(f"{f}: {d[f]}")
        for f in ("skills", "goals", "career_aspirations", "interests",
                  "likes", "dislikes", "favorite_anime", "favorite_movies",
                  "favorite_books", "languages", "routines", "relationships"):
            vals = d.get(f) or []
            if vals:
                lines.append(f"{f}: " + ", ".join(map(str, vals[-6:])))
        text = "\n".join(lines)
        return text[:max_chars]
