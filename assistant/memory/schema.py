"""
Data schemas + atomic JSON persistence for the multi-layer memory system.

Everything the memory layers persist goes through JSONStore, which writes
atomically (temp file + os.replace) so a crash mid-write can never corrupt a
memory file. All timestamps are float epoch seconds; helpers render ISO for
prompts.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime


# ----------------------------------------------------------------- helpers
def now_ts() -> float:
    return time.time()


def iso(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time()).isoformat(timespec="seconds")


def human_age(ts: float) -> str:
    """A natural 'how long ago' string for relationship/recall phrasing."""
    secs = max(0.0, time.time() - ts)
    days = secs / 86400.0
    if days < 1:
        hours = int(secs // 3600)
        return "earlier today" if hours < 12 else "today"
    if days < 2:
        return "yesterday"
    if days < 14:
        return f"{int(days)} days ago"
    if days < 60:
        return f"{int(days // 7)} weeks ago"
    if days < 730:
        return f"{int(days // 30)} months ago"
    return f"{int(days // 365)} years ago"


def new_id(prefix: str = "m") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ----------------------------------------------------------------- JSON store
class JSONStore:
    """Thread-safe, atomic whole-document JSON file."""

    def __init__(self, path: str, default):
        self.path = path
        self._default = default
        self._lock = threading.RLock()

    def load(self):
        with self._lock:
            if not os.path.exists(self.path):
                return json.loads(json.dumps(self._default))
            try:
                with open(self.path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return json.loads(json.dumps(self._default))

    def save(self, data) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self.path)
            except Exception:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

    def update(self, fn):
        """Load → mutate in place via fn(data) → save. Returns the new data."""
        with self._lock:
            data = self.load()
            fn(data)
            self.save(data)
            return data


# ----------------------------------------------------------------- records
@dataclass
class Episode:
    """A meaningful life event worth remembering long-term (Layer 2)."""
    id: str = field(default_factory=lambda: new_id("ep"))
    timestamp: float = field(default_factory=now_ts)
    summary: str = ""
    importance: int = 5          # 0-10
    tags: list = field(default_factory=list)
    emotion: str = ""
    source_turn: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EmotionRecord:
    """A point in the user's emotional timeline (Layer 6)."""
    emotion: str = ""
    intensity: str = "medium"    # low | medium | high
    timestamp: float = field(default_factory=now_ts)
    trigger: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
