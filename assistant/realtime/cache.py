"""
Step 7 — Caching.

Realtime context is cached on disk at ``cache/<query_hash>.json`` keyed by a
hash of (normalised query + query class). TTL depends on the class so we don't
re-hit the search API for the same question:

    news    = 30 min      weather = 15 min      general = 12 h

A read past its TTL is a miss (the stale file is ignored, and lazily replaced on
the next write). The cache is best-effort: any disk error degrades to "no cache"
rather than failing the lookup.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

from assistant import config
from assistant.core.log import get

log = get("realtime")


def _key(query: str, kind: str) -> str:
    norm = " ".join((query or "").lower().split())
    return hashlib.sha256(f"{kind}:{norm}".encode("utf-8")).hexdigest()[:24]


def _path(query: str, kind: str) -> str:
    return os.path.join(config.REALTIME_CACHE_DIR, f"{_key(query, kind)}.json")


def get_cached(query: str, kind: str, ttl_s: int) -> str | None:
    """Return cached context if present and fresher than ttl_s, else None."""
    path = _path(query, kind)
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        age = time.time() - float(data.get("ts", 0))
        if age > ttl_s:
            return None
        return data.get("context") or None
    except Exception as e:
        log.debug("[REALTIME] cache read failed: %s", e)
        return None


def put_cached(query: str, kind: str, context: str) -> None:
    """Persist context for a query; best-effort (errors are swallowed)."""
    try:
        os.makedirs(config.REALTIME_CACHE_DIR, exist_ok=True)
        with open(_path(query, kind), "w", encoding="utf-8") as f:
            json.dump(
                {"query": query, "kind": kind, "ts": time.time(),
                 "context": context},
                f, ensure_ascii=False,
            )
    except Exception as e:
        log.debug("[REALTIME] cache write failed: %s", e)
