"""
One-time migration: old flat memory → the new multi-layer engine.

Reads the legacy files in DATA_DIR and folds them into the engine:
    user_name.txt / jarvis_memory.json  → profile.name
    memory.json (buckets)               → profile lists (likes/goals/...)
    conversation_history.json           → working memory turns + semantic index
    vector_memory.json                  → semantic index (re-embedded)
    mood_state.json                     → emotional memory

Idempotent: writes a .migrated marker in MEMORY_DIR and skips on re-run.

Run:  python -m assistant.memory.migrate     (from the jarvis/ folder)
"""
from __future__ import annotations

import json
import os

from assistant import config
from assistant.core.log import get
from assistant.memory.engine import get_engine
from assistant.memory.schema import Episode

log = get("memory.migrate")

_MARKER = "_migrated_v2.flag"


def _read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def migrate(force: bool = False) -> dict:
    eng = get_engine()
    marker = os.path.join(config.MEMORY_DIR, _MARKER)
    if os.path.exists(marker) and not force:
        log.info("migration already done (%s)", marker)
        return {"skipped": True}

    stats = {"name": 0, "profile_items": 0, "turns": 0, "vectors": 0, "moods": 0}
    dd = config.DATA_DIR

    # ---- name ----
    name = ""
    jm = _read_json(os.path.join(dd, "jarvis_memory.json"), {})
    if isinstance(jm, dict) and jm.get("name"):
        name = str(jm["name"]).strip()
    nf = os.path.join(dd, "user_name.txt")
    if not name and os.path.exists(nf):
        try:
            name = open(nf, encoding="utf-8").read().strip()
        except Exception:
            pass
    if name:
        eng.profile.set_scalar("name", name)
        stats["name"] = 1

    # ---- legacy buckets ----
    mem = _read_json(os.path.join(dd, "memory.json"), {})
    bucket_map = {"preferences": "likes", "dislikes": "dislikes", "goals": "goals",
                  "routines": "routines", "projects": None, "learned_topics": "interests"}
    if isinstance(mem, dict):
        for bucket, field in bucket_map.items():
            vals = mem.get(bucket) or []
            if not vals:
                continue
            if bucket == "projects":
                for v in vals:
                    eng.projects.upsert(str(v))
            elif field:
                # learned_topics is noisy single words; skip 1-2 char fragments.
                clean = [v for v in vals if len(str(v)) > 2]
                eng.profile.merge_list(field, clean)
            stats["profile_items"] += len(vals)

    # ---- conversation history → working memory + semantic index ----
    hist = _read_json(os.path.join(dd, "conversation_history.json"), [])
    idx = eng._get_index()
    if isinstance(hist, list):
        # Drop the known role-confusion noise ("user"/"assistant" as a message).
        clean = [h for h in hist if isinstance(h, dict)
                 and h.get("message", "").strip().lower() not in {"user", "assistant"}]
        for h in clean[-config.MEMORY_WORKING_TURNS:]:
            eng.working.add_turn(h.get("role", "user"), h.get("message", ""))
            stats["turns"] += 1
        # Index user/assistant pairs for recall.
        for i in range(0, len(clean) - 1, 2):
            u, a = clean[i], clean[i + 1]
            if u.get("role") == "user":
                idx.add(f"User: {u.get('message','')}\nJarvis: {a.get('message','')}",
                        {"type": "conversation", "importance": 4})
                stats["vectors"] += 1

    # ---- old vector memory → re-embed into the semantic index ----
    vm = _read_json(os.path.join(dd, "vector_memory.json"), [])
    if isinstance(vm, list):
        for row in vm[-200:]:
            text = (row.get("text") or "").strip() if isinstance(row, dict) else ""
            if text:
                idx.add(text, {"type": "conversation", "importance": 4})
                stats["vectors"] += 1

    # ---- mood ----
    ms = _read_json(os.path.join(dd, "mood_state.json"), {})
    if isinstance(ms, dict) and ms.get("mood"):
        eng.emotional.add(str(ms["mood"]), trigger="(migrated)")
        stats["moods"] = 1

    os.makedirs(config.MEMORY_DIR, exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    log.info("migration complete: %s", stats)
    return stats


if __name__ == "__main__":
    print(json.dumps(migrate(force=False), indent=2))
