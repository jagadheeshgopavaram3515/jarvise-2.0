"""
MemoryEngine — orchestrates all seven memory layers.

Design constraints (so NOTHING in the protected architecture changes):
  * The WRITE path called from the dispatcher thread (record_exchange) is cheap
    and synchronous only for Working + Relationship memory. Everything expensive
    (embedding, extraction, consolidation) is pushed to ONE background daemon
    thread — additive, consistent with the existing producer/consumer model,
    and it never blocks a turn.
  * The READ path (build_context) assembles a relevance-ranked, budget-limited
    context string. It does not dump everything.

Layer map:
  1 Working      working.WorkingMemory
  2 Episodic     episodic.EpisodicMemory  (+ semantic index, type="episode")
  3 Profile      profile.UserProfile
  4 Relationship relationship.RelationshipMemory
  5 Projects     projects.ProjectMemory   (+ semantic index, type="project")
  6 Emotional    emotional.EmotionalMemory
  7 Semantic     vector_index.VectorIndex (conversations/episodes/projects/facts)
"""
from __future__ import annotations

import os
import queue
import threading

from assistant import config
from assistant.core.log import get
from assistant.memory import embeddings, extractor
from assistant.memory.emotional import EmotionalMemory
from assistant.memory.episodic import EpisodicMemory
from assistant.memory.profile import UserProfile
from assistant.memory.projects import ProjectMemory
from assistant.memory.relationship import RelationshipMemory
from assistant.memory.schema import Episode
from assistant.memory.vector_index import VectorIndex
from assistant.memory.working import WorkingMemory

log = get("memory.engine")


def _p(name: str) -> str:
    return os.path.join(config.MEMORY_DIR, name)


class MemoryEngine:
    def __init__(self):
        d = config.MEMORY_DIR
        os.makedirs(d, exist_ok=True)
        self.working = WorkingMemory(_p("working.json"))
        self.profile = UserProfile(_p("profile.json"))
        self.projects = ProjectMemory(_p("projects.json"))
        self.episodic = EpisodicMemory(_p("episodes.json"))
        self.emotional = EmotionalMemory(_p("emotions.json"))
        self.relationship = RelationshipMemory(_p("relationship.json"))

        self._index: VectorIndex | None = None   # lazy: built ONLY in the worker
        self._index_lock = threading.Lock()
        self._jobs: "queue.Queue" = queue.Queue(maxsize=256)
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._index_ready = threading.Event()
        self._turns_since_consolidation = 0
        self._counter_lock = threading.Lock()

    def warmup(self) -> None:
        """Load the persisted index off the turn thread for first-turn recall."""
        self._ensure_worker()
        try:
            self._jobs.put_nowait(("warmup",))
        except queue.Full:
            pass

    def flush(self, timeout: float = 2.0) -> None:
        """Process queued memory writes before the application's forced exit."""
        worker = self._worker
        if worker is None or not worker.is_alive():
            return
        try:
            self._jobs.put(("stop",), timeout=max(0.1, timeout / 2))
            worker.join(timeout=max(0.1, timeout))
        except Exception:
            log.debug("memory worker flush failed", exc_info=True)

    # ================================================================ WRITE
    def record_exchange(self, user_text: str, assistant_text: str,
                        emotion: str | None = None) -> None:
        """Hot-path entry (dispatcher thread). Cheap + synchronous, then defer."""
        user_text = (user_text or "").strip()
        if not user_text:
            return
        # Layer 1 + 4 synchronously so the very next turn already has continuity.
        self.working.add_turn("user", user_text)
        if assistant_text:
            self.working.add_turn("assistant", assistant_text)
        if emotion:
            self.working.set_state(emotion=emotion)
        self.relationship.note_turn()
        # Defer the expensive work.
        self._ensure_worker()
        try:
            self._jobs.put_nowait(("exchange", user_text, assistant_text))
        except queue.Full:
            log.debug("memory job queue full; dropping background processing")

    # =============================================================== WORKER
    def _ensure_worker(self) -> None:
        if self._worker is not None:
            return
        with self._worker_lock:
            if self._worker is None:
                t = threading.Thread(target=self._run, name="MemoryWorker", daemon=True)
                self._worker = t
                t.start()

    def _get_index(self) -> VectorIndex:
        """Build-if-needed accessor. MUST only be called from the background
        worker — building the embedder can download a model / hit the network,
        which would freeze a conversation turn if done on the dispatcher thread."""
        if self._index is None:
            with self._index_lock:
                if self._index is None:
                    self._index = VectorIndex(_p("semantic"), embeddings.get_embedder())
                    self._index_ready.set()
        return self._index

    def _index_if_ready(self) -> VectorIndex | None:
        """Non-blocking accessor for the READ path (turn thread). Returns the
        index only once the worker has built it; never triggers construction."""
        return self._index

    def _run(self) -> None:
        log.info("memory worker started")
        while True:
            try:
                job = self._jobs.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                kind = job[0]
                if kind == "exchange":
                    self._process_exchange(job[1], job[2])
                elif kind == "warmup":
                    self._get_index()
                elif kind == "stop":
                    return
            except Exception:
                log.debug("memory worker job failed", exc_info=True)

    def _process_exchange(self, user_text: str, assistant_text: str) -> None:
        idx = self._get_index()
        # 1) Always index the conversation turn for semantic recall.
        importance = extractor.score_exchange(user_text, assistant_text)
        idx.add(f"User: {user_text}\nJarvis: {assistant_text}",
                {"type": "conversation", "importance": importance})

        # 2) Cheap emotion read → emotional timeline + working state.
        emo = extractor.detect_emotion(user_text)
        if emo:
            self.emotional.add(emo[0], intensity=emo[1], trigger=user_text[:80])
            self.working.set_state(emotion=emo[0])

        # 3) Project name mentioned? touch its record so "last_discussed" is fresh.
        for rec in self.projects.find_mentioned(user_text):
            self.projects.upsert(rec["project_name"])

        # 4) Immediate user name update if stated directly by user.
        detected_name = extractor.extract_user_name(user_text)
        if detected_name:
            self.profile.set_scalar("name", detected_name)
            try:
                with open(config.NAME_FILE, "w", encoding="utf-8") as f:
                    f.write(detected_name)
            except Exception:
                pass
            log.info("immediately updated user name in memory to %r", detected_name)

        # 4) Periodic consolidation (LLM) — the deep extraction pass.
        with self._counter_lock:
            self._turns_since_consolidation += 1
            due = self._turns_since_consolidation >= config.MEMORY_CONSOLIDATE_EVERY
            if due:
                self._turns_since_consolidation = 0
        if due:
            self._consolidate()

    # ========================================================= CONSOLIDATION
    def _consolidate(self) -> None:
        turns = self.working.turns(limit=config.MEMORY_CONSOLIDATE_EVERY * 2)
        if not turns:
            return
        log.info("consolidating %d turns", len(turns))
        delta = extractor.consolidate(turns)
        idx = self._get_index()

        # Session summary → working topic + a fact in the index.
        summary = (delta.get("session_summary") or "").strip()
        if summary:
            self.working.set_state(topic=summary[:120])
            idx.add(summary, {"type": "fact", "importance": 5})

        # Profile updates.
        self.profile.apply_updates(delta.get("profile_updates") or {})

        # Projects.
        for pr in delta.get("projects") or []:
            name = (pr.get("project_name") or "").strip()
            if name:
                self.projects.upsert(
                    name, status=pr.get("status") or None,
                    goals=pr.get("goals"), blockers=pr.get("blockers"),
                    next_steps=pr.get("next_steps"))
                idx.add(f"Project {name}: {pr.get('status', '')} "
                        f"{' '.join(pr.get('next_steps') or [])}",
                        {"type": "project", "importance": 6})

        # Episodes — store + index only the important ones (>= threshold).
        for ep in delta.get("episodes") or []:
            imp = int(ep.get("importance", 0) or 0)
            if imp < config.MEMORY_IMPORTANCE_THRESHOLD:
                continue
            episode = Episode(summary=(ep.get("summary") or "").strip(),
                              importance=imp, tags=ep.get("tags") or [],
                              emotion=ep.get("emotion") or "")
            if not episode.summary:
                continue
            self.episodic.add(episode)
            idx.add(episode.summary, {"type": "episode", "importance": imp,
                                      "ref_id": episode.id})
            if imp >= 8:   # truly major → relationship milestone
                self.relationship.add_milestone(episode.summary)

        # Emotion reading.
        emo = delta.get("emotion") or {}
        if emo.get("emotion"):
            self.emotional.add(emo["emotion"], intensity=emo.get("intensity", "medium"),
                               trigger=emo.get("trigger", ""))

    # ================================================================ READ
    def build_context(self, query: str, recall: bool = False,
                      max_chars: int | None = None) -> str:
        """Assemble a relevance-ranked, budget-limited memory context block."""
        budget = max_chars or config.MEMORY_CONTEXT_MAX_CHARS
        blocks: list[str] = []

        # Relationship (cheap, grounding, always first).
        rel = self.relationship.render()
        if rel:
            blocks.append("Relationship:\n" + rel)

        # User profile.
        prof = self.profile.render()
        if prof:
            blocks.append("About the user:\n" + prof)

        # Active / mentioned projects.
        mentioned = self.projects.find_mentioned(query)
        active = mentioned or self.projects.active(limit=3)
        if active:
            rendered = "\n".join(self.projects.render_one(r) for r in active[:3])
            blocks.append("Projects:\n" + rendered)

        # Semantic recall: episodes + facts + past conversations.
        # Uses the index ONLY if the worker has already built it — never builds
        # it here, so the turn thread can't block on embedder init/download.
        idx = self._index_if_ready()
        if idx is not None and len(idx):
            try:
                min_imp = 0 if recall else 4
                top_k = config.MEMORY_RETRIEVAL_TOP_K * (2 if recall else 1)
                hits = idx.search(query, top_k=top_k * 2,
                                  type_filter={"episode", "fact", "project", "conversation"})
                min_similarity = (config.MEMORY_RECALL_MIN_SIMILARITY if recall
                                  else config.MEMORY_MIN_SIMILARITY)
                ranked = self._rank(hits, recall=recall, min_importance=min_imp,
                                    min_similarity=min_similarity)
                if ranked:
                    lines = [f"- {h.get('text', '').strip()[:200]}" for h in ranked[:top_k]]
                    blocks.append("Relevant memories (most relevant first):\n" + "\n".join(lines))
            except Exception:
                log.debug("semantic retrieval failed", exc_info=True)

        # Recent important episodes even if not semantically matched (continuity).
        recents = self.episodic.recent(limit=3, min_importance=6)
        if recents:
            blocks.append("Recent milestones:\n" +
                          "\n".join(self.episodic.render(e) for e in recents))

        # Emotional pattern (kept short, non-clinical).
        emo = self.emotional.render()
        if emo:
            blocks.append(emo)

        # Working-memory state header.
        wm = self.working.render()
        if wm:
            blocks.append("Working memory: " + wm)

        return self._budget(blocks, budget)

    @staticmethod
    def _rank(hits: list[dict], recall: bool, min_importance: int,
              min_similarity: float = 0.0) -> list[dict]:
        scored = []
        for h in hits:
            imp = int(h.get("importance", 5) or 5)
            if imp < min_importance and not recall:
                continue
            sim = float(h.get("score", 0.0))
            if sim < min_similarity:
                continue
            # similarity dominates; importance gently boosts; recall widens.
            rank = sim * (0.7 + 0.06 * imp)
            scored.append((rank, h))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in scored]

    @staticmethod
    def _budget(blocks: list[str], budget: int) -> str:
        out, used = [], 0
        for b in blocks:
            if used + len(b) > budget:
                continue
            out.append(b)
            used += len(b) + 2
        return "\n\n".join(out)


# Process-wide singleton (so the worker/index aren't duplicated).
_engine: MemoryEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> MemoryEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = MemoryEngine()
                _engine.warmup()
    return _engine
