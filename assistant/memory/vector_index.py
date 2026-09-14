"""
Semantic vector index (Layer 7) — replaces the old bag-of-words scorer.

One typed index stores embeddings for conversations, episodes, projects and
facts (metadata["type"] distinguishes them). Backed by FAISS (IndexFlatIP on
normalized vectors == cosine) when installed, else a NumPy brute-force scan —
identical results, just slower at large scale (fine for a personal assistant).

Persistence: a float32 matrix (.npy) + a parallel metadata list (.meta.json).
On load the FAISS index is rebuilt from the matrix, so the on-disk format is
backend-independent and human-inspectable.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np

from assistant import config
from assistant.core.log import get

log = get("memory.index")

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


class VectorIndex:
    def __init__(self, base_path: str, embedder):
        self.base = base_path
        self.embedder = embedder
        self.dim = embedder.dim
        self._lock = threading.RLock()
        self._vecs = np.zeros((0, self.dim), dtype=np.float32)
        self._meta: list[dict] = []
        self._faiss = None
        self._load()

    # ---- paths ----------------------------------------------------------
    @property
    def _vec_path(self) -> str:
        return self.base + ".npy"

    @property
    def _meta_path(self) -> str:
        return self.base + ".meta.json"

    # ---- persistence ----------------------------------------------------
    def _load(self) -> None:
        try:
            if os.path.exists(self._vec_path) and os.path.exists(self._meta_path):
                vecs = np.load(self._vec_path)
                with open(self._meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                # Dimensionality mismatch (e.g. embedder changed) → start fresh
                # rather than corrupt search. Re-indexing happens over time.
                if vecs.ndim == 2 and vecs.shape[1] == self.dim and len(meta) == len(vecs):
                    self._vecs = vecs.astype(np.float32)
                    self._meta = meta
                else:
                    log.info("vector index reset (dim %s != %d or length mismatch)",
                             getattr(vecs, "shape", None), self.dim)
        except Exception:
            log.info("vector index load failed; starting empty", exc_info=True)
        self._rebuild_faiss()

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.base) or ".", exist_ok=True)
            np.save(self._vec_path, self._vecs)
            tmp = self._meta_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, ensure_ascii=False)
            os.replace(tmp, self._meta_path)
        except Exception:
            log.debug("vector index save failed", exc_info=True)

    def _rebuild_faiss(self) -> None:
        if not _HAS_FAISS:
            self._faiss = None
            return
        try:
            idx = faiss.IndexFlatIP(self.dim)
            if len(self._vecs):
                idx.add(self._vecs)
            self._faiss = idx
        except Exception:
            self._faiss = None

    # ---- writes ---------------------------------------------------------
    def add(self, text: str, metadata: dict) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            # Retries and migration reruns must not grow identical vectors.
            if any(m.get("text") == text for m in self._meta):
                return
            vec = self.embedder.encode([text])[0:1]
            meta = dict(metadata)
            meta.setdefault("text", text)
            self._vecs = np.vstack([self._vecs, vec]) if len(self._vecs) else vec.copy()
            self._meta.append(meta)
            cap = max(100, config.MEMORY_VECTOR_MAX_ENTRIES)
            rebuilt = False
            if len(self._meta) > cap:
                overflow = len(self._meta) - cap
                self._meta = self._meta[overflow:]
                self._vecs = self._vecs[overflow:]
                self._rebuild_faiss()
                rebuilt = True
            if self._faiss is not None and not rebuilt:
                try:
                    self._faiss.add(vec)
                except Exception:
                    self._rebuild_faiss()
            self._save()

    # ---- reads ----------------------------------------------------------
    def search(self, query: str, top_k: int = 6,
               type_filter: set[str] | None = None) -> list[dict]:
        """Return up to top_k metadata dicts, each with an added 'score'."""
        query = (query or "").strip()
        with self._lock:
            if not query or len(self._vecs) == 0:
                return []
            q = self.embedder.encode([query]).astype(np.float32)
            # Over-fetch so a type filter still yields top_k matches.
            k = min(len(self._vecs), max(top_k * 4, top_k))
            if self._faiss is not None:
                scores, idxs = self._faiss.search(q, k)
                pairs = [(float(scores[0][j]), int(idxs[0][j]))
                         for j in range(len(idxs[0])) if idxs[0][j] != -1]
            else:
                sims = (self._vecs @ q[0])
                top = np.argsort(-sims)[:k]
                pairs = [(float(sims[i]), int(i)) for i in top]

            out = []
            for score, i in pairs:
                meta = dict(self._meta[i])
                if type_filter and meta.get("type") not in type_filter:
                    continue
                meta["score"] = score
                out.append(meta)
                if len(out) >= top_k:
                    break
            return out

    def __len__(self) -> int:
        return len(self._meta)
