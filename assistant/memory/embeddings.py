"""
Pluggable embedding backend (Layer 7 foundation).

Selection order for MEMORY_EMBED_BACKEND="auto":
  1. sentence-transformers  — local, free, fast (if installed)
  2. Gemini text-embedding   — reuses GOOGLE_API_KEY, no big download (if online)
  3. hashing                 — deterministic offline fallback, ALWAYS works

All vectors are L2-normalized, so a dot product == cosine similarity. The whole
backend runs inside the background memory worker thread, so a slow first call
(model download / network) never blocks a conversation turn.
"""
from __future__ import annotations

import hashlib
import re
import threading

import numpy as np

from assistant import config
from assistant.core.log import get

log = get("memory.embed")

_WORD = re.compile(r"[\wऀ-ॿఀ-౿]+", re.UNICODE)
_lock = threading.Lock()
_singleton = None


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class _HashingEmbedder:
    """Signed feature-hashing bag-of-words. Not semantic, but never fails and
    is still a strict upgrade over raw token-count cosine (fixed dim, hashed,
    normalized) so retrieval degrades gracefully when no real model is present."""
    name = "hash"
    dim = 256

    def encode(self, texts) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _WORD.findall((t or "").lower()):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        return _normalize(out)


class _STEmbedder:
    name = "sentence-transformers"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(config.MEMORY_EMBED_MODEL_ST)
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def encode(self, texts) -> np.ndarray:
        v = self._m.encode(list(texts), normalize_embeddings=True,
                           convert_to_numpy=True)
        return v.astype(np.float32)


class _GeminiEmbedder:
    name = "gemini"

    def __init__(self):
        from google import genai
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self.model = config.MEMORY_EMBED_MODEL_GEMINI
        # Probe once to learn the real dimensionality.
        probe = self._embed(["ok"])
        self.dim = int(probe.shape[1])

    def _embed(self, texts) -> np.ndarray:
        res = self._client.models.embed_content(model=self.model, contents=list(texts))
        vecs = [np.asarray(e.values, dtype=np.float32) for e in res.embeddings]
        return _normalize(np.vstack(vecs))

    def encode(self, texts) -> np.ndarray:
        return self._embed(texts)


def _build(backend: str):
    order = (["sentence-transformers", "gemini", "hash"] if backend == "auto"
             else [backend])
    for name in order:
        try:
            if name == "sentence-transformers":
                e = _STEmbedder()
            elif name == "gemini":
                if not config.GOOGLE_API_KEY:
                    continue
                e = _GeminiEmbedder()
            else:
                e = _HashingEmbedder()
            log.info("embedding backend = %s (dim=%d)", e.name, e.dim)
            return e
        except Exception:
            log.info("embedding backend %s unavailable, trying next", name, exc_info=True)
    log.info("falling back to hashing embedder")
    return _HashingEmbedder()


def get_embedder():
    """Process-wide singleton embedder (built lazily on first use)."""
    global _singleton
    if _singleton is None:
        with _lock:
            if _singleton is None:
                _singleton = _build(config.MEMORY_EMBED_BACKEND)
    return _singleton
