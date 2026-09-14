import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from assistant import config
from assistant.memory import extractor
from assistant.memory import store
from assistant.memory.emotional import EmotionalMemory
from assistant.memory.engine import MemoryEngine
from assistant.memory.vector_index import VectorIndex


class _FakeEmbedder:
    name = "fake"
    dim = 4

    def encode(self, texts):
        rows = []
        for text in texts:
            seed = sum(ord(c) for c in text)
            row = np.array([seed % 7 + 1, seed % 11 + 1, 1, 2], dtype=np.float32)
            rows.append(row / np.linalg.norm(row))
        return np.vstack(rows)


class MemoryRegressionTests(unittest.TestCase):
    def test_legacy_migration_runs_only_for_fresh_v2_store(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(config, "MEMORY_DIR", str(Path(tmp) / "v2")), \
                patch.object(config, "MEMORY_FILE", str(Path(tmp) / "memory.json")), \
                patch.object(config, "CONVERSATION_HISTORY_FILE", str(Path(tmp) / "history.json")), \
                patch.object(config, "VECTOR_MEMORY_FILE", str(Path(tmp) / "vectors.json")), \
                patch.object(config, "MOOD_FILE", str(Path(tmp) / "mood.json")), \
                patch("assistant.memory.migrate.migrate") as migrate:
            Path(config.MEMORY_FILE).write_text("{}", encoding="utf-8")
            store.ensure_legacy_migrated()
            migrate.assert_called_once_with(force=False)
            migrate.reset_mock()
            Path(config.MEMORY_DIR).mkdir()
            (Path(config.MEMORY_DIR) / "profile.json").write_text("{}", encoding="utf-8")
            store.ensure_legacy_migrated()
            migrate.assert_not_called()

    def test_emotion_markers_do_not_match_word_fragments(self):
        self.assertIsNone(extractor.detect_emotion("What is the down payment?"))
        self.assertEqual(extractor.detect_emotion("I feel down today")[0], "sad")

    def test_consecutive_duplicate_emotions_are_not_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = EmotionalMemory(str(Path(tmp) / "emotions.json"))
            memory.add("happy", trigger="good news")
            memory.add("happy", trigger="good news")
            self.assertEqual(len(memory.recent()), 1)

    def test_legacy_false_emotions_are_ignored_without_rewriting_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "emotions.json"
            path.write_text(
                '[{"emotion":"sad","intensity":"medium","timestamp":1,'
                '"trigger":"minimum down payment"},'
                '{"emotion":"happy","intensity":"medium","timestamp":2,'
                '"trigger":"good news"},'
                '{"emotion":"happy","intensity":"medium","timestamp":3,'
                '"trigger":"good news"}]', encoding="utf-8")
            memory = EmotionalMemory(str(path))
            self.assertEqual([row["emotion"] for row in memory.recent()], ["happy"])
            self.assertEqual(len(json.loads(path.read_text())), 3)

    def test_vector_index_deduplicates_and_caps_entries(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                config, "MEMORY_VECTOR_MAX_ENTRIES", 100):
            index = VectorIndex(str(Path(tmp) / "semantic"), _FakeEmbedder())
            index.add("same memory", {"type": "conversation"})
            index.add("same memory", {"type": "conversation"})
            self.assertEqual(len(index), 1)
            for number in range(105):
                index.add(f"memory {number}", {"type": "conversation"})
            self.assertEqual(len(index), 100)

    def test_rank_filters_low_similarity(self):
        hits = [
            {"text": "relevant", "importance": 5, "score": 0.7},
            {"text": "noise", "importance": 9, "score": 0.1},
        ]
        ranked = MemoryEngine._rank(
            hits, recall=False, min_importance=4, min_similarity=0.18)
        self.assertEqual([row["text"] for row in ranked], ["relevant"])

    def test_warmup_loads_index_and_flush_stops_worker(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(config, "MEMORY_DIR", tmp), \
                patch("assistant.memory.engine.embeddings.get_embedder",
                      return_value=_FakeEmbedder()):
            engine = MemoryEngine()
            engine.warmup()
            self.assertTrue(engine._index_ready.wait(timeout=1.0))
            self.assertIsNotNone(engine._index_if_ready())
            engine.flush(timeout=1.0)
            self.assertFalse(engine._worker.is_alive())


if __name__ == "__main__":
    unittest.main()
