from __future__ import annotations

import hashlib
import json
from pathlib import Path

from assistant.orchestration.decision import Capability, DecisionRequest
from benchmarks.fast_classifier.evaluate import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks" / "fast_classifier" / "v4" / "dataset.jsonl"
MANIFEST = ROOT / "benchmarks" / "fast_classifier" / "v4" / "split_manifest.json"


def _raw_rows():
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]


def test_v4_dataset_integrity_and_review_status():
    dataset = load_dataset(DATASET)
    rows = _raw_rows()
    assert dataset.schema_version == "4.0"
    assert len(rows) == 434
    assert len({row["id"] for row in rows}) == len(rows)
    assert {row["annotation_status"] for row in rows} == {"pending_review"}
    assert {row["review_status"] for row in rows} == {"pending_review"}


def test_v4_manifest_hash_and_group_split_isolation():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["dataset_sha256"] == hashlib.sha256(DATASET.read_bytes()).hexdigest()
    rows = _raw_rows()
    observed = {}
    for row in rows:
        observed.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in observed.values())
    assert set(observed) == set(manifest["groups"])


def test_v4_sealed_test_has_required_language_slices_and_no_v3_test_ids():
    test = load_dataset(DATASET, split="test")
    languages = {}
    for example in test.examples:
        languages[example.language] = languages.get(example.language, 0) + 1
    assert languages == {
        "en": 20, "te": 20, "hi": 20, "te-Latn": 20, "hi-Latn": 20,
        "mixed-en-te": 20, "mixed-en-hi": 20,
    }
    v3_test_ids = {example.example_id for example in load_dataset(
        ROOT / "benchmarks" / "fast_classifier" / "v3" / "dataset.jsonl", split="test"
    ).examples}
    assert not v3_test_ids.intersection(example.example_id for example in test.examples)


def test_v4_capabilities_are_real_registry_names():
    from assistant.tools.registry import build_default_registry

    registry_names = set(build_default_registry().names())
    advertised = {cap.name for example in load_dataset(DATASET).examples for cap in example.capabilities}
    assert advertised <= registry_names


def test_laya_adapter_uses_one_worker_request_and_never_executes_tools(monkeypatch):
    import assistant.orchestration.laya_adapter as module

    writes = []

    class Stream:
        def __init__(self):
            self.lines = iter([
                json.dumps({"status": "ready", "adapter_id": "test-laya"}) + "\n",
                json.dumps({"status": "success", "decision": {
                    "intent": "use:open_app", "execution_path": "single_action",
                    "request_modes": ["action"], "selected_capability": "open_app",
                    "context_references": [], "missing_information": [], "confidence": 0.8,
                    "needs_clarification": False, "needs_escalation": False, "evidence": [],
                }, "timings": {"model_inference_ms": 1.0}}) + "\n",
            ])

        def readline(self):
            return next(self.lines)

        def write(self, value):
            writes.append(value)

        def flush(self):
            pass

    class Process:
        def __init__(self, *args, **kwargs):
            self.stdin = Stream(); self.stdout = Stream(); self.stderr = Stream(); self.returncode = 0

        def poll(self): return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    adapter = module.LayaOfflineAdapter("python", "worker")
    output = adapter.predict(DecisionRequest("open calculator", capabilities=(Capability("open_app", "Open app", ("app",)),)))
    assert output["selected_capability"] == "open_app"
    assert len(writes) == 1

