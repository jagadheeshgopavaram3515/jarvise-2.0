"""Offline candidate and v2 dataset tests; no real encoder or tool is invoked."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from assistant.orchestration.compact_adapter import CompactMultilingualAdapter
from assistant.orchestration.decision import (
    Capability,
    DecisionModelAdapter,
    DecisionRequest,
    FastClassifier,
)
from benchmarks.fast_classifier.evaluate import load_dataset


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "benchmarks" / "fast_classifier" / "v1" / "seed.jsonl"
V1_INVALID = ROOT / "benchmarks" / "fast_classifier" / "v1" / "invalid_outputs.jsonl"
V2 = ROOT / "benchmarks" / "fast_classifier" / "v2" / "dataset.jsonl"
MANIFEST = ROOT / "benchmarks" / "fast_classifier" / "v2" / "split_manifest.json"
V3 = ROOT / "benchmarks" / "fast_classifier" / "v3" / "dataset.jsonl"
V3_MANIFEST = ROOT / "benchmarks" / "fast_classifier" / "v3" / "split_manifest.json"
V3_META = ROOT / "benchmarks" / "fast_classifier" / "models" / "v3" / "compact_heads.json"
V1_BASELINE = ROOT / "benchmark-results" / "legacy-qwen-v1.json"


class StaticModel:
    def __init__(self, prediction, classes, probabilities):
        self.prediction = prediction
        self.classes_ = np.array(classes)
        self.probabilities = np.array(probabilities, dtype=float)

    def predict(self, vectors):
        return np.array([self.prediction] * len(vectors))

    def predict_proba(self, vectors):
        return np.stack([self.probabilities] * len(vectors))


class FakeEncoder:
    def __init__(self):
        self.forward_calls = 0
        self.last_timings = {
            "preprocessing_ms": 0.1,
            "model_inference_ms": 0.2,
        }

    def encode(self, texts):
        self.forward_calls += 1
        return np.ones((len(texts), 2), dtype=float)


def _binary(positive_probability):
    return StaticModel(
        int(positive_probability >= 0.5),
        [0, 1],
        [1.0 - positive_probability, positive_probability],
    )


def _adapter(
    *,
    path="single_action",
    clarification=0.1,
    escalation=0.1,
    reference_kind="__none__",
    reference_resolved=0.1,
    capability_probability=0.9,
):
    adapter = CompactMultilingualAdapter.__new__(CompactMultilingualAdapter)
    adapter.artifact_path = "mock"
    adapter.encoder = FakeEncoder()
    adapter._adapter_id = "mock-compact"
    adapter.inference_count = 0
    adapter.last_inference_metrics = {}
    adapter.artifact = {
        "execution_path": StaticModel(path, [path, "direct_response"], [0.9, 0.1]),
        "request_modes": {
            "conversational": _binary(0.1),
            "memory": _binary(0.1),
            "research": _binary(0.1),
            "action": _binary(0.9),
        },
        "clarification": _binary(clarification),
        "escalation": _binary(escalation),
        "missing_information": StaticModel("target", ["target"], [1.0]),
        "reference_kind": StaticModel(reference_kind, [reference_kind], [1.0]),
        "reference_resolved": _binary(reference_resolved),
        "capability_matcher": StaticModel(1, [0, 1], [1.0 - capability_probability, capability_probability]),
        "capability_threshold": 0.5,
        "mode_threshold": 0.5,
        "path_default_mode": {path: "action", "clarify": "action", "escalate": "action"},
    }
    return adapter


def _request(capabilities=True, context=None):
    caps = (
        Capability("open_app", "Open an installed desktop application.", ("app",)),
        Capability("google_search", "Search the web.", ("query",)),
    ) if capabilities else ()
    return DecisionRequest("Open the calculator", context=context or {}, capabilities=caps)


def test_v1_artifacts_are_byte_preserved():
    assert hashlib.sha256(V1.read_bytes()).hexdigest() == "d8bb2f891b9de700962dc753491d24cf5341ae337eeb31ae87a422363ea3eb7d"
    assert hashlib.sha256(V1_INVALID.read_bytes()).hexdigest() == "5311beb91c141dc798adeef8b8f0978a957b99628e0b90abdc7374c6606bdb7e"
    assert hashlib.sha256(V1_BASELINE.read_bytes()).hexdigest() == "56525c079db0242d705f9352602b90034a6187011f777565de07510d1c78d34c"


def test_v2_counts_ids_versions_and_provenance():
    records = [json.loads(line) for line in V2.read_text(encoding="utf-8").splitlines()]
    ids = [record["id"] for record in records]

    assert len(records) == 228
    assert len(ids) == len(set(ids))
    assert all(record["schema_version"] == "2.0" for record in records)
    assert sum("derived_from_id" in record for record in records) == 28
    assert sum(record["provenance"] == "template_generated" for record in records) == 200
    assert all(record["review_status"] == "pending_human_review" for record in records)
    assert not any(record["annotation_status"] == "reviewed" for record in records)


def test_all_v1_ids_have_traceable_v2_derivations():
    v1_ids = {json.loads(line)["id"] for line in V1.read_text(encoding="utf-8").splitlines()}
    v2_records = [json.loads(line) for line in V2.read_text(encoding="utf-8").splitlines()]
    derived = {record["derived_from_id"] for record in v2_records if "derived_from_id" in record}

    assert derived == v1_ids
    assert (V2.parent / "ADJUDICATION.md").exists()


def test_manifest_prevents_group_split_leakage_and_seals_test():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in V2.read_text(encoding="utf-8").splitlines()]
    split_by_group = {}
    for record in records:
        split_by_group.setdefault(record["group_id"], set()).add(record["split"])

    assert all(len(splits) == 1 for splits in split_by_group.values())
    assert all(
        record["provenance"] != "template_generated"
        for record in records
        if record["split"] == "test"
    )
    assert manifest["source_v1_sha256"] == hashlib.sha256(V1.read_bytes()).hexdigest()


def test_training_metadata_excludes_dev_and_sealed_test_ids():
    metadata = json.loads(
        (ROOT / "benchmarks" / "fast_classifier" / "models" / "v2" / "compact_heads.json").read_text(encoding="utf-8")
    )
    records = [json.loads(line) for line in V2.read_text(encoding="utf-8").splitlines()]
    training_ids = set(metadata["training_example_ids"])
    non_training_ids = {record["id"] for record in records if record["split"] != "train"}

    assert metadata["training_split"] == "train"
    assert len(training_ids) == 160
    assert training_ids.isdisjoint(non_training_ids)


def test_v2_loader_validates_each_split():
    assert len(load_dataset(V2, split="train").examples) == 160
    assert len(load_dataset(V2, split="dev").examples) == 50
    assert len(load_dataset(V2, split="test").examples) == 18


def test_v3_preserves_sealed_test_and_group_integrity():
    v2_test = {e.example_id for e in load_dataset(V2, split="test").examples}
    v3_test = {e.example_id for e in load_dataset(V3, split="test").examples}
    manifest = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))

    assert v3_test == v2_test
    assert len(load_dataset(V3, split="train").examples) == 216
    assert len(load_dataset(V3, split="dev").examples) == 78
    assert all(group["split"] in {"train", "dev", "test"} for group in manifest["groups"].values())


def test_v3_artifact_excludes_dev_and_test_ids():
    metadata = json.loads(V3_META.read_text(encoding="utf-8"))
    train_ids = {e.example_id for e in load_dataset(V3, split="train").examples}
    held_out = {
        e.example_id
        for split in ("dev", "test")
        for e in load_dataset(V3, split=split).examples
    }

    assert set(metadata["training_example_ids"]) == train_ids
    assert set(metadata["training_example_ids"]).isdisjoint(held_out)
    assert metadata["training_configuration"]["seed"] == 42
    assert metadata["capability_encoding"] == "independent"


def test_v3_adjudication_traceability_and_review_provenance():
    rows = [json.loads(line) for line in V3.read_text(encoding="utf-8").splitlines() if line]
    manifest = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))

    assert len(manifest["label_changes"]) == 142
    assert all(row["review_status"] == "assistant_reviewed_pending_human_review" for row in rows)
    assert all(row["annotation_status"] == "synthetic_candidate" for row in rows)


def test_adapter_interface_and_exactly_one_encoder_inference(monkeypatch):
    from assistant.orchestration.router import QwenClassifier

    monkeypatch.setattr(
        QwenClassifier,
        "classify",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Qwen called")),
    )
    adapter = _adapter()

    result = adapter.predict(_request())

    assert isinstance(adapter, DecisionModelAdapter)
    assert adapter.encoder.forward_calls == 1
    assert adapter.inference_count == 1
    assert result["execution_path"] == "single_action"


def test_capability_selection_is_restricted_to_advertised_set():
    adapter = _adapter()
    result = adapter.predict(_request())

    assert result["selected_capability"] == "open_app"
    assert result["selected_capability"] in {"open_app", "google_search"}


def test_no_advertised_capability_escalates_safely():
    result = _adapter().predict(_request(capabilities=False))

    assert result["execution_path"] == "escalate"
    assert result["selected_capability"] is None
    assert result["needs_escalation"] is True


def test_clarification_has_missing_information_and_no_capability():
    result = _adapter(clarification=0.9).predict(_request())

    assert result["execution_path"] == "clarify"
    assert result["needs_clarification"] is True
    assert result["selected_capability"] is None
    assert result["missing_information"][0]["name"] == "target"


def test_contradictory_uncertainty_is_resolved_deterministically():
    result = _adapter(clarification=0.7, escalation=0.9).predict(_request())

    assert result["execution_path"] == "escalate"
    assert result["needs_escalation"] is True
    assert result["needs_clarification"] is False


def test_unresolved_reference_is_represented_without_resolution():
    result = _adapter(
        path="clarify", clarification=0.9, reference_kind="item", reference_resolved=0.1
    ).predict(_request(context={}))

    assert result["context_references"][0]["kind"] == "item"
    assert result["context_references"][0]["resolved_value"] is None


def test_malformed_artifact_triggers_fastclassifier_fallback():
    adapter = _adapter()
    del adapter.artifact["execution_path"]
    classifier = FastClassifier(adapter)

    decision = classifier.decide("Anything")

    assert decision.intent == "classification_failed"
    assert decision.needs_escalation is True


def test_confidence_is_bounded_even_if_head_probability_is_invalidly_large():
    adapter = _adapter()
    adapter.artifact["execution_path"] = StaticModel(
        "single_action", ["single_action", "direct_response"], [2.0, -1.0]
    )

    result = adapter.predict(_request())

    assert result["confidence"] == 1.0


def test_offline_inference_does_not_use_network_or_execute_tools(monkeypatch):
    import socket
    from assistant.tools.executor import ToolExecutor

    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    monkeypatch.setattr(ToolExecutor, "execute", lambda *a, **k: (_ for _ in ()).throw(AssertionError("tool")))

    result = _adapter().predict(_request())

    assert result["selected_capability"] == "open_app"
