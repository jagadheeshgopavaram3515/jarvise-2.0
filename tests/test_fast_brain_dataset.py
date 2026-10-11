import json

import pytest

from benchmarks.fast_classifier.fast_brain_v1.dataset import load_jsonl
from benchmarks.fast_classifier.fast_brain_v1.schema import DatasetValidationError, FastBrainRecord
from benchmarks.fast_classifier.fast_brain_v1.split import grouped_split, validate_group_leakage
from benchmarks.fast_classifier.evaluate_fast_brain import evaluate


def record(**overrides):
    value = {"id": "x", "text": "Open Chrome", "intent_family": "deterministic_action",
             "capability_family": "application", "clarification_required": False,
             "missing_information": [], "provenance": "human_reviewed", "group_id": "g",
             "split": "train", "review_status": "reviewed", "language_group": "english",
             "script_group": "latin", "noise_type": "clean"}
    value.update(overrides)
    return FastBrainRecord.from_mapping(value)


def test_schema_uses_frozen_labels_and_provenance():
    assert record().capability_family == "application"
    with pytest.raises(DatasetValidationError):
        record(intent_family="not-an-intent")
    with pytest.raises(DatasetValidationError):
        record(provenance="guessed")


def test_clarification_invariants_and_metadata():
    with pytest.raises(DatasetValidationError):
        record(clarification_required=True, missing_information=[])
    assert record(language_group="romanized_telugu", script_group="latin", noise_type="stt_variant")


def test_loader_rejects_duplicate_ids_and_cross_group_text(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(record().__dict__) + "\n" + json.dumps(record().__dict__) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="duplicate id"):
        load_jsonl(path)
    path.write_text(json.dumps(record(id="a", group_id="a").__dict__) + "\n" + json.dumps(record(id="b", group_id="b").__dict__) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="duplicate text"):
        load_jsonl(path)


def test_grouped_split_is_deterministic_and_leak_free():
    records = [record(id=f"{i}", group_id=f"g{i // 2}") for i in range(8)]
    first = grouped_split(records, seed=7)
    second = grouped_split(records, seed=7)
    assert [x.split for x in first] == [x.split for x in second]


def test_duplicate_text_across_groups_is_rejected_to_prevent_leakage(tmp_path):
    from benchmarks.fast_classifier.fast_brain_v1.dataset import load_jsonl
    import json
    first = {"id": "a", "text": "same", "intent_family": "conversation", "capability_family": "none",
             "clarification_required": False, "missing_information": [], "provenance": "historical_synthetic",
             "group_id": "g1", "split": "train", "review_status": "pending_review"}
    second = {**first, "id": "b", "group_id": "g2", "split": "development"}
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="duplicate text crosses groups"):
        load_jsonl(path)


def test_group_leakage_is_rejected():
    with pytest.raises(DatasetValidationError, match="group crosses"):
        validate_group_leakage([record(id="a", split="train"), record(id="b", split="sealed_candidate")])


def test_machine_readable_evaluation_reports_metrics_without_model():
    value = record()
    result = evaluate([value], {"x": {"intent_family": "deterministic_action",
                                      "capability_family": "application",
                                      "clarification_required": False}})
    assert result["metrics"]["intent_accuracy"] == 1.0
    assert result["metrics"]["clarification"]["f1"] == 0.0

