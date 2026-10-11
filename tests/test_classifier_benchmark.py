"""Deterministic tests for the offline classifier benchmark, not model accuracy."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from assistant.orchestration.decision import ClassifierDecision
from benchmarks.fast_classifier.evaluate import (
    DatasetValidationError,
    evaluate_legacy_classifier,
    evaluate_structured_adapter,
    latency_statistics,
    load_dataset,
    load_invalid_output_cases,
    render_summary,
)


ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "benchmarks" / "fast_classifier" / "v1" / "seed.jsonl"
INVALID_OUTPUTS = (
    ROOT / "benchmarks" / "fast_classifier" / "v1" / "invalid_outputs.jsonl"
)
ROUTER = ROOT / "assistant" / "orchestration" / "router.py"


class ExpectedAdapter:
    adapter_id = "mock-expected-structured"

    def __init__(self, examples):
        self.examples = {example.utterance: example for example in examples}
        self.calls = 0

    def predict(self, request):
        self.calls += 1
        expected = self.examples[request.user_text].expected
        return {
            "intent": expected.intent,
            "execution_path": expected.execution_path,
            "request_modes": [mode.value for mode in expected.request_modes],
            "selected_capability": expected.selected_capability,
            "context_references": [
                {
                    "reference": item.reference,
                    "kind": item.kind,
                    "resolved_value": item.resolved_value,
                }
                for item in expected.context_references
            ],
            "missing_information": [
                {"name": item.name, "description": item.description}
                for item in expected.missing_information
            ],
            "confidence": 0.9,
            "needs_clarification": expected.needs_clarification,
            "needs_escalation": expected.needs_escalation,
            "evidence": [{"kind": "test_fixture", "reference": "expected-label"}],
        }


class LegacyByExpectedPath:
    def __init__(self, examples):
        self.paths = {example.utterance: example.expected.execution_path for example in examples}

    def classify(self, text):
        path = self.paths[text]
        if path not in {"LOCAL", "GEMINI", "TASK"}:
            path = "GEMINI"
        return path, 1.25


def _write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_seed_dataset_loads_and_has_versioned_reviewed_coverage():
    dataset = load_dataset(SEED)

    assert dataset.schema_version == "1.0"
    assert len(dataset.examples) == 28
    assert {example.split for example in dataset.examples} == {"dev", "test"}
    assert {"en", "te", "hi", "ta"}.issubset(
        {example.language.split("-")[0] for example in dataset.examples}
    )
    assert all(example.annotation_status == "reviewed" for example in dataset.examples)


def test_dataset_selection_by_split_ids_and_limit():
    dataset = load_dataset(
        SEED,
        split="test",
        example_ids={"en-knowledge-001", "ta-romanized-action-001"},
        limit=1,
    )

    assert len(dataset.examples) == 1
    assert dataset.examples[0].example_id == "en-knowledge-001"


@pytest.mark.parametrize(
    "mutation,error",
    [
        ({"schema_version": "9.0"}, "schema_version"),
        ({"expected": {"request_modes": ["unknown"]}}, "request mode"),
        ({"context": []}, "context"),
    ],
)
def test_dataset_schema_validation_rejects_bad_records(tmp_path, mutation, error):
    raw = json.loads(SEED.read_text(encoding="utf-8").splitlines()[0])
    if "expected" in mutation:
        raw["expected"].update(mutation["expected"])
    else:
        raw.update(mutation)
    path = tmp_path / "bad.jsonl"
    _write_jsonl(path, [raw])

    with pytest.raises(DatasetValidationError, match=error):
        load_dataset(path)


def test_invalid_output_corpus_is_versioned_and_rejected_by_contract():
    cases = load_invalid_output_cases(INVALID_OUTPUTS)

    assert len(cases) == 6
    for case in cases:
        with pytest.raises(Exception) as caught:
            ClassifierDecision.from_mapping(case.output)
        assert case.expected_error in str(caught.value)


def test_legacy_evaluation_scores_only_real_legacy_fields():
    dataset = load_dataset(SEED, example_ids={"en-conversation-001", "en-desktop-001"})
    report = evaluate_legacy_classifier(
        LegacyByExpectedPath(dataset.examples),
        dataset,
        measurement_kind="mock_test",
    )

    assert report["metrics"]["execution_path"]["accuracy"] == 1.0
    assert report["metrics"]["intent"]["available"] is False
    assert report["counts"]["successful_predictions"] == 2
    assert report["adapter"]["measurement_kind"] == "mock_test"


def test_legacy_unrepresentable_paths_are_not_counted_as_errors():
    dataset = load_dataset(SEED, example_ids={"ctx-continue-unresolved-001"})
    report = evaluate_legacy_classifier(
        LegacyByExpectedPath(dataset.examples), dataset, measurement_kind="mock_test"
    )

    assert report["metrics"]["execution_path"]["applicable"] == 0
    assert report["metrics"]["execution_path"]["accuracy"] is None
    assert report["counts"]["not_applicable_execution_paths"] == 1


def test_unknown_legacy_route_is_an_invalid_output():
    dataset = load_dataset(SEED, limit=1)

    class UnknownRoute:
        def classify(self, text):
            return "MOON", 0.1

    report = evaluate_legacy_classifier(
        UnknownRoute(), dataset, measurement_kind="mock_test"
    )

    assert report["counts"]["invalid_outputs"] == 1
    assert report["metrics"]["execution_path"]["applicable"] == 0


def test_structured_adapter_scores_each_contract_field():
    dataset = load_dataset(SEED, limit=3)
    adapter = ExpectedAdapter(dataset.examples)

    report = evaluate_structured_adapter(
        adapter, dataset, measurement_kind="mock_test"
    )

    for field in (
        "intent",
        "execution_path",
        "request_modes",
        "selected_capability",
        "context_references",
        "missing_information",
        "needs_clarification",
        "needs_escalation",
    ):
        assert report["metrics"][field]["accuracy"] == 1.0
    assert report["metrics"]["schema_validity"]["accuracy"] == 1.0
    assert report["metrics"]["fallback_trigger_rate"]["rate"] == 0.0
    assert report["metrics"]["clarification_binary"]["macro_f1"] == 0.5
    assert report["metrics"]["escalation_binary"]["macro_f1"] == 0.5


def test_malformed_structured_output_is_not_silently_scored_correct():
    dataset = load_dataset(SEED, limit=1)

    class MalformedAdapter:
        adapter_id = "mock-malformed"

        def predict(self, request):
            return {"intent": "missing-most-fields"}

    report = evaluate_structured_adapter(
        MalformedAdapter(), dataset, measurement_kind="mock_test"
    )

    assert report["counts"]["invalid_outputs"] == 1
    assert report["metrics"]["schema_validity"]["accuracy"] == 0.0
    assert report["metrics"]["fallback_trigger_rate"]["rate"] == 1.0


def test_exceptions_and_timeouts_are_recorded_separately():
    dataset = load_dataset(SEED, limit=2)

    class FailureAdapter:
        adapter_id = "mock-failures"

        def __init__(self):
            self.calls = 0

        def predict(self, request):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("model failed")
            time.sleep(0.03)
            return {}

    report = evaluate_structured_adapter(
        FailureAdapter(), dataset, timeout_s=0.005, measurement_kind="mock_test"
    )

    assert report["counts"]["errors"] == 1
    assert report["counts"]["timeouts"] == 1
    assert report["counts"]["successful_predictions"] == 0


def test_latency_statistics_use_nearest_rank_p50_and_p95():
    stats = latency_statistics([40.0, 10.0, 30.0, 20.0])

    assert stats == {"count": 4, "mean_ms": 25.0, "p50_ms": 20.0, "p95_ms": 40.0}


def test_evaluation_never_executes_capabilities_or_changes_live_router(monkeypatch):
    from assistant.tools.executor import ToolExecutor

    before = ROUTER.read_bytes()
    dataset = load_dataset(SEED, example_ids={"en-desktop-001"})
    adapter = ExpectedAdapter(dataset.examples)

    def forbidden_execute(*args, **kwargs):
        raise AssertionError("benchmark attempted to execute a tool")

    monkeypatch.setattr(ToolExecutor, "execute", forbidden_execute)
    report = evaluate_structured_adapter(
        adapter, dataset, measurement_kind="mock_test"
    )

    assert report["counts"]["successful_predictions"] == 1
    assert adapter.calls == 1
    assert ROUTER.read_bytes() == before


def test_terminal_summary_identifies_mock_measurement():
    dataset = load_dataset(SEED, limit=1)
    report = evaluate_structured_adapter(
        ExpectedAdapter(dataset.examples), dataset, measurement_kind="mock_test"
    )

    summary = render_summary(report)

    assert "mock_test" in summary
    assert "Examples=1" in summary
