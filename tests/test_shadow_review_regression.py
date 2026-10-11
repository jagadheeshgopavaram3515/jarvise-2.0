"""Offline end-to-end regressions using captured Qwen exchanges, never live inference."""
import copy
import importlib
import io
import json
from pathlib import Path
import sys
import types
import urllib.request

import pytest


FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "shadow_review_regression.json").read_text(encoding="utf-8"))


@pytest.fixture
def reviewer(monkeypatch):
    # Import the real shadow modules without the eager production package initializer.
    # Restore the namespace afterwards so this test does not alter other test imports.
    with monkeypatch.context() as isolated:
        if "assistant.orchestration" not in sys.modules:
            package = types.ModuleType("assistant.orchestration")
            package.__path__ = [str(Path(__file__).parents[1] / "assistant" / "orchestration")]
            isolated.setitem(sys.modules, "assistant.orchestration", package)
        module = importlib.import_module("benchmarks.fast_classifier.review_fast_brain")
    return module


def make_store(reviewer, tmp_path, code="A"):
    candidate = copy.deepcopy(FIXTURES[code]["candidate"])
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    return reviewer.ReviewStore(queue, tmp_path / "decisions.jsonl")


def mock_http(monkeypatch, proposal=None, body=None, expected=None):
    captured = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self):
            return body if body is not None else json.dumps({"response": json.dumps(proposal)}).encode()

    def send(request, timeout):
        payload = json.loads(request.data)
        captured.append(payload)
        assert request.full_url == "http://localhost:11434/api/generate"
        assert request.get_method() == "POST"
        assert request.headers == {"Content-type": "application/json", "User-agent": "JARVIS-review"}
        assert timeout == 20.0
        if expected is not None:
            assert payload == expected
            assert request.data == json.dumps(expected).encode("utf-8")
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", send)
    return captured


def propose(reviewer, store):
    record = store.pending[0]
    result = reviewer.OllamaReviewProposer().propose(record, store.canonical_label(record))
    result.validate()
    return result


@pytest.mark.parametrize("code", list("ABCD"))
def test_captured_requests_and_final_semantic_outcomes(reviewer, monkeypatch, tmp_path, code):
    store = make_store(reviewer, tmp_path, code)
    fixture = FIXTURES[code]
    captured = mock_http(monkeypatch, fixture["model_proposal"], expected=fixture["request"]["payload"])
    result = propose(reviewer, store)
    assert len(captured) == 1
    if code == "D":
        assert result.mode == "HUMAN_REVIEW"
        assert result.proposed_label is None and result.human_attention
        assert result.source_evidence == ("Unresolved missing information: target",)
    else:
        assert result.to_dict() == fixture["model_proposal"]
        assert result.mode == "AUTO_CONFIDENT"
        assert result.proposed_label == fixture["candidate"]["canonical_label"]
        from assistant.orchestration.fast_brain.contract import FastBrainDecision
        FastBrainDecision.from_mapping(result.proposed_label)
    assert store.decisions == [] and not store.decisions_path.exists()


def test_reported_failure_is_not_silently_promoted_or_persisted(reviewer, monkeypatch, tmp_path):
    store = make_store(reviewer, tmp_path)
    mock_http(monkeypatch, FIXTURES["reported_failure"], expected=FIXTURES["A"]["request"]["payload"])
    assert propose(reviewer, store).to_dict() == FIXTURES["reported_failure"]
    answers = iter(["a", "q"])
    output = io.StringIO()
    reviewer.run_ai_cli(store, "test-human", reviewer.OllamaReviewProposer(), lambda _: next(answers), output)
    assert "not safe to accept automatically" in output.getvalue()
    assert store.decisions == [] and not store.decisions_path.exists()


def test_supported_correction_of_wrong_candidate(reviewer, monkeypatch, tmp_path):
    store = make_store(reviewer, tmp_path, "B")
    record = store.pending[0]
    correct = copy.deepcopy(store.canonical_label(record))
    wrong = copy.deepcopy(FIXTURES["C"]["candidate"]["canonical_label"])
    store._canonical_labels[record.id] = wrong
    response = copy.deepcopy(FIXTURES["B"]["model_proposal"])
    response.update(mode="AI_CORRECTION", proposed_label=correct)
    mock_http(monkeypatch, response)
    assert propose(reviewer, store).to_dict() == response
    assert store.decisions == []
    reviewer.run_ai_cli(store, "test-human", reviewer.OllamaReviewProposer(), lambda _: "a", io.StringIO())
    assert store.decisions[0]["reviewed_label"] == correct


def test_corrected_clear_request_can_resolve_wrong_clarification_candidate(reviewer, monkeypatch, tmp_path):
    store = make_store(reviewer, tmp_path, "C")
    record = store.pending[0]
    correct = copy.deepcopy(store.canonical_label(record))
    store._canonical_labels[record.id] = copy.deepcopy(FIXTURES["D"]["candidate"]["canonical_label"])
    response = copy.deepcopy(FIXTURES["C"]["model_proposal"])
    response.update(mode="AI_CORRECTION", proposed_label=correct)
    mock_http(monkeypatch, response)
    assert propose(reviewer, store).to_dict() == response


def test_genuine_model_abstention_is_preserved(reviewer, monkeypatch, tmp_path):
    store = make_store(reviewer, tmp_path, "D")
    response = {"mode": "HUMAN_REVIEW", "proposed_label": None, "confidence": "medium",
                "rationale": "No target referent is available.", "source_evidence": ["Open it."], "human_attention": True}
    mock_http(monkeypatch, response)
    assert propose(reviewer, store).to_dict() == response


@pytest.mark.parametrize("mode", ["AUTO_CONFIDENT", "AI_CORRECTION"])
def test_unresolved_proposed_label_requires_human_even_when_candidate_was_clear(reviewer, monkeypatch, tmp_path, mode):
    store = make_store(reviewer, tmp_path, "C")
    response = copy.deepcopy(FIXTURES["D"]["model_proposal"])
    response["mode"] = mode
    mock_http(monkeypatch, response)
    result = propose(reviewer, store)
    assert result.mode == "HUMAN_REVIEW" and result.proposed_label is None
    assert result.source_evidence == ("Unresolved missing information: target",)


def test_real_source_conflict_is_retained(reviewer, monkeypatch, tmp_path):
    store = make_store(reviewer, tmp_path)
    record = store.pending[0]
    store._canonical_labels[record.id] = copy.deepcopy(FIXTURES["C"]["candidate"]["canonical_label"])
    response = {"mode": "SOURCE_CONFLICT", "proposed_label": None, "confidence": "medium",
                "rationale": "Original research intent contradicts the local-task candidate.",
                "source_evidence": ["Original intent=research; candidate intent_family=local_task."], "human_attention": True}
    mock_http(monkeypatch, response)
    assert propose(reviewer, store).to_dict() == response


@pytest.mark.parametrize("body", [b"bad json", b'{"response":"truncated"}', b'{"response":"{}"}'])
def test_invalid_response_falls_back_without_acceptance(reviewer, monkeypatch, tmp_path, body):
    store = make_store(reviewer, tmp_path)
    mock_http(monkeypatch, body=body)
    result = propose(reviewer, store)
    assert result.mode == "HUMAN_REVIEW" and result.confidence == "low"
    assert result.source_evidence == ("No reliable AI output was available.",)
    assert result.to_dict() != FIXTURES["reported_failure"]


@pytest.mark.parametrize("error", [TimeoutError("offline timeout"), OSError("offline connection failure")])
def test_transport_failure_falls_back(reviewer, monkeypatch, tmp_path, error):
    store = make_store(reviewer, tmp_path)
    def fail(*args, **kwargs): raise error
    monkeypatch.setattr(urllib.request, "urlopen", fail)
    result = propose(reviewer, store)
    assert result.mode == "HUMAN_REVIEW" and result.confidence == "low"
    assert result.source_evidence == ("No reliable AI output was available.",)


def test_cross_field_validation_is_not_weakened(reviewer, monkeypatch, tmp_path):
    store = make_store(reviewer, tmp_path)
    response = copy.deepcopy(FIXTURES["A"]["model_proposal"])
    response["proposed_label"]["clarification"] = {"required": True, "missing": []}
    mock_http(monkeypatch, response)
    result = propose(reviewer, store)
    assert result.mode == "HUMAN_REVIEW" and result.rationale.startswith("invalid contract:")


def test_human_acceptance_preserves_existing_history_and_queue(reviewer, monkeypatch, tmp_path):
    store = make_store(reviewer, tmp_path)
    current = json.loads(store.queue_path.read_text())
    previous = copy.deepcopy(current)
    previous["record_id"] = "previous-reviewed-record"
    store.queue_path.write_text(json.dumps(previous) + "\n" + json.dumps(current) + "\n", encoding="utf-8")
    store = reviewer.ReviewStore(store.queue_path, store.decisions_path)
    store.append(reviewer.ReviewDecision("previous-reviewed-record", "earlier-human", "skip"))
    history = store.decisions_path.read_bytes()
    queue = store.queue_path.read_bytes()
    mock_http(monkeypatch, FIXTURES["A"]["model_proposal"])
    reviewer.run_ai_cli(store, "test-human", reviewer.OllamaReviewProposer(), lambda _: "q", io.StringIO())
    assert store.decisions_path.read_bytes() == history
    reviewer.run_ai_cli(store, "test-human", reviewer.OllamaReviewProposer(), lambda _: "a", io.StringIO())
    assert store.decisions_path.read_bytes().startswith(history)
    assert len(store.decisions) == 2 and store.decisions[-1]["reviewer_id"] == "test-human"
    assert store.queue_path.read_bytes() == queue
