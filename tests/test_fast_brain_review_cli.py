import json

import pytest

from benchmarks.fast_classifier.review_fast_brain import (
    AIReviewProposal, ReviewStore, OllamaReviewProposer, run_ai_cli, run_cli,
    source_conflict_evidence, validate_ai_proposal,
)
from benchmarks.fast_classifier.review_fast_brain import _compatibility_proposal
from benchmarks.fast_classifier.fast_brain_v1.review import ReviewDecision, readiness
from benchmarks.fast_classifier.fast_brain_v1.schema import DatasetValidationError


def _queue(tmp_path):
    record = {"id": "r1", "text": "hello", "intent_family": "conversation", "capability_family": "none",
              "clarification_required": False, "missing_information": [], "provenance": "historical_synthetic",
              "group_id": "g1", "split": "train", "review_status": "pending_review"}
    path = tmp_path / "review_queue.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def test_first_record_accept_and_source_unchanged(tmp_path):
    queue = _queue(tmp_path)
    source = queue.read_bytes()
    store = ReviewStore(queue, tmp_path / "decisions.jsonl")
    answers = iter(["a"])
    run_cli(store, "human", lambda _: next(answers))
    assert queue.read_bytes() == source
    assert len(store.decisions) == 1
    assert readiness(store.records, store.decisions) == "TRAINING_READY"


def test_correct_and_malformed_label_rejected(tmp_path):
    store = ReviewStore(_queue(tmp_path), tmp_path / "decisions.jsonl")
    answers = iter(["c", "{}", "c", json.dumps({
        "intent_family": "conversation", "execution_path": "respond", "capability_family": "none",
        "context_requirements": ["none"], "clarification": {"required": False, "missing": []},
        "uncertainty": "medium", "escalation": {"candidate": False, "reason": None}})])
    run_cli(store, "human", lambda _: next(answers))
    assert len(store.decisions) == 1
    assert set(store.decisions[0]["reviewed_label"]) == {"intent_family", "execution_path", "capability_family", "context_requirements", "clarification", "uncertainty", "escalation"}


def test_unknown_duplicate_and_resume(tmp_path):
    queue = _queue(tmp_path)
    path = tmp_path / "decisions.jsonl"
    store = ReviewStore(queue, path)
    with pytest.raises(DatasetValidationError):
        store.append(ReviewDecision("unknown", "human", "skip"))
    store.append(ReviewDecision("r1", "human", "skip", reviewed_at="2026-01-01T00:00:00+00:00"))
    with pytest.raises(DatasetValidationError):
        store.append(ReviewDecision("r1", "human", "skip"))


def test_decision_validation_requires_reviewer_and_complete_label():
    with pytest.raises(DatasetValidationError):
        ReviewDecision("r1", "", "accept").validate()
    with pytest.raises(DatasetValidationError):
        ReviewDecision("r1", "human", "correct", reviewed_label={"intent_family": "conversation"}).validate()


def test_correction_retries_single_line_json_without_returning_to_menu(tmp_path):
    queue = _queue(tmp_path)
    source = queue.read_bytes()
    decisions = tmp_path / "decisions.jsonl"
    store = ReviewStore(queue, decisions)
    valid = {"intent_family": "conversation", "execution_path": "respond", "capability_family": "none",
             "context_requirements": ["none"], "clarification": {"required": False, "missing": []},
             "uncertainty": "medium", "escalation": {"candidate": False, "reason": None}}
    answers = iter(["c", "{", '"intent_family": "conversation"', json.dumps({**valid, "intent_family": "bad"}), json.dumps(valid), "q"])
    output = __import__("io").StringIO()
    run_cli(store, "human", lambda _: next(answers), output)
    assert "Invalid choice." not in output.getvalue()
    assert len(store.decisions) == 1
    assert json.loads(decisions.read_text(encoding="utf-8").splitlines()[0])["reviewer_decision"] == "correct"
    assert queue.read_bytes() == source


def test_accepted_review_is_persisted_and_resume_skips_it_in_isolated_artifact(tmp_path):
    queue = _queue(tmp_path)
    decisions = tmp_path / "review_decisions.jsonl"
    store = ReviewStore(queue, decisions)
    answers = iter(["a"])
    run_cli(store, "human", lambda _: next(answers))

    resumed = ReviewStore(queue, decisions)
    assert len(resumed.decisions) == 1
    assert resumed.decisions[0]["record_id"] == "r1"
    assert resumed.decisions[0]["reviewer_decision"] == "accept"
    assert resumed.pending == []


def _proposal(mode="AUTO_CONFIDENT"):
    label = {"intent_family": "conversation", "execution_path": "respond", "capability_family": "none",
             "context_requirements": ["none"], "clarification": {"required": False, "missing": []},
             "uncertainty": "low", "escalation": {"candidate": False, "reason": None}}
    return AIReviewProposal(mode, label if mode == "AUTO_CONFIDENT" else None,
                            "high" if mode == "AUTO_CONFIDENT" else "low", "Evidence supports the proposal.",
                            ("Candidate text and canonical label agree.",), mode != "AUTO_CONFIDENT")


class _FakeProposer:
    def __init__(self, proposal):
        self.proposal_value = proposal

    def propose(self, record, canonical):
        return self.proposal_value


def test_ai_proposal_schema_rejects_invalid_output():
    with pytest.raises(DatasetValidationError):
        validate_ai_proposal({"mode": "AUTO_CONFIDENT", "confidence": "high", "rationale": "x"})


def test_ai_proposal_does_not_persist_until_human_acceptance(tmp_path):
    queue = _queue(tmp_path)
    decisions = tmp_path / "decisions.jsonl"
    store = ReviewStore(queue, decisions)
    answers = iter(["q"])
    run_ai_cli(store, "human", _FakeProposer(_proposal()), lambda _: next(answers))
    assert store.decisions == []
    assert not decisions.exists()


def test_ai_acceptance_persists_exactly_one_decision(tmp_path):
    store = ReviewStore(_queue(tmp_path), tmp_path / "decisions.jsonl")
    answers = iter(["a"])
    run_ai_cli(store, "human", _FakeProposer(_proposal()), lambda _: next(answers))
    assert len(store.decisions) == 1
    assert store.decisions[0]["reviewer_decision"] == "accept"


def test_ai_correction_can_be_overridden_by_human_label(tmp_path):
    store = ReviewStore(_queue(tmp_path), tmp_path / "decisions.jsonl")
    label = {"intent_family": "conversation", "execution_path": "respond", "capability_family": "none",
             "context_requirements": ["none"], "clarification": {"required": False, "missing": []},
             "uncertainty": "medium", "escalation": {"candidate": False, "reason": None}}
    answers = iter(["c", json.dumps(label)])
    run_ai_cli(store, "human", _FakeProposer(_proposal()), lambda _: next(answers))
    assert store.decisions[0]["reviewer_decision"] == "correct"
    assert store.decisions[0]["reviewed_label"] == label


def test_ai_source_conflict_requires_human_attention(tmp_path):
    store = ReviewStore(_queue(tmp_path), tmp_path / "decisions.jsonl")
    answers = iter(["a", "q"])
    run_ai_cli(store, "human", _FakeProposer(_proposal("SOURCE_CONFLICT")), lambda _: next(answers))
    assert store.decisions == []


def test_ai_timeout_falls_back_to_human_review(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise TimeoutError("timed out")
    monkeypatch.setattr("urllib.request.urlopen", fail)
    store = ReviewStore(_queue(tmp_path), tmp_path / "decisions.jsonl")
    proposal = OllamaReviewProposer(timeout=0.01).propose(next(iter(store.pending)), {})
    assert proposal.mode == "HUMAN_REVIEW"
    assert proposal.proposed_label is None


def test_legacy_direct_response_is_compatible_with_respond(tmp_path):
    record = next(r for r in ReviewStore(_queue(tmp_path), tmp_path / "d").records if r.id == "r1")
    record.original_label = '{"execution_path":"direct_response","intent":"respond"}'
    canonical = {"execution_path": "respond", "intent_family": "conversation"}
    assert source_conflict_evidence(record, canonical)[0] is False


def test_missing_provenance_is_not_a_source_conflict(tmp_path):
    record = next(r for r in ReviewStore(_queue(tmp_path), tmp_path / "d").records if r.id == "r1")
    record.original_label = None
    assert source_conflict_evidence(record, {"execution_path": "respond"})[0] is False


def test_kubernetes_source_semantics_support_qwen_agent(tmp_path):
    record = next(r for r in ReviewStore(_queue(tmp_path), tmp_path / "d").records if r.id == "r1")
    record.text = "Kubernetes overview se main ideas nikalo"
    record.original_label = '{"execution_path":"multi_step_task","intent":"research","selected_capability":"google_search"}'
    canonical = {"execution_path": "qwen_agent", "intent_family": "multi_step_task", "capability_family": "browser"}
    assert source_conflict_evidence(record, canonical)[0] is False


def test_genuine_source_intent_contradiction_is_conflict(tmp_path):
    record = next(r for r in ReviewStore(_queue(tmp_path), tmp_path / "d").records if r.id == "r1")
    record.original_label = '{"intent":"research","execution_path":"research"}'
    assert source_conflict_evidence(record, {"execution_path": "respond", "intent_family": "conversation"})[0] is True


def test_real_unsupported_tamil_records_are_semantically_compatible_not_aliases():
    store = ReviewStore()
    for record_id in ("hist-184946cff44a2ae0", "hist-14b4023aa37a6d05"):
        record = next(record for record in store.records if record.id == record_id)
        conflict, evidence = source_conflict_evidence(record, store.canonical_label(record))
        assert conflict is False
        assert "unsupported/escalation" in evidence


def test_real_rainbow_record_uses_explicit_legacy_mapping():
    store = ReviewStore()
    record = next(record for record in store.records if record.id == "hist-106e0dba6c2df3db")
    conflict, evidence = source_conflict_evidence(record, store.canonical_label(record))
    assert conflict is False
    assert "legacy/current" in evidence


def test_critical_unsupported_record_keeps_human_review_after_nonconflicting_source():
    store = ReviewStore()
    record = next(record for record in store.records if record.id == "hist-184946cff44a2ae0")
    canonical = store.canonical_label(record)
    conflict, evidence = source_conflict_evidence(record, canonical)
    assert conflict is False
    assert evidence == "source unsupported/escalation semantics support canonical unsupported label"
    # Provenance agreement is not an AI label proposal and must not bypass
    # critical human review for unsupported requests.
    assert _compatibility_proposal(record, canonical) is None


def test_audited_semantically_supported_records_can_return_complete_ai_proposals(monkeypatch):
    label = {"intent_family": "research", "execution_path": "research", "capability_family": "research",
             "context_requirements": ["active_task"], "clarification": {"required": False, "missing": []},
             "uncertainty": "medium", "escalation": {"candidate": False, "reason": None}}
    body = json.dumps({"response": json.dumps({
        "mode": "AUTO_CONFIDENT", "proposed_label": label, "confidence": "high",
        "rationale": "Semantic meaning and source metadata agree.",
        "source_evidence": ["Original source intent and capability support the canonical semantics."],
        "human_attention": False})}).encode()

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return body

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    store = ReviewStore()
    texts = ("పుణెలో", "AI stories", "spending spreadsheet", "browser security")
    for text in texts:
        record = next(record for record in store.records if text in record.text)
        proposal = OllamaReviewProposer().propose(record, store.canonical_label(record))
        assert proposal.mode == "AUTO_CONFIDENT"
        assert proposal.proposed_label is not None


def test_prompt_requires_semantic_proposal_before_conflict_classification():
    from benchmarks.fast_classifier.review_fast_brain import _ai_prompt
    prompt = _ai_prompt(next(iter(ReviewStore().records)), {"intent_family": "conversation"})
    assert "First perform semantic label review" in prompt
    assert "COMPLETE proposed_label" in prompt
    assert "Never return SOURCE_CONFLICT merely because" in prompt
