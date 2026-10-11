import json

import pytest

from benchmarks.fast_classifier.fast_brain_v1.review import ReviewDecision, experiment_readiness_report, readiness, review_artifacts_are_protected, review_priority
from benchmarks.fast_classifier.fast_brain_v1.schema import DatasetValidationError, FastBrainRecord


def candidate(**overrides):
    value = {"id": "r", "text": "nuvvu repu office ki vastava?", "intent_family": "information",
             "capability_family": "none", "clarification_required": False, "missing_information": [],
             "provenance": "historical_synthetic", "group_id": "g", "split": "train",
             "review_status": "pending_review", "language_group": "telugu", "script_group": "latin",
             "semantic_language": "telugu", "input_script": "romanized_telugu", "response_language": "telugu"}
    value.update(overrides)
    return FastBrainRecord.from_mapping(value)


def test_review_priority_protects_romanized_telugu():
    priority, reasons = review_priority(candidate())
    assert priority == "critical"
    assert "multilingual_metadata" in reasons


def test_review_requires_explicit_reviewer_and_label_for_correction():
    with pytest.raises(DatasetValidationError):
        ReviewDecision(record_id="r", reviewer_id="", decision="accept").validate()
    with pytest.raises(DatasetValidationError):
        ReviewDecision(record_id="r", reviewer_id="human", decision="correct").validate()
    decision = ReviewDecision(record_id="r", reviewer_id="human", decision="accept")
    assert decision.to_dict()["reviewer_id"] == "human"


def test_pending_data_cannot_be_training_ready():
    assert readiness([candidate()], []) == "CANDIDATE"


def test_invalid_or_incomplete_reviews_cannot_promote_readiness():
    record = candidate()
    base = {"record_id": record.id, "reviewer_id": "human", "reviewer_decision": "accept",
            "reviewed_label": None, "adjudication_status": "not_required"}
    assert readiness([record], [base]) == "TRAINING_READY"
    assert readiness([record], [{**base, "reviewer_id": ""}]) == "CANDIDATE"
    assert readiness([record], [{**base, "record_id": "unknown"}]) == "CANDIDATE"
    assert readiness([record], [base, {**base}]) == "CANDIDATE"
    assert readiness([record], [{**base, "reviewer_decision": "skip"}]) == "REVIEWED_NOT_ADJUDICATED"
    assert readiness([record], [{**base, "reviewer_decision": "correct"}]) == "CANDIDATE"
    assert readiness([record], [{**base, "adjudication_status": "requires_adjudication"}]) == "REVIEWED_NOT_ADJUDICATED"


def test_reviewed_label_does_not_replace_original():
    original = candidate()
    review = ReviewDecision(record_id=original.id, reviewer_id="human", decision="correct",
                            reviewed_label={"intent_family": "conversation"})
    assert original.intent_family == "information"
    assert review.reviewed_label["intent_family"] == "conversation"


def test_multilingual_regression_metadata():
    assert candidate().semantic_language == "telugu"
    assert candidate().input_script == "romanized_telugu"
    assert candidate(text="kal tum office ja rahe ho?", language_group="hindi",
                     semantic_language="hindi", input_script="romanized_hindi").input_script == "romanized_hindi"
    assert candidate(text="నా ప్రాజెక్ట్ చూడి", language_group="telugu",
                     semantic_language="telugu", input_script="telugu").input_script == "telugu"


def test_review_artifacts_cannot_be_overwritten_after_human_work():
    queue = json.dumps({"record_id": "r", "reviewer_id": "human", "reviewer_decision": "accept"})
    assert review_artifacts_are_protected(queue, "")[0]
    assert review_artifacts_are_protected("", '{"record_id":"r"}')[0]
    assert not review_artifacts_are_protected('{"reviewer_decision":null}', "")[0]


def test_experiment_readiness_requires_human_language_coverage():
    record = candidate(language_group="english")
    label = {"intent_family": "information", "execution_path": "respond", "capability_family": "none",
             "context_requirements": ["active_task"], "clarification": {"required": False, "missing": []},
             "uncertainty": "medium", "escalation": {"candidate": False, "reason": None}}
    report = experiment_readiness_report([record], [{"record_id": "r", "reviewer_id": "human",
        "reviewer_decision": "accept", "reviewed_label": label, "adjudication_status": "not_required"}])
    assert report["status"] == "BLOCKED_INSUFFICIENT_HUMAN_LABELS"
    assert report["usable_sample_count"] == 1
    assert set(report["missing_languages"]) == {"telugu", "hindi", "tamil", "code_switch"}


def test_experiment_readiness_never_counts_skip_as_eligible():
    record = candidate(language_group="english")
    report = experiment_readiness_report([record], [{"record_id": "r", "reviewer_id": "human",
        "reviewer_decision": "skip", "reviewed_label": None, "adjudication_status": "not_required"}])
    assert report["usable_sample_count"] == 0
    assert report["status"] == "BLOCKED_INSUFFICIENT_HUMAN_LABELS"
