"""Synthetic unit fixtures below are not independently reviewed gold evidence."""
import copy
import pytest

from benchmarks.fast_classifier.gold_review import (
    FIELDS, baseline_modules, evaluate_records, input_hash, score, verified_gold, verify_inputs,
)


def record():
    return {"review_id": "test-only-001", "utterance": "Explain photosynthesis.", "context": {},
            "capabilities": [], "language": {"semantic_language": "english", "input_script": "latin"},
            "review_requirements": {"required_languages": ["english"], "required_scripts": ["latin"],
                                    "transliteration_expertise_required": False},
            "rubric_sha256": "test-only-rubric", "judgments": [], "adjudication": None}


def judgment(r, reviewer="test-human-a"):
    # Explicit synthetic reference fixture; never derive reference from prediction.
    label = {"intent_family": "information", "execution_path": "respond",
             "capability_family": "none", "context_requirements": ["none"],
             "clarification": {"required": False, "missing": []}, "uncertainty": "low",
             "escalation": {"candidate": False, "reason": None}}
    return {"judgment": "labeled", "review_kind": "independent", "reviewer_type": "human",
            "reviewer_id": reviewer, "prediction_blinded": True, "input_sha256": input_hash(r),
            "qualification": {"languages": ["english"], "scripts": ["latin"], "transliteration": False},
            "rubric_sha256": r["rubric_sha256"], "field_rationale": dict.fromkeys(FIELDS, "Synthetic unit-test rationale."),
            "timestamp": "2026-10-10T13:00:00+05:30", "label": label, "notes": "Synthetic test fixture, never real gold."}


def test_no_reference_labels_means_not_measured_and_no_prediction(monkeypatch):
    r = record()
    backend = baseline_modules()["deterministic"].DeterministicFastBrain
    monkeypatch.setattr(backend, "decide", lambda *a: pytest.fail("must not predict before verified gold"))
    predictions, metrics = evaluate_records([r])
    assert predictions == {} and metrics["independently_adjudicated"] == 0
    assert metrics["exact_match"] == {"status": "NOT_MEASURED", "numerator": None, "denominator": 0, "value": None}


@pytest.mark.parametrize("change", [
    {"review_kind": "self_review"}, {"reviewer_type": "AI"}, {"prediction_blinded": False},
    {"judgment": "ambiguous"}, {"judgment": "unresolved"}, {"input_sha256": "changed"},
    {"field_rationale": {}}, {"qualification": {}}, {"timestamp": "2026-10-10"},
])
def test_unverified_reviews_never_become_gold(change):
    r = record(); j = judgment(r); j.update(change); r["judgments"] = [j]
    assert verified_gold(r)[0] is None
    assert score([r], {})["exact_match"]["status"] == "NOT_MEASURED"


def test_qualified_language_and_transliteration_required():
    r = record(); r["review_requirements"]["required_languages"] = ["english", "telugu"]
    r["review_requirements"]["transliteration_expertise_required"] = True
    j = judgment(r); r["judgments"] = [j]
    assert verified_gold(r)[0] is None
    j["qualification"].update(languages=["english", "telugu"], transliteration=True)
    assert verified_gold(r)[1] is None


def test_disagreement_requires_separate_qualified_adjudicator():
    r = record(); a = judgment(r); b = judgment(r, "test-human-b")
    b["label"]["intent_family"] = "conversation"; r["judgments"] = [a, b]
    assert "disagreement" in verified_gold(r)[1]
    r["adjudication"] = copy.deepcopy(a)
    assert verified_gold(r)[0] is None
    r["adjudication"]["reviewer_id"] = "test-human-c"
    assert verified_gold(r)[1] is None


def test_real_offline_baseline_runs_only_on_eligible_unit_fixture():
    r = record(); r["judgments"] = [judgment(r)]
    predictions, metrics = evaluate_records([r])
    assert predictions[r["review_id"]] == r["judgments"][0]["label"]
    assert metrics["exact_match"]["numerator"] == metrics["exact_match"]["denominator"] == 1


def test_prediction_failure_is_included_in_accuracy_denominator():
    r = record(); r["judgments"] = [judgment(r)]
    metrics = score([r], {})
    assert metrics["exact_match"]["numerator"] == 0 and metrics["exact_match"]["denominator"] == 1
    assert metrics["execution_failure"]["numerator"] == 1


def test_every_required_dimension_affects_strict_match():
    r = record(); r["judgments"] = [judgment(r)]
    prediction = copy.deepcopy(r["judgments"][0]["label"])
    prediction["uncertainty"] = "medium"
    metrics = score([r], {r["review_id"]: prediction})
    assert metrics["exact_match"]["numerator"] == 0
    assert metrics["per_field"]["uncertainty"]["numerator"] == 0
    assert metrics["per_field"]["intent_family"]["numerator"] == 1


def test_invalid_enum_and_boolean_are_not_gold():
    r = record(); j = judgment(r); r["judgments"] = [j]
    j["label"]["intent_family"] = "computer_control"
    assert verified_gold(r)[0] is None
    j["label"]["intent_family"] = "information"; j["label"]["clarification"]["required"] = 0
    assert verified_gold(r)[0] is None


def test_frozen_manifest_rejects_input_changes():
    r = record(); manifest = {"review_inputs": {r["review_id"]: input_hash(r)}, "rubric_sha256": r["rubric_sha256"]}
    verify_inputs([r], manifest)
    r["context"] = {"invented_referent": "something"}
    with pytest.raises(ValueError, match="input changed"):
        verify_inputs([r], manifest)


def test_repeated_scoring_does_not_mutate_reviews_or_predictions():
    r = record(); r["judgments"] = [judgment(r)]; predictions = {r["review_id"]: copy.deepcopy(r["judgments"][0]["label"])}
    before = copy.deepcopy((r, predictions))
    assert score([r], predictions) == score([r], predictions)
    assert (r, predictions) == before
