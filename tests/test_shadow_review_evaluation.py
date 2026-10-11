"""Offline benchmark-audit regressions; semantic screening is explicitly provisional."""
import copy
import importlib
import io
import json
from pathlib import Path
import sys
import types
import urllib.request

import pytest

from benchmarks.fast_classifier.shadow_review_evaluation import evaluate, structural_key


DATA = json.loads((Path(__file__).parent / "fixtures/shadow_review_regression.json").read_text(encoding="utf-8"))
FOLDER_DOMAIN = {"allowed": {"intent_family": ["local_task", "deterministic_action"],
                            "capability_family": ["file"], "execution_path": ["qwen_agent", "fast_path"]},
                 "standalone_clear": True, "basis": "Provisional interpretation: open a named folder."}


@pytest.fixture
def reviewer(monkeypatch):
    with monkeypatch.context() as isolated:
        if "assistant.orchestration" not in sys.modules:
            package = types.ModuleType("assistant.orchestration")
            package.__path__ = [str(Path(__file__).parents[1] / "assistant/orchestration")]
            isolated.setitem(sys.modules, "assistant.orchestration", package)
        return importlib.import_module("benchmarks.fast_classifier.review_fast_brain")


def label():
    return copy.deepcopy(DATA["A"]["candidate"]["canonical_label"])


def proposal(value, mode="AUTO_CONFIDENT", rationale="Candidate matches request meaning."):
    return {"mode": mode, "proposed_label": value, "confidence": "high", "rationale": rationale,
            "source_evidence": [], "human_attention": False}


def test_open_folder_rationale_with_unchanged_information_fields_is_flagged(reviewer):
    original = label()
    original.update(intent_family="information", capability_family="none", execution_path="respond")
    raw = proposal(original, rationale="The user wants to open the archive folder.")
    result = evaluate(original, raw, raw, judgment=FOLDER_DOMAIN)
    assert result["schema_valid"] and result["exact_label_agreement"]
    assert result["potential_rationale_flags"]
    assert result["original_domain_supported_provisional"] is False
    assert not result["gold_accuracy_available"]


def test_invalid_computer_control_enum_is_not_accepted(reviewer):
    original = label()
    wrong = copy.deepcopy(original)
    wrong["intent_family"] = "computer_control"
    result = evaluate(original, proposal(wrong, "AI_CORRECTION"), None)
    assert not result["schema_valid"] and result["label_schema_valid"] is False
    assert result["category"] == "invalid_model_proposal"


def test_valid_material_folder_correction_is_structural_and_provisional(reviewer):
    original = label()
    original.update(intent_family="information", capability_family="none", execution_path="respond")
    corrected = copy.deepcopy(original)
    corrected.update(intent_family="local_task", capability_family="file", execution_path="qwen_agent")
    raw = proposal(corrected, "AI_CORRECTION", "Open the archive folder.")
    result = evaluate(original, raw, raw, judgment=FOLDER_DOMAIN)
    assert result["schema_valid"] and result["label_schema_valid"]
    assert result["material_change"] and not result["exact_label_agreement"]
    assert set(result["changed_fields"]) == {"intent_family", "capability_family", "execution_path"}
    assert result["category"] == "domain_correction_supported_provisional"
    assert not result["gold_accuracy_available"]


def test_agreement_is_not_reported_as_accuracy(reviewer):
    original = label()
    raw = proposal(original)
    result = evaluate(original, raw, raw)
    assert result["schema_valid"] and result["exact_label_agreement"]
    assert result["category"] == "schema_valid_agreement_correctness_unresolved"
    assert not result["gold_accuracy_available"]


def abstention():
    return {"mode": "HUMAN_REVIEW", "proposed_label": None, "confidence": "medium",
            "rationale": "Missing evidence.", "source_evidence": [], "human_attention": True}


def test_missing_referent_abstention_is_distinguished(reviewer):
    result = evaluate(label(), abstention(), abstention(), judgment={"missing_referent": True})
    assert result["schema_valid"] and result["material_change"] is None
    assert result["category"] == "abstention_supported_by_missing_referent_provisional"


def test_clear_request_abstention_is_not_silently_scored_correct(reviewer):
    result = evaluate(label(), abstention(), abstention(), judgment={"standalone_clear": True})
    assert result["schema_valid"]
    assert result["category"] == "possible_unnecessary_abstention_provisional"


def test_code_switched_archive_action_uses_real_schema(reviewer, tmp_path, monkeypatch):
    record = copy.deepcopy(DATA["A"]["candidate"])
    record["text"] = "archive directory paiki teesuku ra"
    record["canonical_label"].update(intent_family="information", capability_family="none", execution_path="respond")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(record), encoding="utf-8")
    store = reviewer.ReviewStore(queue, tmp_path / "decisions.jsonl")
    corrected = copy.deepcopy(record["canonical_label"])
    corrected.update(intent_family="local_task", capability_family="file", execution_path="qwen_agent")
    raw = proposal(corrected, "AI_CORRECTION", "The request means open the archive folder.")
    class Response(io.BytesIO):
        pass
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: Response(json.dumps({"response": json.dumps(raw)}).encode()))
    final = reviewer.OllamaReviewProposer().propose(store.records[0], record["canonical_label"]).to_dict()
    result = evaluate(record["canonical_label"], raw, final, judgment=FOLDER_DOMAIN)
    assert result["schema_valid"] and result["category"] == "domain_correction_supported_provisional"
    assert not store.decisions_path.exists()


def test_exchange_rate_rationale_with_conversation_label_is_flagged(reviewer):
    original = label()
    original.update(intent_family="conversation", execution_path="respond", capability_family="none")
    raw = proposal(original, rationale="This requests the current exchange rate.")
    result = evaluate(original, raw, raw)
    assert result["schema_valid"] and result["potential_rationale_flags"]


def test_multistep_web_rationale_requires_matching_fields(reviewer):
    original = label()
    raw = proposal(original, rationale="This requires multiple steps: find sources, read, compare.")
    result = evaluate(original, raw, raw)
    assert result["schema_valid"] and result["potential_rationale_flags"]
    corrected = copy.deepcopy(original)
    corrected.update(intent_family="multi_step_task", execution_path="qwen_agent")
    raw = proposal(corrected, "AI_CORRECTION", "This requires multiple steps: find sources, read, compare.")
    assert not evaluate(original, raw, raw)["potential_rationale_flags"]


@pytest.mark.parametrize("raw", [None, {}, {"mode": "AUTO_CONFIDENT"}])
def test_missing_or_malformed_proposal_is_not_a_success(reviewer, raw):
    result = evaluate(label(), raw, abstention())
    assert not result["schema_valid"] and result["category"] == "invalid_model_proposal"


def test_timeout_is_not_model_abstention(reviewer):
    result = evaluate(label(), None, abstention(), transport_error={"type": "TimeoutError"})
    assert result["category"] == "inference_error" and not result["schema_valid"]


def test_correction_with_identical_fields_is_a_confirmed_mode_contradiction(reviewer):
    original = label()
    raw = proposal(original, "AI_CORRECTION", "The classification is incorrect.")
    result = evaluate(original, raw, raw)
    assert result["schema_valid"] and result["confirmed_contradictions"]
    assert result["exact_label_agreement"]


def test_repeated_evaluation_preserves_actual_human_history(reviewer):
    queue_before = reviewer.DEFAULT_QUEUE.read_bytes()
    history_before = reviewer.DEFAULT_DECISIONS.read_bytes()
    original = label()
    raw = proposal(original)
    assert evaluate(original, raw, raw) == evaluate(original, raw, raw)
    assert reviewer.DEFAULT_QUEUE.read_bytes() == queue_before
    assert reviewer.DEFAULT_DECISIONS.read_bytes() == history_before


def test_only_harmless_object_representation_is_normalized():
    assert structural_key({"a": 1, "b": 2}) == structural_key({"b": 2, "a": 1})
    assert structural_key({"context": ["memory", "conversation"]}) != structural_key({"context": ["conversation", "memory"]})
