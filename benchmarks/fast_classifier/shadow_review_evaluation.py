"""Read-only evaluation helpers; categories here are not classifier enum values.

Schema validity and structural contradictions are deterministic. Semantic domain
checks are screening flags, not independent gold labels or accuracy estimates.
"""
from __future__ import annotations

import json
import re
from typing import Any


def structural_key(value: Any) -> str:
    """Ignore object-key ordering/spacing only; preserve array order and values."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def label_error(value: Any) -> str | None:
    from assistant.orchestration.fast_brain.contract import FastBrainDecision
    try:
        FastBrainDecision.from_mapping(value)
    except (ValueError, TypeError, AttributeError) as exc:
        return str(exc)
    return None


def evaluate(candidate: dict, raw: Any, final: Any, *, judgment: dict | None = None,
             transport_error: dict | None = None) -> dict:
    """Audit full labels; optional judgments describe provisional semantic domains.

    A judgment is deliberately not a complete seven-field gold label. It cannot
    establish uncertainty/context/escalation correctness or measured accuracy.
    """
    from benchmarks.fast_classifier.review_fast_brain import validate_ai_proposal
    result = {"schema_valid": False, "schema_error": None, "label_schema_valid": None,
              "exact_label_agreement": None, "material_change": None,
              "changed_fields": [], "confirmed_contradictions": [],
              "potential_rationale_flags": [], "semantic_judgment": judgment,
              "gold_accuracy_available": False}
    if transport_error:
        result.update(category="inference_error", schema_error=transport_error)
        return result
    try:
        proposal = validate_ai_proposal(raw)
        result["schema_valid"] = True
    except (ValueError, TypeError, AttributeError) as exc:
        result.update(category="invalid_model_proposal", schema_error=str(exc))
        if isinstance(raw, dict) and raw.get("proposed_label") is not None:
            result["label_schema_valid"] = label_error(raw["proposed_label"]) is None
        return result
    value = proposal.to_dict()
    result["application_changed_proposal"] = structural_key(value) != structural_key(final)
    proposed = value["proposed_label"]
    if proposed is not None:
        result["label_schema_valid"] = label_error(proposed) is None
        equal = structural_key(proposed) == structural_key(candidate)
        result["exact_label_agreement"] = equal
        result["material_change"] = not equal
        result["changed_fields"] = [k for k in sorted(set(candidate) | set(proposed))
                                    if structural_key(candidate.get(k)) != structural_key(proposed.get(k))]
        if proposal.mode == "AI_CORRECTION" and equal:
            result["confirmed_contradictions"].append("AI_CORRECTION proposes an identical label")
        if proposal.mode == "AUTO_CONFIDENT" and not equal:
            result["confirmed_contradictions"].append("AUTO_CONFIDENT changes the candidate label")
        rationale = proposal.rationale.lower()
        if equal and re.search(r"\b(incorrect|misclassified|wrong classification|needs correction)\b", rationale):
            result["potential_rationale_flags"].append("Rationale claims an error but structured label is unchanged")
        if re.search(r"\b(open(?:ing)?|bring up)\b[^.!?]*\b(folder|directory)\b", rationale):
            if proposed.get("capability_family") != "file" or proposed.get("execution_path") == "respond":
                result["potential_rationale_flags"].append("Opening a folder described, but file/action fields do not match")
        if re.search(r"\b(exchange rate|current information|latest information)\b", rationale):
            if proposed.get("intent_family") not in {"research", "multi_step_task"}:
                result["potential_rationale_flags"].append("Current-information rationale with non-research intent")
        if re.search(r"\b(multi.step|multiple steps)\b", rationale):
            if proposed.get("intent_family") != "multi_step_task":
                result["potential_rationale_flags"].append("Multi-step rationale with a non-multi-step label")
    domain_original = domain_proposed = None
    if judgment:
        def fits(label):
            return all(label.get(field) in accepted for field, accepted in judgment.get("allowed", {}).items())
        domain_original = fits(candidate)
        domain_proposed = fits(proposed) if proposed is not None else None
        result["original_domain_supported_provisional"] = domain_original
        result["proposed_domain_supported_provisional"] = domain_proposed
    if result["confirmed_contradictions"] or result["potential_rationale_flags"]:
        category = "rationale_or_mode_label_inconsistency"
    elif proposed is None:
        if judgment and judgment.get("missing_referent"):
            category = "abstention_supported_by_missing_referent_provisional"
        elif judgment and judgment.get("standalone_clear"):
            category = "possible_unnecessary_abstention_provisional"
        else:
            category = "abstention_unresolved"
    elif domain_proposed is False:
        category = "proposal_domain_questionable_provisional"
    elif result["material_change"]:
        category = "schema_valid_material_change_semantics_unresolved"
        if domain_original is False and domain_proposed is True:
            category = "domain_correction_supported_provisional"
    else:
        category = "schema_valid_agreement_correctness_unresolved"
        if domain_original is False:
            category = "agreement_with_questionable_original_provisional"
    result["category"] = category
    result["explanation"] = ("; ".join(result["confirmed_contradictions"] + result["potential_rationale_flags"])
                             or "Schema/structure established; full semantic correctness requires independent adjudication.")
    return result
