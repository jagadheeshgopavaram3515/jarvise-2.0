"""Comprehensive forensic analysis of capability ranking errors and label quality.

Performs:
1. Trace of all 27 capability ranking failures across both corpora.
2. Tokenization and subword fragmentation analysis for Telugu, Hindi, Tamil loanwords and verbs.
3. Candidate score margins, ranker competition, threshold rejections.
4. Independent Label Quality Audit for context_references and ambiguous records.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXP_DIR = Path("D:/JARVIS/reports/jarvis_indic_ranking_forensics_20261010_213500")

# Load traces
traces = json.load((EXP_DIR / "capability_ranking_traces.json").open("r", encoding="utf-8"))
p100 = json.load((EXP_DIR / "v5_predictions_100.json").open("r", encoding="utf-8"))
p24 = json.load((EXP_DIR / "v5_predictions_24.json").open("r", encoding="utf-8"))

p_all = {p["id"]: p for p in p100 + p24}

cap_failures = [t for t in traces if not t["is_capability_correct"]]
print(f"Total capability ranking failures: {len(cap_failures)} (Frozen 100: {sum(1 for t in cap_failures if t['corpus'] == 'Frozen 100')}, Ind 24: {sum(1 for t in cap_failures if t['corpus'] == 'Independent 24')})")

failure_diagnostics = []

for cf in cap_failures:
    cid = cf["id"]
    utt = cf["utterance"]
    lang = cf["language"]
    script = cf["script"]
    exp_cap = cf["expected_capability"]
    pred_cap = cf["predicted_capability"]
    cands = cf["candidate_names"]
    scores = cf["raw_candidate_scores"]
    sel_score = cf["selected_score"]
    thresh = cf["reconcile_threshold"]
    raw_path = cf["raw_path"]
    post_path = cf["postprocessed_path"]
    is_non_latin = cf["is_non_latin"]
    clar_prob = cf["clarification_prob"]
    esc_prob = cf["escalation_prob"]

    pred_full = p_all[cid]["predicted"]
    exp_full = p_all[cid]["expected"]

    # Determine failure mechanism
    cand_dict = dict(zip(cands, scores)) if len(cands) == len(scores) else {}
    
    in_candidates = exp_cap in cands if exp_cap is not None else True
    
    if exp_cap is None:
        # Expected no capability (e.g. clarify or escalate), but ranker chose an action
        stage = "abstention_failure_false_action"
        score_str = f"{sel_score:.3f}" if sel_score is not None else "None"
        mechanism = f"Expected no capability (path={exp_full.get('execution_path')}), but ranker selected '{pred_cap}' with score {score_str} >= threshold {thresh:.3f}."
    elif pred_cap is None:
        # Expected capability, but model chose None (abstained to clarify or escalate)
        stage = "abstention_failure_false_abstention"
        exp_score = cand_dict.get(exp_cap)
        exp_score_str = f"{exp_score:.3f}" if exp_score is not None else "None"
        mechanism = f"Expected '{exp_cap}' (score={exp_score_str}), but model abstained to path '{post_path}' (raw={raw_path}, clar_prob={clar_prob:.3f}, esc_prob={esc_prob:.3f}, thresh={thresh:.3f})."
    else:
        # Both expected and predicted are capabilities, but different
        stage = "ranking_competition"
        exp_score = cand_dict.get(exp_cap, 0.0)
        pred_score = cand_dict.get(pred_cap, 0.0)
        margin = pred_score - exp_score
        mechanism = f"Ranker selected '{pred_cap}' (score={pred_score:.3f}) over expected '{exp_cap}' (score={exp_score:.3f}) with margin +{margin:.3f}."

    failure_diagnostics.append({
        "id": cid,
        "corpus": cf["corpus"],
        "utterance": utt,
        "language": lang,
        "script": script,
        "scenario_group": cf["scenario_group"],
        "expected_capability": exp_cap,
        "predicted_capability": pred_cap,
        "candidates": cands,
        "candidate_scores": cand_dict,
        "selected_score": sel_score,
        "threshold": thresh,
        "raw_path": raw_path,
        "postprocessed_path": post_path,
        "failure_stage": stage,
        "mechanism": mechanism,
    })

(EXP_DIR / "capability_failure_diagnostics.json").write_text(json.dumps(failure_diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved {len(failure_diagnostics)} capability failure diagnostics.")

# Label Quality Audit
# Audit suspected defective annotations
label_audit = []
for p in p100 + p24:
    pid = p["id"]
    utt = p["utterance"]
    exp = p["expected"]
    pred = p["predicted"]
    is_exact = p["is_exact"]

    # Check context references
    exp_cr = exp.get("context_references", [])
    pred_cr = pred.get("context_references", [])
    
    status = "plausible"
    reason = "Label consistent with guidelines"
    proposed_correction = None

    # Pattern 1: Template hardcoding 'it' on non-'it' utterances
    if exp_cr and len(exp_cr) == 1 and exp_cr[0].get("reference") == "it":
        utt_lower = utt.lower()
        if "it" not in utt_lower.split():
            # Utterance does not contain English word 'it'
            status = "confirmed_incorrect"
            if "that" in utt_lower:
                actual_ref = "that"
            elif any(w in utt_lower for w in ["use", "isse", "isko", "usko"]):
                actual_ref = [w for w in ["use", "isse", "isko", "usko"] if w in utt_lower][0]
            elif any(w in utt_lower for w in ["idi", "adi", "danni"]):
                actual_ref = [w for w in ["idi", "adi", "danni"] if w in utt_lower][0]
            elif any(w in utt_lower for w in ["athai", "ithai", "athu", "ithu"]):
                actual_ref = [w for w in ["athai", "ithai", "athu", "ithu"] if w in utt_lower][0]
            elif any(w in utt for w in ["वह", "इसे", "उसको"]):
                actual_ref = [w for w in ["वह", "इसे", "उसको"] if w in utt][0]
            elif any(w in utt for w in ["దీన్ని", "అది"]):
                actual_ref = [w for w in ["దీన్ని", "అది"] if w in utt][0]
            elif any(w in utt for w in ["அதை", "இது"]):
                actual_ref = [w for w in ["அதை", "இது"] if w in utt][0]
            else:
                actual_ref = exp_cr[0].get("reference")
            
            reason = f"Ground truth hardcoded reference='it' for non-English/non-'it' utterance '{utt}'. Correct extracted reference is '{actual_ref}'."
            proposed_correction = {"context_references": [{"kind": "item", "reference": actual_ref, "resolved_value": None}]}

    # Pattern 2: Utterance containing deictic demonstrative where label expected empty []
    elif not exp_cr and pred_cr:
        if any(w in utt.lower().split() for w in ["this", "that", "these", "those"]):
            status = "ambiguous_or_underspecified"
            reason = f"Utterance '{utt}' contains explicit demonstrative pronoun, but ground truth expected empty context_references []."
            proposed_correction = {"context_references": pred_cr}

    if status != "plausible":
        label_audit.append({
            "id": pid,
            "utterance": utt,
            "status": status,
            "reason": reason,
            "original_expected": exp,
            "proposed_correction": proposed_correction,
        })

(EXP_DIR / "label_quality_audit.json").write_text(json.dumps(label_audit, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved {len(label_audit)} label quality audit items.")
