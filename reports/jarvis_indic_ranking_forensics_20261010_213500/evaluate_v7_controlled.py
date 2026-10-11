"""Controlled Comparative Evaluation: Baseline v5 vs Experimental v7.

Evaluates on:
1. Unchanged Frozen 100-Request Benchmark
2. Unchanged Independent 24-Request Corpus

Reports:
- Strict eight-field exact match (both on original frozen labels and audited/reviewed labels)
- Capability top-1 agreement
- Per-field match rates
- Precision, recall, F1 for capabilities, clarification, and escalation
- Breakdowns by language, script, and scenario group
- Paired change accounting (corrected errors, regressions, unchanged)
- Latency (mean, median, p95)
- Contract validation rate
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

ROOT = Path("D:/JARVIS")
EXP_DIR = Path("D:/JARVIS/reports/jarvis_indic_ranking_forensics_20261010_213500")
V5_DIR = Path("D:/JARVIS/reports/jarvis_brain_forensic_improvement_20261010_183000")
EXP4 = Path("D:/JARVIS/reports/classifier_v4_experiment_20261010_175140")
PRIOR = Path("D:/JARVIS/reports/autonomous_bert_training_20261010_161410")
FROZEN_100_DIR = Path("D:/JARVIS/reports/classifier_v4_frozen_100_eval_20261010_180014")

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V5_DIR))
sys.path.insert(0, str(EXP_DIR))

from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    DecisionRequest,
)
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder
from experimental_adapter_v5 import ExperimentalV5Adapter
from experimental_adapter_v7 import ExperimentalV7Adapter

FIELDS = (
    "intent",
    "execution_path",
    "request_modes",
    "selected_capability",
    "needs_clarification",
    "needs_escalation",
    "missing_information",
    "context_references",
)

# Load label quality audit
audit_file = EXP_DIR / "label_quality_audit.json"
label_audit_map = {}
if audit_file.exists():
    for item in json.loads(audit_file.read_text(encoding="utf-8")):
        if item["status"] == "confirmed_incorrect":
            label_audit_map[item["id"]] = item["proposed_correction"]


def load_corpus(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_model(adapter: Any, corpus: list[dict[str, Any]], corpus_name: str) -> dict[str, Any]:
    exact_frozen = 0
    exact_reviewed = 0
    cap_matches = 0
    field_matches = {f: 0 for f in FIELDS}
    contract_valid_count = 0
    latencies_ms = []

    scenario_metrics: dict[str, dict[str, int]] = {}
    language_metrics: dict[str, dict[str, int]] = {}
    script_metrics: dict[str, dict[str, int]] = {}

    clar_confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    esc_confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    cap_true = []
    cap_pred = []
    predictions = []

    for r in corpus:
        caps_tuple = tuple(Capability(c["name"], c["description"], tuple(c["required_arguments"])) for c in r["capabilities"])
        req = DecisionRequest(r["utterance"], r["context"], caps_tuple)

        t0 = time.perf_counter_ns()
        pred = dict(adapter.predict(req))
        dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        latencies_ms.append(dt_ms)

        try:
            ClassifierDecision.from_mapping(pred)
            contract_valid_count += 1
        except Exception:
            pass

        exp_frozen = r["expected"]
        
        # Audited expectation
        exp_reviewed = dict(exp_frozen)
        if r["id"] in label_audit_map:
            exp_reviewed.update(label_audit_map[r["id"]])

        is_exact_frozen = all(
            (sorted(pred[k]) == sorted(exp_frozen[k]) if k == "request_modes" else pred[k] == exp_frozen[k])
            for k in FIELDS
        )
        is_exact_reviewed = all(
            (sorted(pred[k]) == sorted(exp_reviewed[k]) if k == "request_modes" else pred[k] == exp_reviewed[k])
            for k in FIELDS
        )
        is_cap = pred["selected_capability"] == exp_frozen["selected_capability"]

        exact_frozen += int(is_exact_frozen)
        exact_reviewed += int(is_exact_reviewed)
        cap_matches += int(is_cap)

        for k in FIELDS:
            if k == "request_modes":
                field_matches[k] += int(sorted(pred[k]) == sorted(exp_frozen[k]))
            else:
                field_matches[k] += int(pred[k] == exp_frozen[k])

        # Clarification confusion
        exp_c = exp_frozen.get("needs_clarification", False)
        pred_c = pred.get("needs_clarification", False)
        if exp_c and pred_c:
            clar_confusion["TP"] += 1
        elif not exp_c and pred_c:
            clar_confusion["FP"] += 1
        elif not exp_c and not pred_c:
            clar_confusion["TN"] += 1
        else:
            clar_confusion["FN"] += 1

        # Escalation confusion
        exp_e = exp_frozen.get("needs_escalation", False)
        pred_e = pred.get("needs_escalation", False)
        if exp_e and pred_e:
            esc_confusion["TP"] += 1
        elif not exp_e and pred_e:
            esc_confusion["FP"] += 1
        elif not exp_e and not pred_e:
            esc_confusion["TN"] += 1
        else:
            esc_confusion["FN"] += 1

        cap_true.append(exp_frozen["selected_capability"] or "__none__")
        cap_pred.append(pred["selected_capability"] or "__none__")

        scen = r.get("scenario_group", "unknown")
        if scen not in scenario_metrics:
            scenario_metrics[scen] = {"total": 0, "exact_frozen": 0, "exact_reviewed": 0, "cap": 0}
        scenario_metrics[scen]["total"] += 1
        scenario_metrics[scen]["exact_frozen"] += int(is_exact_frozen)
        scenario_metrics[scen]["exact_reviewed"] += int(is_exact_reviewed)
        scenario_metrics[scen]["cap"] += int(is_cap)

        lang = r.get("language", "unknown")
        if lang not in language_metrics:
            language_metrics[lang] = {"total": 0, "exact_frozen": 0, "exact_reviewed": 0, "cap": 0}
        language_metrics[lang]["total"] += 1
        language_metrics[lang]["exact_frozen"] += int(is_exact_frozen)
        language_metrics[lang]["exact_reviewed"] += int(is_exact_reviewed)
        language_metrics[lang]["cap"] += int(is_cap)

        script = r.get("script", "unknown")
        if script not in script_metrics:
            script_metrics[script] = {"total": 0, "exact_frozen": 0, "exact_reviewed": 0, "cap": 0}
        script_metrics[script]["total"] += 1
        script_metrics[script]["exact_frozen"] += int(is_exact_frozen)
        script_metrics[script]["exact_reviewed"] += int(is_exact_reviewed)
        script_metrics[script]["cap"] += int(is_cap)

        predictions.append({
            "id": r["id"],
            "utterance": r["utterance"],
            "is_exact_frozen": is_exact_frozen,
            "is_exact_reviewed": is_exact_reviewed,
            "is_cap": is_cap,
            "predicted": pred,
            "expected_frozen": exp_frozen,
            "expected_reviewed": exp_reviewed,
        })

    n = len(corpus)
    # Clarification metrics
    c_prec = clar_confusion["TP"] / (clar_confusion["TP"] + clar_confusion["FP"]) if (clar_confusion["TP"] + clar_confusion["FP"]) else 0.0
    c_rec = clar_confusion["TP"] / (clar_confusion["TP"] + clar_confusion["FN"]) if (clar_confusion["TP"] + clar_confusion["FN"]) else 0.0
    c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) else 0.0

    # Escalation metrics
    e_prec = esc_confusion["TP"] / (esc_confusion["TP"] + esc_confusion["FP"]) if (esc_confusion["TP"] + esc_confusion["FP"]) else 0.0
    e_rec = esc_confusion["TP"] / (esc_confusion["TP"] + esc_confusion["FN"]) if (esc_confusion["TP"] + esc_confusion["FN"]) else 0.0
    e_f1 = 2 * e_prec * e_rec / (e_prec + e_rec) if (e_prec + e_rec) else 0.0

    return {
        "corpus_name": corpus_name,
        "total": n,
        "exact_frozen": exact_frozen,
        "exact_frozen_pct": exact_frozen / n * 100.0,
        "exact_reviewed": exact_reviewed,
        "exact_reviewed_pct": exact_reviewed / n * 100.0,
        "cap": cap_matches,
        "cap_pct": cap_matches / n * 100.0,
        "contract_valid_pct": contract_valid_count / n * 100.0,
        "per_field_pct": {k: v / n * 100.0 for k, v in field_matches.items()},
        "clarification_confusion": clar_confusion,
        "clarification_metrics": {"precision": c_prec, "recall": c_rec, "f1": c_f1},
        "escalation_confusion": esc_confusion,
        "escalation_metrics": {"precision": e_prec, "recall": e_rec, "f1": e_f1},
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_median_ms": float(np.median(latencies_ms)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "scenario_metrics": scenario_metrics,
        "language_metrics": language_metrics,
        "script_metrics": script_metrics,
        "predictions": predictions,
    }


def compare_models(p_base: list[dict[str, Any]], p_cand: list[dict[str, Any]]):
    corrected_frozen = []
    regressions_frozen = []
    corrected_cap = []
    regressions_cap = []
    unchanged = 0

    for i in range(len(p_base)):
        b = p_base[i]
        c = p_cand[i]
        # Frozen exact match diff
        if not b["is_exact_frozen"] and c["is_exact_frozen"]:
            corrected_frozen.append(b["id"])
        elif b["is_exact_frozen"] and not c["is_exact_frozen"]:
            regressions_frozen.append(b["id"])

        # Cap top-1 diff
        if not b["is_cap"] and c["is_cap"]:
            corrected_cap.append(b["id"])
        elif b["is_cap"] and not c["is_cap"]:
            regressions_cap.append(b["id"])

        if b["is_exact_frozen"] == c["is_exact_frozen"] and b["is_cap"] == c["is_cap"]:
            unchanged += 1

    return {
        "corrected_frozen": corrected_frozen,
        "regressions_frozen": regressions_frozen,
        "corrected_capability": corrected_cap,
        "regressions_capability": regressions_cap,
        "net_exact_frozen": len(corrected_frozen) - len(regressions_frozen),
        "net_capability": len(corrected_cap) - len(regressions_cap),
        "unchanged": unchanged,
    }


def main():
    print("=" * 65)
    print("CONTROLLED EVALUATION: BASELINE v5 vs EXPERIMENTAL v7 ADAPTER")
    print("=" * 65)

    cache_manifest = json.loads((PRIOR / "feature_cache_manifest.json").read_text(encoding="utf-8"))
    encoder = FrozenMultilingualEncoder(cache_manifest["encoder_snapshot"], 160)
    artifact = joblib.load(EXP4 / "best_candidate/compact_heads.joblib")

    # Adapter v5
    v5 = ExperimentalV5Adapter.__new__(ExperimentalV5Adapter)
    v5.artifact = artifact
    v5.encoder = encoder
    v5._adapter_id = "v5-baseline"
    v5.inference_count = 0
    v5.last_inference_metrics = {}

    # Adapter v7
    v7 = ExperimentalV7Adapter.__new__(ExperimentalV7Adapter)
    v7.artifact = artifact
    v7.encoder = encoder
    v7._adapter_id = "v7-indic-ranking"
    v7.inference_count = 0
    v7.last_inference_metrics = {}
    from slot_contract_resolver import SlotContractResolver
    v7.contract_resolver = SlotContractResolver()

    c100 = load_corpus(FROZEN_100_DIR / "unseen_evaluation_corpus_100.jsonl")
    c24 = load_corpus(EXP4 / "unseen_evaluation_corpus_v4.jsonl")

    # Evaluate Frozen 100
    print("\n--- Evaluating Frozen 100 Corpus (100 records) ---")
    r100_v5 = evaluate_model(v5, c100, "Frozen 100")
    print(f"v5 Baseline: Exact(Frozen)={r100_v5['exact_frozen']}/100 ({r100_v5['exact_frozen_pct']:.1f}%), Exact(Reviewed)={r100_v5['exact_reviewed']}/100 ({r100_v5['exact_reviewed_pct']:.1f}%), Cap={r100_v5['cap']}/100 ({r100_v5['cap_pct']:.1f}%)")

    r100_v7 = evaluate_model(v7, c100, "Frozen 100")
    print(f"v7 Candidate: Exact(Frozen)={r100_v7['exact_frozen']}/100 ({r100_v7['exact_frozen_pct']:.1f}%), Exact(Reviewed)={r100_v7['exact_reviewed']}/100 ({r100_v7['exact_reviewed_pct']:.1f}%), Cap={r100_v7['cap']}/100 ({r100_v7['cap_pct']:.1f}%)")

    comp100 = compare_models(r100_v5["predictions"], r100_v7["predictions"])
    print(f"Frozen 100 Paired Changes:")
    print(f"  Capability Top-1: +{len(comp100['corrected_capability'])} fixed, -{len(comp100['regressions_capability'])} regressed (Net: {comp100['net_capability']:+d})")
    print(f"  Exact Frozen:    +{len(comp100['corrected_frozen'])} fixed, -{len(comp100['regressions_frozen'])} regressed (Net: {comp100['net_exact_frozen']:+d})")

    # Evaluate Independent 24
    print("\n--- Evaluating Independent 24 Corpus (24 records) ---")
    r24_v5 = evaluate_model(v5, c24, "Independent 24")
    print(f"v5 Baseline: Exact(Frozen)={r24_v5['exact_frozen']}/24 ({r24_v5['exact_frozen_pct']:.1f}%), Exact(Reviewed)={r24_v5['exact_reviewed']}/24 ({r24_v5['exact_reviewed_pct']:.1f}%), Cap={r24_v5['cap']}/24 ({r24_v5['cap_pct']:.1f}%)")

    r24_v7 = evaluate_model(v7, c24, "Independent 24")
    print(f"v7 Candidate: Exact(Frozen)={r24_v7['exact_frozen']}/24 ({r24_v7['exact_frozen_pct']:.1f}%), Exact(Reviewed)={r24_v7['exact_reviewed']}/24 ({r24_v7['exact_reviewed_pct']:.1f}%), Cap={r24_v7['cap']}/24 ({r24_v7['cap_pct']:.1f}%)")

    comp24 = compare_models(r24_v5["predictions"], r24_v7["predictions"])
    print(f"Independent 24 Paired Changes:")
    print(f"  Capability Top-1: +{len(comp24['corrected_capability'])} fixed, -{len(comp24['regressions_capability'])} regressed (Net: {comp24['net_capability']:+d})")
    print(f"  Exact Frozen:    +{len(comp24['corrected_frozen'])} fixed, -{len(comp24['regressions_frozen'])} regressed (Net: {comp24['net_exact_frozen']:+d})")

    summary = {
        "frozen_100": {
            "v5": {k: v for k, v in r100_v5.items() if k != "predictions"},
            "v7": {k: v for k, v in r100_v7.items() if k != "predictions"},
            "comparison": comp100,
        },
        "independent_24": {
            "v5": {k: v for k, v in r24_v5.items() if k != "predictions"},
            "v7": {k: v for k, v in r24_v7.items() if k != "predictions"},
            "comparison": comp24,
        },
    }

    (EXP_DIR / "controlled_evaluation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (EXP_DIR / "v7_predictions_100.json").write_text(json.dumps(r100_v7["predictions"], indent=2, ensure_ascii=False), encoding="utf-8")
    (EXP_DIR / "v7_predictions_24.json").write_text(json.dumps(r24_v7["predictions"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved evaluation summary and predictions to {EXP_DIR}")


if __name__ == "__main__":
    main()

