"""Controlled Comparative Evaluation of Baseline v5 vs Experimental v6 Adapter.

Evaluates on:
1. Unchanged Frozen 100 Corpus
2. Unchanged Independent 24 Corpus

Produces complete per-field metrics, scenario group breakdowns, language breakdowns,
clarification confusion matrices, contract validation rates, latency measurements,
and audit logs.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

ROOT = Path("D:/JARVIS")
EXP_DIR = Path("D:/JARVIS/reports/jarvis_slot_contract_fix_20261010_211600")
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
from experimental_adapter_v6 import ExperimentalV6Adapter

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


def load_corpus(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_adapter(adapter: Any, corpus: list[dict[str, Any]], corpus_name: str) -> dict[str, Any]:
    exact_matches = 0
    cap_matches = 0
    field_matches = {f: 0 for f in FIELDS}
    contract_valid_count = 0
    latencies_ms = []

    scenario_metrics: dict[str, dict[str, int]] = {}
    language_metrics: dict[str, dict[str, int]] = {}
    script_metrics: dict[str, dict[str, int]] = {}

    clarification_confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    predictions = []

    for r in corpus:
        req = DecisionRequest(
            r["utterance"],
            r["context"],
            tuple(Capability(c["name"], c["description"], tuple(c["required_arguments"])) for c in r["capabilities"]),
        )
        t0 = time.perf_counter_ns()
        pred = dict(adapter.predict(req))
        dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        latencies_ms.append(dt_ms)

        # Contract validation check
        try:
            ClassifierDecision.from_mapping(pred)
            contract_valid_count += 1
        except Exception:
            pass

        exp = r["expected"]
        is_exact = all(
            (sorted(pred[k]) == sorted(exp[k]) if k == "request_modes" else pred[k] == exp[k])
            for k in FIELDS
        )
        is_cap = pred["selected_capability"] == exp["selected_capability"]

        exact_matches += int(is_exact)
        cap_matches += int(is_cap)

        for k in FIELDS:
            if k == "request_modes":
                field_matches[k] += int(sorted(pred[k]) == sorted(exp[k]))
            else:
                field_matches[k] += int(pred[k] == exp[k])

        # Clarification confusion
        exp_clar = exp.get("needs_clarification", False)
        pred_clar = pred.get("needs_clarification", False)
        if exp_clar and pred_clar:
            clarification_confusion["TP"] += 1
        elif not exp_clar and pred_clar:
            clarification_confusion["FP"] += 1
        elif not exp_clar and not pred_clar:
            clarification_confusion["TN"] += 1
        else:
            clarification_confusion["FN"] += 1

        # Scenario group
        scen = r.get("scenario_group", "unknown")
        if scen not in scenario_metrics:
            scenario_metrics[scen] = {"total": 0, "exact": 0, "cap": 0}
        scenario_metrics[scen]["total"] += 1
        scenario_metrics[scen]["exact"] += int(is_exact)
        scenario_metrics[scen]["cap"] += int(is_cap)

        # Language
        lang = r.get("language", "unknown")
        if lang not in language_metrics:
            language_metrics[lang] = {"total": 0, "exact": 0, "cap": 0}
        language_metrics[lang]["total"] += 1
        language_metrics[lang]["exact"] += int(is_exact)
        language_metrics[lang]["cap"] += int(is_cap)

        # Script
        script = r.get("script", "unknown")
        if script not in script_metrics:
            script_metrics[script] = {"total": 0, "exact": 0, "cap": 0}
        script_metrics[script]["total"] += 1
        script_metrics[script]["exact"] += int(is_exact)
        script_metrics[script]["cap"] += int(is_cap)

        predictions.append({
            "id": r["id"],
            "utterance": r["utterance"],
            "is_exact": is_exact,
            "is_cap": is_cap,
            "predicted": pred,
            "expected": exp,
        })

    n = len(corpus)
    return {
        "corpus_name": corpus_name,
        "total": n,
        "exact": exact_matches,
        "exact_pct": exact_matches / n * 100.0,
        "cap": cap_matches,
        "cap_pct": cap_matches / n * 100.0,
        "contract_valid_pct": contract_valid_count / n * 100.0,
        "per_field_pct": {k: v / n * 100.0 for k, v in field_matches.items()},
        "clarification_confusion": clarification_confusion,
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "scenario_metrics": scenario_metrics,
        "language_metrics": language_metrics,
        "script_metrics": script_metrics,
        "predictions": predictions,
    }


def main():
    print("=" * 60)
    print("Evaluating Slot Contract Alignment: Baseline v5 vs v6")
    print("=" * 60)

    # 1. Load resources
    cache_manifest = json.loads((PRIOR / "feature_cache_manifest.json").read_text(encoding="utf-8"))
    encoder = FrozenMultilingualEncoder(cache_manifest["encoder_snapshot"], 160)
    artifact = joblib.load(EXP4 / "best_candidate/compact_heads.joblib")

    # 2. Instantiate adapters
    adapter_v5 = ExperimentalV5Adapter.__new__(ExperimentalV5Adapter)
    adapter_v5.artifact = artifact
    adapter_v5.encoder = encoder
    adapter_v5._adapter_id = "v5-baseline"
    adapter_v5.inference_count = 0
    adapter_v5.last_inference_metrics = {}

    adapter_v6 = ExperimentalV6Adapter.__new__(ExperimentalV6Adapter)
    adapter_v6.artifact = artifact
    adapter_v6.encoder = encoder
    adapter_v6._adapter_id = "v6-slot-contract"
    adapter_v6.inference_count = 0
    adapter_v6.last_inference_metrics = {}
    from slot_contract_resolver import SlotContractResolver
    adapter_v6.contract_resolver = SlotContractResolver()
    adapter_v6.last_contract_metrics = {}

    corpus_100 = load_corpus(FROZEN_100_DIR / "unseen_evaluation_corpus_100.jsonl")
    corpus_24 = load_corpus(EXP4 / "unseen_evaluation_corpus_v4.jsonl")

    # 3. Run evaluations
    print("Evaluating v5 on Frozen 100...")
    v5_100 = evaluate_adapter(adapter_v5, corpus_100, "Frozen 100")
    print(f"v5 100: Exact={v5_100['exact']}/100 ({v5_100['exact_pct']:.1f}%), Cap={v5_100['cap']}/100 ({v5_100['cap_pct']:.1f}%)")

    print("Evaluating v6 on Frozen 100...")
    v6_100 = evaluate_adapter(adapter_v6, corpus_100, "Frozen 100")
    print(f"v6 100: Exact={v6_100['exact']}/100 ({v6_100['exact_pct']:.1f}%), Cap={v6_100['cap']}/100 ({v6_100['cap_pct']:.1f}%)")

    print("Evaluating v5 on Independent 24...")
    v5_24 = evaluate_adapter(adapter_v5, corpus_24, "Independent 24")
    print(f"v5 24:  Exact={v5_24['exact']}/24 ({v5_24['exact_pct']:.1f}%), Cap={v5_24['cap']}/24 ({v5_24['cap_pct']:.1f}%)")

    print("Evaluating v6 on Independent 24...")
    v6_24 = evaluate_adapter(adapter_v6, corpus_24, "Independent 24")
    print(f"v6 24:  Exact={v6_24['exact']}/24 ({v6_24['exact_pct']:.1f}%), Cap={v6_24['cap']}/24 ({v6_24['cap_pct']:.1f}%)")

    # Compare predictions: check regressions or fixes
    def compare_preds(p5, p6):
        fixes = []
        regressions = []
        identical = 0
        for i in range(len(p5)):
            r5 = p5[i]
            r6 = p6[i]
            if r5["is_exact"] and not r6["is_exact"]:
                regressions.append(r5["id"])
            elif not r5["is_exact"] and r6["is_exact"]:
                fixes.append(r5["id"])
            else:
                identical += 1
        return {"fixes": fixes, "regressions": regressions, "unchanged": identical}

    comp_100 = compare_preds(v5_100["predictions"], v6_100["predictions"])
    comp_24 = compare_preds(v5_24["predictions"], v6_24["predictions"])

    print(f"\nFrozen 100 Comparison: Fixes={len(comp_100['fixes'])}, Regressions={len(comp_100['regressions'])}, Unchanged={comp_100['unchanged']}")
    print(f"Independent 24 Comparison: Fixes={len(comp_24['fixes'])}, Regressions={len(comp_24['regressions'])}, Unchanged={comp_24['unchanged']}")

    # Save summary metrics
    summary = {
        "frozen_100": {
            "v5": {k: v for k, v in v5_100.items() if k != "predictions"},
            "v6": {k: v for k, v in v6_100.items() if k != "predictions"},
            "comparison": comp_100,
        },
        "independent_24": {
            "v5": {k: v for k, v in v5_24.items() if k != "predictions"},
            "v6": {k: v for k, v in v6_24.items() if k != "predictions"},
            "comparison": comp_24,
        },
    }

    (EXP_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (EXP_DIR / "predictions_v6_100.json").write_text(json.dumps(v6_100["predictions"], indent=2, ensure_ascii=False), encoding="utf-8")
    (EXP_DIR / "predictions_v6_24.json").write_text(json.dumps(v6_24["predictions"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved evaluation summary and predictions to {EXP_DIR}")


if __name__ == "__main__":
    main()

