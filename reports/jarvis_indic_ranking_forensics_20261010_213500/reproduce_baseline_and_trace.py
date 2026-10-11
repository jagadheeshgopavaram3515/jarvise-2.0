"""Reproduce native-input v5 baseline and dump detailed capability ranking forensics.

Saves:
- baseline_metrics.json: verified exact match and capability match numbers
- v5_predictions_100.json: full decision mappings for Frozen 100
- v5_predictions_24.json: full decision mappings for Independent 24
- capability_ranking_traces.json: detailed per-candidate cosine similarities,
  margin to threshold, rejection reasons, and language/script diagnostics.
"""
from __future__ import annotations

import json
import time
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

from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    DecisionRequest,
)
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder
from experimental_adapter_v5 import ExperimentalV5Adapter

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


def run_evaluation_and_trace(adapter: ExperimentalV5Adapter, corpus: list[dict[str, Any]], name: str):
    exact_count = 0
    cap_count = 0
    traces = []
    predictions = []

    for r in corpus:
        caps_tuple = tuple(Capability(c["name"], c["description"], tuple(c["required_arguments"])) for c in r["capabilities"])
        req = DecisionRequest(r["utterance"], r["context"], caps_tuple)
        
        pred = dict(adapter.predict(req))
        diag = dict(adapter.last_diagnostics)
        exp = r["expected"]

        is_exact = all(
            (sorted(pred[k]) == sorted(exp[k]) if k == "request_modes" else pred[k] == exp[k])
            for k in FIELDS
        )
        is_cap = pred["selected_capability"] == exp["selected_capability"]

        exact_count += int(is_exact)
        cap_count += int(is_cap)

        predictions.append({
            "id": r["id"],
            "utterance": r["utterance"],
            "is_exact": is_exact,
            "is_cap": is_cap,
            "predicted": pred,
            "expected": exp,
            "diagnostics": diag,
        })

        # Detailed tracing for capability ranking
        traces.append({
            "id": r["id"],
            "corpus": name,
            "utterance": r["utterance"],
            "language": r.get("language"),
            "script": r.get("script"),
            "scenario_group": r.get("scenario_group"),
            "expected_capability": exp["selected_capability"],
            "predicted_capability": pred["selected_capability"],
            "is_capability_correct": is_cap,
            "is_exact_match": is_exact,
            "candidate_names": [c["name"] for c in r["capabilities"]],
            "raw_candidate_scores": diag.get("capability_scores", []),
            "selected_score": diag.get("best_score"),
            "raw_path": diag.get("raw_path"),
            "postprocessed_path": diag.get("postprocessed_path"),
            "reconcile_threshold": 0.58 if diag.get("is_non_latin") else 0.52,
            "is_non_latin": diag.get("is_non_latin", False),
            "clarification_prob": diag.get("clarification_prob", 0.0),
            "escalation_prob": diag.get("escalation_prob", 0.0),
            "domain_token_found": diag.get("has_domain_token", False),
        })

    n = len(corpus)
    return {
        "corpus": name,
        "total": n,
        "exact": exact_count,
        "exact_pct": exact_count / n * 100.0,
        "cap": cap_count,
        "cap_pct": cap_count / n * 100.0,
        "predictions": predictions,
        "traces": traces,
    }


def main():
    print("Loading resources...")
    cache_manifest = json.loads((PRIOR / "feature_cache_manifest.json").read_text(encoding="utf-8"))
    encoder = FrozenMultilingualEncoder(cache_manifest["encoder_snapshot"], 160)
    artifact = joblib.load(EXP4 / "best_candidate/compact_heads.joblib")

    adapter = ExperimentalV5Adapter.__new__(ExperimentalV5Adapter)
    adapter.artifact = artifact
    adapter.encoder = encoder
    adapter._adapter_id = "v5-baseline-trace"
    adapter.inference_count = 0
    adapter.last_inference_metrics = {}

    corpus_100 = load_corpus(FROZEN_100_DIR / "unseen_evaluation_corpus_100.jsonl")
    corpus_24 = load_corpus(EXP4 / "unseen_evaluation_corpus_v4.jsonl")

    print("Evaluating Frozen 100...")
    res_100 = run_evaluation_and_trace(adapter, corpus_100, "Frozen 100")
    print(f"Frozen 100: Exact={res_100['exact']}/100 ({res_100['exact_pct']:.1f}%), Cap={res_100['cap']}/100 ({res_100['cap_pct']:.1f}%)")

    print("Evaluating Independent 24...")
    res_24 = run_evaluation_and_trace(adapter, corpus_24, "Independent 24")
    print(f"Independent 24: Exact={res_24['exact']}/24 ({res_24['exact_pct']:.1f}%), Cap={res_24['cap']}/24 ({res_24['cap_pct']:.1f}%)")

    # Save outputs
    metrics = {
        "frozen_100": {
            "total": res_100["total"],
            "exact": res_100["exact"],
            "exact_pct": res_100["exact_pct"],
            "cap": res_100["cap"],
            "cap_pct": res_100["cap_pct"],
        },
        "independent_24": {
            "total": res_24["total"],
            "exact": res_24["exact"],
            "exact_pct": res_24["exact_pct"],
            "cap": res_24["cap"],
            "cap_pct": res_24["cap_pct"],
        },
    }
    (EXP_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (EXP_DIR / "v5_predictions_100.json").write_text(json.dumps(res_100["predictions"], indent=2, ensure_ascii=False), encoding="utf-8")
    (EXP_DIR / "v5_predictions_24.json").write_text(json.dumps(res_24["predictions"], indent=2, ensure_ascii=False), encoding="utf-8")
    (EXP_DIR / "capability_ranking_traces.json").write_text(json.dumps(res_100["traces"] + res_24["traces"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved baseline metrics and traces to {EXP_DIR}")


if __name__ == "__main__":
    main()

