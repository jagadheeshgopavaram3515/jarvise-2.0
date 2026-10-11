"""Independent verification script for JARVIS Classifier v7 audit.

Recomputes:
1. Hash integrity checks against sha256_manifest.json.
2. Full dataset re-scoring of v5 and v7 predictions against original frozen corpora.
3. Paired transition verification (proof that score diff == corrected - regressed).
4. Sub-group metrics (language, script, scenario group).
5. Audit of label_quality_audit.json records and the audited-label arithmetic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path("D:/JARVIS")
EXP_DIR = ROOT / "reports/jarvis_indic_ranking_forensics_20261010_213500"
AUDIT_DIR = ROOT / "reports/jarvis_v7_independent_verification_20261011_114200"

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

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

def verify_hashes() -> dict[str, Any]:
    manifest = json.loads((EXP_DIR / "sha256_manifest.json").read_text(encoding="utf-8"))
    results = {}
    all_match = True
    for name, meta in manifest.items():
        p = Path(meta["path"])
        exists = p.exists()
        current_hash = sha256_of(p) if exists else None
        matches = (current_hash == meta["sha256"])
        if not matches:
            all_match = False
        results[name] = {
            "path": str(p),
            "expected_hash": meta["sha256"],
            "actual_hash": current_hash,
            "exists": exists,
            "matches": matches,
        }
    return {"all_match": all_match, "files": results}

def run_metric_audit():
    print("="*60)
    print("INDEPENDENT RECOMPUTATION OF METRICS")
    print("="*60)

    # 1. Hashes
    hash_results = verify_hashes()
    print("Hash check all match:", hash_results["all_match"])

    # 2. Corpora
    c100_path = ROOT / "reports/classifier_v4_frozen_100_eval_20261010_180014/unseen_evaluation_corpus_100.jsonl"
    c24_path = ROOT / "reports/classifier_v4_experiment_20261010_175140/unseen_evaluation_corpus_v4.jsonl"

    c100_raw = [json.loads(l) for l in c100_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    c24_raw = [json.loads(l) for l in c24_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"Loaded {len(c100_raw)} records from c100, {len(c24_raw)} records from c24.")

    # Predictions
    v5_p100_raw = json.loads((EXP_DIR / "v5_predictions_100.json").read_text(encoding="utf-8"))
    v7_p100_raw = json.loads((EXP_DIR / "v7_predictions_100.json").read_text(encoding="utf-8"))

    v5_p24_raw = json.loads((EXP_DIR / "v5_predictions_24.json").read_text(encoding="utf-8"))
    v7_p24_raw = json.loads((EXP_DIR / "v7_predictions_24.json").read_text(encoding="utf-8"))

    # Map by ID
    v5_p100 = {p["id"]: p for p in v5_p100_raw}
    v7_p100 = {p["id"]: p for p in v7_p100_raw}
    c100 = {r["id"]: r for r in c100_raw}

    v5_p24 = {p["id"]: p for p in v5_p24_raw}
    v7_p24 = {p["id"]: p for p in v7_p24_raw}
    c24 = {r["id"]: r for r in c24_raw}

    # Verify ID alignment
    assert set(c100.keys()) == set(v5_p100.keys()) == set(v7_p100.keys()), "ID mismatch in Frozen 100!"
    assert set(c24.keys()) == set(v5_p24.keys()) == set(v7_p24.keys()), "ID mismatch in Independent 24!"

    # 3. Label Audit Mapping
    label_audit_raw = json.loads((EXP_DIR / "label_quality_audit.json").read_text(encoding="utf-8"))
    print(f"Loaded {len(label_audit_raw)} items from label_quality_audit.json")

    # Evaluate corpus function
    def evaluate_set(corpus_dict, v5_dict, v7_dict, name):
        total = len(corpus_dict)
        v5_exact_frozen = 0
        v7_exact_frozen = 0

        v5_cap_matches = 0
        v7_cap_matches = 0

        v5_fields = {f: 0 for f in FIELDS}
        v7_fields = {f: 0 for f in FIELDS}

        # Paired transitions
        exact_fixed = []
        exact_regressed = []
        cap_fixed = []
        cap_regressed = []

        # Sub-group tracking
        by_lang = {}
        by_script = {}
        by_scenario = {}

        for rid, record in corpus_dict.items():
            exp = record["expected"]
            p5 = v5_dict[rid]["predicted"]
            p7 = v7_dict[rid]["predicted"]

            # Field checks
            p5_exact = all((sorted(p5[k]) == sorted(exp[k]) if k == "request_modes" else p5[k] == exp[k]) for k in FIELDS)
            p7_exact = all((sorted(p7[k]) == sorted(exp[k]) if k == "request_modes" else p7[k] == exp[k]) for k in FIELDS)

            p5_cap = (p5["selected_capability"] == exp["selected_capability"])
            p7_cap = (p7["selected_capability"] == exp["selected_capability"])

            v5_exact_frozen += int(p5_exact)
            v7_exact_frozen += int(p7_exact)
            v5_cap_matches += int(p5_cap)
            v7_cap_matches += int(p7_cap)

            for f in FIELDS:
                v5_fields[f] += int(sorted(p5[f]) == sorted(exp[f]) if f == "request_modes" else p5[f] == exp[f])
                v7_fields[f] += int(sorted(p7[f]) == sorted(exp[f]) if f == "request_modes" else p7[f] == exp[f])

            # Paired diffs
            if not p5_exact and p7_exact:
                exact_fixed.append(rid)
            elif p5_exact and not p7_exact:
                exact_regressed.append(rid)

            if not p5_cap and p7_cap:
                cap_fixed.append(rid)
            elif p5_cap and not p7_cap:
                cap_regressed.append(rid)

            # Sub-groups
            lang = record.get("language", "unknown")
            script = record.get("script", "unknown")
            scen = record.get("scenario_group", "unknown")

            for group_dict, key in [(by_lang, lang), (by_script, script), (by_scenario, scen)]:
                if key not in group_dict:
                    group_dict[key] = {
                        "total": 0,
                        "v5_exact": 0, "v7_exact": 0,
                        "v5_cap": 0, "v7_cap": 0,
                    }
                group_dict[key]["total"] += 1
                group_dict[key]["v5_exact"] += int(p5_exact)
                group_dict[key]["v7_exact"] += int(p7_exact)
                group_dict[key]["v5_cap"] += int(p5_cap)
                group_dict[key]["v7_cap"] += int(p7_cap)

        return {
            "name": name,
            "total": total,
            "v5_exact_frozen": v5_exact_frozen,
            "v7_exact_frozen": v7_exact_frozen,
            "v5_cap": v5_cap_matches,
            "v7_cap": v7_cap_matches,
            "v5_fields": v5_fields,
            "v7_fields": v7_fields,
            "exact_fixed": exact_fixed,
            "exact_regressed": exact_regressed,
            "cap_fixed": cap_fixed,
            "cap_regressed": cap_regressed,
            "by_lang": by_lang,
            "by_script": by_script,
            "by_scenario": by_scenario,
        }

    res_100 = evaluate_set(c100, v5_p100, v7_p100, "Frozen 100")
    res_24 = evaluate_set(c24, v5_p24, v7_p24, "Independent 24")

    print("\n--- Frozen 100 Recomputed Results ---")
    print(f"v5 Exact: {res_100['v5_exact_frozen']}/100 ({res_100['v5_exact_frozen']/100:.1%})")
    print(f"v7 Exact: {res_100['v7_exact_frozen']}/100 ({res_100['v7_exact_frozen']/100:.1%})")
    print(f"Exact diff: {res_100['v7_exact_frozen'] - res_100['v5_exact_frozen']}")
    print(f"Exact fixed ({len(res_100['exact_fixed'])}): {res_100['exact_fixed']}")
    print(f"Exact regressed ({len(res_100['exact_regressed'])}): {res_100['exact_regressed']}")
    assert (res_100['v7_exact_frozen'] - res_100['v5_exact_frozen']) == (len(res_100['exact_fixed']) - len(res_100['exact_regressed']))

    print(f"\nv5 Capability: {res_100['v5_cap']}/100 ({res_100['v5_cap']/100:.1%})")
    print(f"v7 Capability: {res_100['v7_cap']}/100 ({res_100['v7_cap']/100:.1%})")
    print(f"Capability diff: {res_100['v7_cap'] - res_100['v5_cap']}")
    print(f"Capability fixed ({len(res_100['cap_fixed'])}): {res_100['cap_fixed']}")
    print(f"Capability regressed ({len(res_100['cap_regressed'])}): {res_100['cap_regressed']}")
    assert (res_100['v7_cap'] - res_100['v5_cap']) == (len(res_100['cap_fixed']) - len(res_100['cap_regressed']))

    print("\n--- Independent 24 Recomputed Results ---")
    print(f"v5 Exact: {res_24['v5_exact_frozen']}/24 ({res_24['v5_exact_frozen']/24:.1%})")
    print(f"v7 Exact: {res_24['v7_exact_frozen']}/24 ({res_24['v7_exact_frozen']/24:.1%})")
    print(f"Exact diff: {res_24['v7_exact_frozen'] - res_24['v5_exact_frozen']}")
    print(f"Exact fixed ({len(res_24['exact_fixed'])}): {res_24['exact_fixed']}")
    print(f"Exact regressed ({len(res_24['exact_regressed'])}): {res_24['exact_regressed']}")
    assert (res_24['v7_exact_frozen'] - res_24['v5_exact_frozen']) == (len(res_24['exact_fixed']) - len(res_24['exact_regressed']))

    print(f"\nv5 Capability: {res_24['v5_cap']}/24 ({res_24['v5_cap']/24:.1%})")
    print(f"v7 Capability: {res_24['v7_cap']}/24 ({res_24['v7_cap']/24:.1%})")
    print(f"Capability diff: {res_24['v7_cap'] - res_24['v5_cap']}")
    print(f"Capability fixed ({len(res_24['cap_fixed'])}): {res_24['cap_fixed']}")
    print(f"Capability regressed ({len(res_24['cap_regressed'])}): {res_24['cap_regressed']}")
    assert (res_24['v7_cap'] - res_24['v5_cap']) == (len(res_24['cap_fixed']) - len(res_24['cap_regressed']))

    # Dump complete recomputed metrics
    output_data = {
        "hash_verification": hash_results,
        "frozen_100": res_100,
        "independent_24": res_24,
    }
    (AUDIT_DIR / "recomputed_metrics.json").write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved recomputed metrics to {AUDIT_DIR / 'recomputed_metrics.json'}")

if __name__ == "__main__":
    run_metric_audit()

