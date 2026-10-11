import json
from pathlib import Path

audit_dir = Path("reports/jarvis_indic_ranking_forensics_20261010_213500")
out_dir = Path("reports/jarvis_v7_independent_verification_20261011_114200")

with open(audit_dir / "label_quality_audit.json", "r", encoding="utf-8") as f:
    label_audit = json.load(f)

frozen100_raw = [json.loads(line) for line in Path("reports/classifier_v4_frozen_100_eval_20261010_180014/unseen_evaluation_corpus_100.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
corpus24_raw = [json.loads(line) for line in Path("reports/classifier_v4_experiment_20261010_175140/unseen_evaluation_corpus_v4.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

v5_pred_100 = {p["id"]: p for p in json.loads((audit_dir / "v5_predictions_100.json").read_text(encoding="utf-8"))}
v7_pred_100 = {p["id"]: p for p in json.loads((audit_dir / "v7_predictions_100.json").read_text(encoding="utf-8"))}

v5_pred_24 = {p["id"]: p for p in json.loads((audit_dir / "v5_predictions_24.json").read_text(encoding="utf-8"))}
v7_pred_24 = {p["id"]: p for p in json.loads((audit_dir / "v7_predictions_24.json").read_text(encoding="utf-8"))}

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

def check_match(pred_dict, exp_dict):
    m = {}
    for f in FIELDS:
        pv = pred_dict.get(f)
        ev = exp_dict.get(f)
        if f == "request_modes":
            m[f] = sorted(pv or []) == sorted(ev or [])
        else:
            m[f] = (pv == ev)
    m["full_exact"] = all(m[f] for f in FIELDS)
    m["capability"] = (pred_dict.get("selected_capability") == exp_dict.get("selected_capability"))
    return m

audit_analysis = []

for item in label_audit:
    rid = item["id"]
    rec = next((x for x in frozen100_raw if x["id"] == rid), None)
    is_24 = False
    if not rec:
        rec = next((x for x in corpus24_raw if x["id"] == rid), None)
        is_24 = True

    if not rec:
        continue

    v5_item = v5_pred_24.get(rid) if is_24 else v5_pred_100.get(rid)
    v7_item = v7_pred_24.get(rid) if is_24 else v7_pred_100.get(rid)

    v5_pred = v5_item["predicted"]
    v7_pred = v7_item["predicted"]
    orig_exp = rec["expected"]

    audited_exp = dict(orig_exp)
    if item["status"] == "confirmed_incorrect":
        audited_exp.update(item["proposed_correction"])

    v5_orig = check_match(v5_pred, orig_exp)
    v5_audit = check_match(v5_pred, audited_exp)
    v7_orig = check_match(v7_pred, orig_exp)
    v7_audit = check_match(v7_pred, audited_exp)

    entry = {
        "id": rid,
        "corpus": "unseen_24" if is_24 else "frozen_100",
        "status": item["status"],
        "reason": item["reason"],
        "v5_orig_exact": v5_orig["full_exact"],
        "v5_audit_exact": v5_audit["full_exact"],
        "v7_orig_exact": v7_orig["full_exact"],
        "v7_audit_exact": v7_audit["full_exact"],
        "v7_mismatches_orig": [f for f in FIELDS if not v7_orig[f]],
        "v7_mismatches_audit": [f for f in FIELDS if not v7_audit[f]],
        "model_pred_context_refs": v7_pred.get("context_references"),
        "orig_exp_context_refs": orig_exp.get("context_references"),
        "audited_exp_context_refs": audited_exp.get("context_references"),
    }
    audit_analysis.append(entry)

with open(out_dir / "label_defect_forensic_analysis.json", "w", encoding="utf-8") as f:
    json.dump(audit_analysis, f, indent=2, ensure_ascii=False)

# Let's count full dataset exact_reviewed
def compute_dataset_exact(pred_map, corpus, use_audit=False):
    exact_count = 0
    flipped_ids = []
    for r in corpus:
        rid = r["id"]
        pred = pred_map[rid]["predicted"]
        exp = dict(r["expected"])
        orig_match = check_match(pred, exp)["full_exact"]
        if use_audit:
            audit_item = next((it for it in label_audit if it["id"] == rid and it["status"] == "confirmed_incorrect"), None)
            if audit_item:
                exp.update(audit_item["proposed_correction"])
        audit_match = check_match(pred, exp)["full_exact"]
        if audit_match:
            exact_count += 1
        if use_audit and audit_match and not orig_match:
            flipped_ids.append(rid)
    return exact_count, flipped_ids

v5_f100_orig, _ = compute_dataset_exact(v5_pred_100, frozen100_raw, False)
v5_f100_audit, v5_f100_flipped = compute_dataset_exact(v5_pred_100, frozen100_raw, True)

v7_f100_orig, _ = compute_dataset_exact(v7_pred_100, frozen100_raw, False)
v7_f100_audit, v7_f100_flipped = compute_dataset_exact(v7_pred_100, frozen100_raw, True)

v5_c24_orig, _ = compute_dataset_exact(v5_pred_24, corpus24_raw, False)
v5_c24_audit, v5_c24_flipped = compute_dataset_exact(v5_pred_24, corpus24_raw, True)

v7_c24_orig, _ = compute_dataset_exact(v7_pred_24, corpus24_raw, False)
v7_c24_audit, v7_c24_flipped = compute_dataset_exact(v7_pred_24, corpus24_raw, True)

summary = {
    "frozen100": {
        "v5_original": v5_f100_orig,
        "v5_audited": v5_f100_audit,
        "v5_flipped_records": v5_f100_flipped,
        "v7_original": v7_f100_orig,
        "v7_audited": v7_f100_audit,
        "v7_flipped_records": v7_f100_flipped,
    },
    "corpus24": {
        "v5_original": v5_c24_orig,
        "v5_audited": v5_c24_audit,
        "v5_flipped_records": v5_c24_flipped,
        "v7_original": v7_c24_orig,
        "v7_audited": v7_c24_audit,
        "v7_flipped_records": v7_c24_flipped,
    }
}

with open(out_dir / "audited_label_score_recomputation.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("SUMMARY OF AUDITED SCORES:")
print(json.dumps(summary, indent=2))

with open(out_dir / "frozen100_audited_details.txt", "w", encoding="utf-8") as f:
    f.write("DETAILS OF ALL 12 FROZEN-100 AUDITED RECORDS:\n")
    for item in audit_analysis:
        if item["corpus"] == "frozen_100":
            f.write(f"ID: {item['id']} | Status: {item['status']} | OrigExact: {item['v7_orig_exact']} -> AuditExact: {item['v7_audit_exact']}\n")
            if not item['v7_audit_exact']:
                f.write(f"   Remaining mismatches: {item['v7_mismatches_audit']}\n")
                f.write(f"   Model context_refs: {item['model_pred_context_refs']}\n")
                f.write(f"   Audit context_refs: {item['audited_exp_context_refs']}\n")
            else:
                f.write("   => FLIPPED TO STRICT EXACT MATCH!\n")
print("Wrote frozen100_audited_details.txt successfully.")

