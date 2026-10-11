"""Create deterministic human-review and adjudication artifacts.

No semantic labels are changed and no record is approved by this command.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .fast_brain_v1.dataset import analysis, load_jsonl, sha256_file
from .fast_brain_v1.review import readiness, review_artifacts_are_protected, review_priority
from .fast_brain_v1.split import validate_group_leakage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(Path(__file__).parent / "fast_brain_v1" / "dataset.jsonl"))
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "fast_brain_v1"))
    args = parser.parse_args()
    dataset = Path(args.dataset); output = Path(args.output_dir)
    records = load_jsonl(dataset, allow_text_conflicts=True)
    validate_group_leakage(records)
    queue_path = output / "review_queue.jsonl"
    adjudication_path = output / "adjudication_queue.jsonl"
    protected, reason = review_artifacts_are_protected(
        queue_path.read_text(encoding="utf-8") if queue_path.exists() else "",
        adjudication_path.read_text(encoding="utf-8") if adjudication_path.exists() else "")
    if protected:
        raise RuntimeError(f"{reason}; refusing to overwrite review artifacts")
    queue = []
    for record in records:
        priority, reasons = review_priority(record)
        if (record.language_group == "english" and record.script_group == "latin" and
                any(token in record.text.lower().split() for token in ("naa", "cheyyi", "lo", "karo", "mein"))):
            reasons.append("possible_romanized_language_metadata_mismatch")
            priority = "critical"
        canonical = {"intent_family": record.intent_family,
                     "execution_path": "clarify" if record.clarification_required else (
                         "research" if record.intent_family == "research" else (
                             "memory" if record.intent_family == "memory" else (
                                 "qwen_agent" if record.intent_family in {"local_task", "multi_step_task"} else "respond"))),
                     "capability_family": record.capability_family,
                     "context_requirements": (["active_task"] if record.context_resolvable is not None else ["none"]),
                     "clarification": {"required": record.clarification_required,
                                       "missing": list(record.missing_information)},
                     "uncertainty": "medium",
                     "escalation": {"candidate": record.intent_family == "unsupported",
                                     "reason": "unsupported_capability" if record.intent_family == "unsupported" else None}}
        queue.append({
            "record_id": record.id, "text": record.text, "context": {"context_resolvable": record.context_resolvable},
            "original_label": record.original_label,
            "canonical_label": canonical,
            "language": {"semantic_language": record.semantic_language or record.language_group,
                          "input_script": record.input_script or record.script_group,
                          "response_language": record.response_language},
            "provenance": record.provenance, "group_id": record.group_id,
            "review_status": record.review_status, "review_priority": priority,
            "review_reasons": reasons, "reviewer_decision": None, "reviewed_label": None,
            "reviewer_id": None, "reviewed_at": None, "reviewer_notes": "",
            "disagreement_fields": [], "adjudication_status": "not_required",
        })
    queue.sort(key=lambda item: ({"critical": 0, "high": 1, "normal": 2, "low": 3}[item["review_priority"]], item["record_id"]))
    queue_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in queue), encoding="utf-8")
    if not adjudication_path.exists():
        adjudication_path.write_text("", encoding="utf-8")
    gaps = {"status": "requires_human_review", "gaps": [], "priority_review_categories": [
        "multilingual_examples", "romanized_telugu", "romanized_hindi", "code_switching",
        "clarification", "unsupported", "context_dependent", "hard_negatives"]}
    if not any(r.language_group == "code_switch" for r in records): gaps["gaps"].append("code_switching")
    if not any(r.script_group == "telugu" for r in records): gaps["gaps"].append("native_telugu")
    if not any(r.script_group == "devanagari" for r in records): gaps["gaps"].append("native_hindi")
    if not any(r.noise_type == "stt_variant" for r in records): gaps["gaps"].append("stt_noise")
    (output / "dataset_gaps.json").write_text(json.dumps(gaps, indent=2), encoding="utf-8")
    report = analysis(records)
    duplicate_texts = {text: count for text, count in Counter(r.text for r in records).items() if count > 1}
    report.update({"dataset_sha256": sha256_file(dataset), "reviewed_records": 0,
                   "ambiguous_records": 0, "requires_adjudication": 0, "quarantined_records": 0,
                   "review_queue_records": len(queue), "readiness_state": readiness(records),
                   "sealed_test_readiness": "SEALED TEST NOT READY", "training_status": "TRAINING_BLOCKED",
                   "human_review_required": len(records), "generated_records": 0,
                   "provenance_preserved": all(x["provenance"] == "historical_synthetic" for x in queue),
                   "suspected_duplicate_text_count": len(duplicate_texts),
                   "suspected_duplicate_texts": duplicate_texts,
                   "group_leakage_checked": True, "model_downloaded": False, "model_trained": False,
                   "network_used": False, "production_routing_changed": False})
    (output / "review_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

