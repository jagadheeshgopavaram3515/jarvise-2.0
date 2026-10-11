"""Machine-readable evaluation scaffold for Fast Brain predictions.

Predictions are supplied by a later local-model runner; this command never
downloads models or invokes teachers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fast_brain_v1.dataset import load_jsonl, sha256_file
from .fast_brain_v1.metrics import accuracy, binary_metrics, class_counts


def evaluate(records, predictions: dict[str, dict]) -> dict:
    matched = [(record, predictions[record.id]) for record in records if record.id in predictions]
    result = {
        "records": len(records), "predictions": len(matched),
        "dataset_class_counts": class_counts(records),
        "metrics": {}, "errors": [],
    }
    if not matched:
        result["errors"].append("no matching predictions; metrics unavailable")
        return result
    result["metrics"]["intent_accuracy"] = accuracy(
        (r.intent_family for r, _ in matched), (p.get("intent_family") for _, p in matched))
    result["metrics"]["capability_accuracy"] = accuracy(
        (r.capability_family for r, _ in matched), (p.get("capability_family") for _, p in matched))
    result["metrics"]["clarification"] = binary_metrics(
        (r.clarification_required for r, _ in matched),
        (bool(p.get("clarification_required")) for _, p in matched))
    result["metrics"]["invalid_label_count"] = sum(
        1 for _, p in matched if not isinstance(p.get("intent_family"), str)
        or not isinstance(p.get("capability_family"), str))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True, help="JSON object keyed by record id")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = load_jsonl(args.dataset)
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    result = evaluate(records, predictions)
    result["dataset_sha256"] = sha256_file(args.dataset)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

