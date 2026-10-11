"""Offline Fast Brain training entry point.

This scaffold validates human-first data and local-only model availability.
It deliberately does not download, synthesize, call teachers, or train when
the selected checkpoint is unavailable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fast_brain_v1.dataset import load_jsonl, provenance_counts, sha256_file
from .fast_brain_v1.split import manifest, manifest_hash, validate_group_leakage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = load_jsonl(args.dataset)
    validate_group_leakage(records)
    if not Path(args.model_path).is_dir():
        result = {"status": "MODEL_UNAVAILABLE", "trained": False,
                  "dataset_sha256": sha256_file(args.dataset), "records": len(records),
                  "provenance": provenance_counts(records), "error": "local model path unavailable"}
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 2
    split = manifest(records, seed=args.seed)
    result = {"status": "VALIDATED_NOT_TRAINED", "trained": False,
              "dataset_sha256": sha256_file(args.dataset), "split": split,
              "split_sha256": manifest_hash(split), "provenance": provenance_counts(records),
              "error": "training implementation is intentionally deferred"}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

