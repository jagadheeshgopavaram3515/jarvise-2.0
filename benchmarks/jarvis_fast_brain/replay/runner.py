from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .base_qwen import BaseQwenAdapter
from .dataset import file_sha256, load_jsonl
from .fast_brain_adapter import FastBrainNotImplemented
from .metrics import failure_category, summarize


def run(dataset: Path, system: str) -> dict:
    cases = load_jsonl(dataset)
    adapter = BaseQwenAdapter() if system == "base_qwen" else FastBrainNotImplemented()
    records = []
    for case in cases:
        result = adapter.run(case)
        result.update({"input": {"request": case.user_input, "context": case.context},
                       "expected": case.expected,
                       "failure_category": failure_category(result, case.expected)})
        records.append(result)
    return {"metadata": {"dataset": str(dataset), "dataset_sha256": file_sha256(dataset),
                          "system": system, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                          "prompt_version": "base-v1"},
            "summary": summarize(records), "cases": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--system", choices=("base_qwen", "fast_brain_qwen"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("benchmarks/jarvis_fast_brain/replay/results") / f"{args.system}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run(args.dataset, args.system), indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
