"""Run repeated offline candidate and matched legacy evaluations on v2 test data."""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import psutil

from assistant.orchestration.compact_adapter import CompactMultilingualAdapter
from assistant.orchestration.router import QwenClassifier, fast_classify
from benchmarks.fast_classifier.evaluate import (
    BenchmarkDataset,
    evaluate_legacy_classifier,
    evaluate_structured_adapter,
    latency_statistics,
    load_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "benchmarks" / "fast_classifier" / "v2" / "dataset.jsonl"
ARTIFACT = ROOT / "benchmarks" / "fast_classifier" / "models" / "v2" / "compact_heads.joblib"
OUTPUT_DIR = ROOT / "benchmark-results"


def _rotate(dataset: BenchmarkDataset, offset: int) -> BenchmarkDataset:
    examples = dataset.examples
    index = offset % len(examples)
    return replace(dataset, examples=examples[index:] + examples[:index])


def _legacy_projection(path: str) -> str | None:
    if path == "direct_response":
        return "LOCAL"
    if path == "memory_recall":
        return "GEMINI"
    if path in {"research", "single_action", "multi_step_task"}:
        return "TASK"
    return None


def _aggregate(reports: list[dict]) -> dict:
    latencies = [
        record["latency_ms"]
        for report in reports
        for record in report["records"]
        if record["status"] == "success"
    ]
    components = {}
    for name in ("preprocessing_ms", "model_inference_ms", "postprocessing_ms"):
        values = [
            record["inference_components_ms"][name]
            for report in reports
            for record in report["records"]
            if name in record.get("inference_components_ms", {})
        ]
        components[name] = latency_statistics(values)
    return {
        "runs": len(reports),
        "predictions": sum(report["counts"]["total"] for report in reports),
        "latency": latency_statistics(latencies),
        "components": components,
        "metrics": reports[0]["metrics"],
        "group_metrics": reports[0]["group_metrics"],
        "counts_per_run": reports[0]["counts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("candidate", "legacy"), required=True)
    args = parser.parse_args()
    process = psutil.Process()
    test = load_dataset(DATASET, split="test")
    payload = {
        "report_schema_version": "1.0",
        "dataset": {
            "path": str(DATASET),
            "schema_version": test.schema_version,
            "sealed_test_examples": len(test.examples),
            "development_examples": len(load_dataset(DATASET, split="dev").examples),
            "labels_authoritative": False,
            "label_status": "v2 adjudications pending human review",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_memory_bytes": psutil.virtual_memory().total,
            "gpu_runtime": "PyTorch 2.9.0+cpu; CUDA unavailable",
        },
        "notes": [
            "Candidate and legacy are run in separate fresh processes with the same test examples and order rotations.",
            "Candidate test labels are pending human review; metrics are provisional.",
            "The frozen encoder checkpoint was already cached and no network download occurred.",
            "No capability was executed.",
        ],
    }
    if args.mode == "candidate":
        dev = load_dataset(DATASET, split="dev")
        rss_before = process.memory_info().rss
        startup_started = time.perf_counter_ns()
        candidate = CompactMultilingualAdapter(ARTIFACT)
        startup_ms = (time.perf_counter_ns() - startup_started) / 1_000_000.0
        rss_after = process.memory_info().rss
        dev_report = evaluate_structured_adapter(
            candidate,
            dev,
            measurement_kind="real_frozen_encoder_candidate",
            configuration={"artifact": str(ARTIFACT), "split_role": "model_selection"},
        )
        candidate_runs = [
            evaluate_structured_adapter(
                candidate,
                _rotate(test, offset),
                measurement_kind="real_frozen_encoder_candidate",
                configuration={"artifact": str(ARTIFACT), "run": run + 1, "order_offset": offset},
            )
            for run, offset in enumerate((0, 6, 12))
        ]
        payload.update({
            "candidate_startup": {
                "adapter_startup_ms": startup_ms,
                "encoder_reported_load_ms": candidate.encoder.load_ms,
                "rss_before_bytes": rss_before,
                "rss_after_bytes": rss_after,
                "rss_delta_bytes": rss_after - rss_before,
            },
            "development": dev_report,
            "candidate_test": {"aggregate": _aggregate(candidate_runs), "runs": candidate_runs},
        })
        output = OUTPUT_DIR / "phase3b-candidate-v2.json"
    else:
        legacy = QwenClassifier()
        legacy_runs = [
            evaluate_legacy_classifier(
                legacy,
                _rotate(test, offset),
                adapter_id=f"legacy-qwen:{legacy.model}",
                measurement_kind="real_model_or_deterministic_fast_path",
                configuration={"model": legacy.model, "run": run + 1, "order_offset": offset},
                provenance_resolver=lambda text: (
                    "deterministic_fast_path" if fast_classify(text) is not None else "local_model"
                ),
                expected_path_projection=_legacy_projection,
            )
            for run, offset in enumerate((0, 6, 12))
        ]
        payload["legacy_test"] = {"aggregate": _aggregate(legacy_runs), "runs": legacy_runs}
        output = OUTPUT_DIR / "phase3b-legacy-v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
