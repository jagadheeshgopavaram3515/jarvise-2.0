"""Run the selected Phase 3C candidate without touching production routing."""
from __future__ import annotations

import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import psutil

from assistant.orchestration.compact_adapter import CompactMultilingualAdapter
from benchmarks.fast_classifier.evaluate import (
    BenchmarkDataset, evaluate_structured_adapter, latency_statistics, load_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "benchmarks" / "fast_classifier" / "v3" / "dataset.jsonl"
ARTIFACT = ROOT / "benchmarks" / "fast_classifier" / "models" / "v3" / "compact_heads.joblib"
OUTPUT = ROOT / "benchmark-results" / "phase3c-candidate-v3.json"


def _rotate(dataset: BenchmarkDataset, offset: int) -> BenchmarkDataset:
    i = offset % len(dataset.examples)
    return replace(dataset, examples=dataset.examples[i:] + dataset.examples[:i])


def _aggregate(reports: list[dict]) -> dict:
    latencies = [r["latency_ms"] for report in reports for r in report["records"] if r["status"] == "success"]
    components = {
        name: latency_statistics([
            r["inference_components_ms"][name]
            for report in reports for r in report["records"]
            if r["status"] == "success" and name in r.get("inference_components_ms", {})
        ])
        for name in ("preprocessing_ms", "model_inference_ms", "postprocessing_ms")
    }
    return {
        "runs": len(reports), "predictions": sum(r["counts"]["total"] for r in reports),
        "latency": latency_statistics(latencies), "components": components,
        "metrics": reports[0]["metrics"], "group_metrics": reports[0]["group_metrics"],
        "counts_per_run": reports[0]["counts"],
    }


def main() -> None:
    process = psutil.Process(); rss_before = process.memory_info().rss
    started = time.perf_counter_ns(); adapter = CompactMultilingualAdapter(ARTIFACT)
    startup_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    rss_after = process.memory_info().rss
    dev = evaluate_structured_adapter(
        adapter, load_dataset(DATASET, split="dev"),
        measurement_kind="real_phase3c_selected_candidate",
        configuration={"split_role": "selection_confirmation", "artifact": str(ARTIFACT)},
    )
    test = load_dataset(DATASET, split="test")
    runs = [evaluate_structured_adapter(
        adapter, _rotate(test, offset), measurement_kind="real_phase3c_selected_candidate",
        configuration={"split_role": "sealed_test", "run": run, "order_offset": offset},
    ) for run, offset in enumerate((0, 6, 12), start=1)]
    payload = {
        "report_schema_version": "1.0",
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
            "processor": platform.processor(), "logical_cpu_count": psutil.cpu_count(True),
            "physical_cpu_count": psutil.cpu_count(False), "total_memory_bytes": psutil.virtual_memory().total,
            "gpu_runtime": "PyTorch CPU; CUDA unavailable"},
        "dataset": {"path": str(DATASET), "train_examples": len(load_dataset(DATASET, split="train").examples),
            "development_examples": len(load_dataset(DATASET, split="dev").examples),
            "sealed_test_examples": len(test.examples), "test_lineage": "same 18 v2 sealed examples"},
        "startup": {"adapter_startup_ms": startup_ms, "encoder_reported_load_ms": adapter.encoder.load_ms,
            "rss_before_bytes": rss_before, "rss_after_bytes": rss_after, "rss_delta_bytes": rss_after-rss_before},
        "development": dev, "sealed_test": {"aggregate": _aggregate(runs), "runs": runs},
        "notes": ["No tools were executed.", "No Qwen pre-classification was used.",
            "All v3 labels remain pending independent human/native-speaker review."],
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(OUTPUT)
    print(json.dumps(payload["sealed_test"]["aggregate"]["metrics"], indent=2))
    print("latency", payload["sealed_test"]["aggregate"]["latency"])


if __name__ == "__main__": main()
