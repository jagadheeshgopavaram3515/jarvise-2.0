"""Run isolated Phase 4 model evaluations against the same sealed v4 split."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import psutil

from benchmarks.fast_classifier.evaluate import (
    evaluate_legacy_classifier,
    evaluate_structured_adapter,
    load_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "benchmarks" / "fast_classifier" / "v4" / "dataset.jsonl"
MODEL_PATH = ROOT / "benchmarks" / "fast_classifier" / "models" / "v3" / "compact_heads.joblib"
RESULTS = ROOT / "benchmark-results"


def environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "processor": platform.processor(), "physical_cpu_count": psutil.cpu_count(False),
        "logical_cpu_count": psutil.cpu_count(True),
        "total_memory_bytes": psutil.virtual_memory().total,
        "declared_laptop": "Intel i5-9300H, 8 GB RAM, GTX 1650 4 GB",
    }


def run_bert() -> Path:
    from assistant.orchestration.compact_adapter import CompactMultilingualAdapter

    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = time.perf_counter_ns()
    adapter = CompactMultilingualAdapter(MODEL_PATH)
    startup_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    rss_after = process.memory_info().rss
    report = evaluate_structured_adapter(
        adapter, load_dataset(DATASET_PATH, split="test"),
        measurement_kind="real_phase3c_bert_on_v4_sealed_test",
        configuration={"artifact": str(MODEL_PATH), "device": "cpu", "training_data": "v3 train only"},
    )
    report["environment"] = environment()
    report["startup"] = {"adapter_and_model_load_ms": startup_ms, "encoder_reported_load_ms": adapter.encoder.load_ms,
                         "rss_before_bytes": rss_before, "rss_after_bytes": rss_after,
                         "rss_delta_bytes": rss_after - rss_before, "device": "cpu",
                         "torch_cuda_available": bool(adapter.encoder.torch.cuda.is_available())}
    output = RESULTS / "phase4-bert-v4.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def run_laya() -> Path:
    from assistant.orchestration.laya_adapter import LayaOfflineAdapter

    started = time.perf_counter_ns()
    with LayaOfflineAdapter() as adapter:
        startup_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        report = evaluate_structured_adapter(
            adapter, load_dataset(DATASET_PATH, split="test"), timeout_s=30.0,
            measurement_kind="real_laya_multilingual_on_v4_sealed_test",
            configuration={"model": adapter.metadata.get("model_id"), "subfolder": adapter.metadata.get("subfolder"),
                           "package_version": adapter.metadata.get("package_version"), "device": adapter.metadata.get("device")},
        )
        report["startup"] = {
            "worker_and_model_load_ms": startup_ms,
            "worker_reported_model_load_ms": adapter.metadata.get("model_load_ms"),
            "rss_before_bytes": adapter.metadata.get("rss_before_bytes"),
            "rss_after_bytes": adapter.metadata.get("rss_after_bytes"),
            "rss_delta_bytes": (
                adapter.metadata["rss_after_bytes"] - adapter.metadata["rss_before_bytes"]
                if adapter.metadata.get("rss_before_bytes") is not None else None
            ),
            "device": adapter.metadata.get("device"), "torch_cuda_available": adapter.metadata.get("torch_cuda_available"),
        }
    report["environment"] = environment()
    output = RESULTS / "phase4-laya-v4.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def legacy_projection(path: str) -> str | None:
    return {
        "direct_response": "LOCAL", "memory_recall": "GEMINI", "research": "TASK",
        "single_action": "TASK", "multi_step_task": "TASK",
        "clarify": None, "escalate": None,
    }[path]


def run_qwen() -> Path:
    from assistant.orchestration.router import QwenClassifier, fast_classify

    classifier = QwenClassifier()
    report = evaluate_legacy_classifier(
        classifier, load_dataset(DATASET_PATH, split="test"), adapter_id="legacy-qwen-production-interface",
        timeout_s=classifier.timeout_s + 1.0, measurement_kind="real_legacy_qwen_on_v4_sealed_test",
        configuration={"model": classifier.model, "base_url": classifier.base_url, "timeout_s": classifier.timeout_s,
                       "interface": "classify(text)->LOCAL|GEMINI|TASK; structured fields unavailable"},
        provenance_resolver=lambda text: "deterministic_fast_path" if fast_classify(text) else "ollama_qwen",
        expected_path_projection=legacy_projection,
    )
    report["environment"] = environment()
    output = RESULTS / "phase4-qwen-v4.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def run_comparison() -> Path:
    paths = {name: RESULTS / f"phase4-{name}-v4.json" for name in ("bert", "laya", "qwen")}
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    payload = {
        "report_schema_version": "1.0", "dataset": str(DATASET_PATH),
        "review_status": "pending_review; no winner may be declared",
        "comparability": {
            "same_test_ids": len({tuple(record["id"] for record in report["records"]) for report in reports.values()}) == 1,
            "same_transcripts": True,
            "structured_fields": "BERT and Laya expose the contract; legacy Qwen exposes only a three-route tuple.",
        },
        "models": {
            name: {"adapter": report["adapter"], "counts": report["counts"], "metrics": report["metrics"],
                   "group_metrics": report["group_metrics"], "execution_path_confusion": report["execution_path_confusion"],
                   "latency": report["latency"], "startup": report.get("startup")}
            for name, report in reports.items()
        },
    }
    output = RESULTS / "phase4-comparison-v4.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("bert", "laya", "qwen", "comparison", "all"))
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    runners = {"bert": run_bert, "laya": run_laya, "qwen": run_qwen}
    if args.model == "all":
        for runner in runners.values():
            print(runner())
        print(run_comparison())
    elif args.model == "comparison":
        print(run_comparison())
    else:
        print(runners[args.model]())


if __name__ == "__main__":
    main()
