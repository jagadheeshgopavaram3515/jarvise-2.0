"""Dependency-free offline evaluation harness for JARVIS classifiers.

This module never executes a selected capability and is not imported by the
production router. Legacy classifiers are evaluated only on their real route
output; structured adapters are evaluated against the Phase 1 decision schema.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    ContextReference,
    DecisionModelAdapter,
    DecisionRequest,
    DecisionValidationError,
    MissingInformation,
    RequestMode,
)


SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0", "2.0", "3.0", "4.0"})
LEGACY_PATHS = frozenset({"LOCAL", "GEMINI", "TASK"})
STRUCTURED_METRICS = (
    "intent",
    "execution_path",
    "request_modes",
    "selected_capability",
    "context_references",
    "missing_information",
    "needs_clarification",
    "needs_escalation",
)


class DatasetValidationError(ValueError):
    """A benchmark record does not conform to the versioned dataset schema."""


@dataclass(frozen=True)
class ExpectedDecision:
    intent: str
    execution_path: str
    request_modes: frozenset[RequestMode]
    selected_capability: str | None
    context_references: tuple[ContextReference, ...]
    missing_information: tuple[MissingInformation, ...]
    needs_clarification: bool
    needs_escalation: bool


@dataclass(frozen=True)
class BenchmarkExample:
    schema_version: str
    example_id: str
    split: str
    utterance: str
    context: Mapping[str, Any]
    capabilities: tuple[Capability, ...]
    expected: ExpectedDecision
    language: str
    categories: tuple[str, ...]
    annotation_status: str
    source: str
    notes: str


@dataclass(frozen=True)
class BenchmarkDataset:
    schema_version: str
    path: str
    examples: tuple[BenchmarkExample, ...]


@dataclass(frozen=True)
class InvalidOutputCase:
    schema_version: str
    case_id: str
    output: Any
    expected_error: str


def load_dataset(
    path: str | Path,
    *,
    split: str | None = None,
    example_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> BenchmarkDataset:
    """Load and validate JSONL records, optionally selecting a stable subset."""
    dataset_path = Path(path)
    selected_ids = set(example_ids or ())
    examples: list[BenchmarkExample] = []
    seen_ids: set[str] = set()

    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetValidationError(f"could not read dataset: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        try:
            example = _parse_example(raw)
        except (DatasetValidationError, DecisionValidationError) as exc:
            raise DatasetValidationError(f"line {line_number}: {exc}") from exc
        if example.example_id in seen_ids:
            raise DatasetValidationError(f"line {line_number}: duplicate id {example.example_id!r}")
        seen_ids.add(example.example_id)
        if split is not None and example.split != split:
            continue
        if selected_ids and example.example_id not in selected_ids:
            continue
        examples.append(example)
        if limit is not None and len(examples) >= limit:
            break

    if limit is not None and limit < 1:
        raise DatasetValidationError("limit must be at least 1")
    if not examples:
        raise DatasetValidationError("dataset selection contains no examples")
    versions = {example.schema_version for example in examples}
    if len(versions) != 1:
        raise DatasetValidationError("a dataset selection must use one schema version")
    return BenchmarkDataset(next(iter(versions)), str(dataset_path), tuple(examples))


def load_invalid_output_cases(path: str | Path) -> tuple[InvalidOutputCase, ...]:
    """Load the companion corpus of intentionally malformed model outputs."""
    corpus_path = Path(path)
    try:
        lines = corpus_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetValidationError(f"could not read invalid-output corpus: {exc}") from exc
    cases = []
    seen_ids = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        item = _mapping(raw, "invalid-output case")
        version = _required_string(item, "schema_version")
        if version != SCHEMA_VERSION:
            raise DatasetValidationError(
                f"line {line_number}: unsupported schema_version {version!r}"
            )
        case_id = _required_string(item, "case_id")
        if case_id in seen_ids:
            raise DatasetValidationError(f"line {line_number}: duplicate case_id {case_id!r}")
        if "output" not in item:
            raise DatasetValidationError(f"line {line_number}: missing output")
        seen_ids.add(case_id)
        cases.append(
            InvalidOutputCase(
                schema_version=version,
                case_id=case_id,
                output=item["output"],
                expected_error=_required_string(item, "expected_error"),
            )
        )
    if not cases:
        raise DatasetValidationError("invalid-output corpus contains no cases")
    return tuple(cases)


def evaluate_legacy_classifier(
    classifier: Any,
    dataset: BenchmarkDataset,
    *,
    adapter_id: str = "legacy-qwen",
    timeout_s: float | None = None,
    measurement_kind: str = "real_model",
    configuration: Mapping[str, Any] | None = None,
    provenance_resolver: Callable[[str], str] | None = None,
    expected_path_projection: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Evaluate only fields exposed by ``classify(text)``."""
    classify = getattr(classifier, "classify", None)
    if not callable(classify):
        raise TypeError("legacy classifier must provide classify(user_text)")

    records = []
    for example in dataset.examples:
        record = _base_record(example)
        projected_expected = (
            expected_path_projection(example.expected.execution_path)
            if expected_path_projection
            else example.expected.execution_path
        )
        record["expected_execution_path"] = projected_expected
        record["inference_provenance"] = (
            provenance_resolver(example.utterance) if provenance_resolver else "unknown"
        )
        outcome, elapsed_ms = _timed_call(
            lambda text=example.utterance: classify(text), timeout_s
        )
        record["latency_ms"] = elapsed_ms
        if outcome[0] == "timeout":
            record.update(status="timeout", error=outcome[1])
        elif outcome[0] == "error":
            record.update(status="error", error=outcome[1])
        else:
            try:
                route, reported_latency = outcome[1]
                route = str(route).upper()
                if route not in LEGACY_PATHS:
                    raise ValueError(f"unsupported legacy route {route!r}")
                record.update(
                    status="success",
                    prediction={"execution_path": route},
                    reported_latency_ms=float(reported_latency),
                    comparisons=(
                        {"execution_path": route == projected_expected}
                        if projected_expected in LEGACY_PATHS
                        else {}
                    ),
                )
                if projected_expected not in LEGACY_PATHS:
                    record["not_applicable"] = ["execution_path"]
            except Exception as exc:
                record.update(status="invalid_output", error=_error_text(exc))
        records.append(record)

    unavailable = {
        field: {
            "available": False,
            "reason": "legacy classify() does not expose this field",
        }
        for field in STRUCTURED_METRICS
        if field != "execution_path"
    }
    return _build_report(
        dataset,
        records,
        adapter_id=adapter_id,
        adapter_kind="legacy",
        measurement_kind=measurement_kind,
        configuration=configuration,
        unavailable_metrics=unavailable,
    )


def evaluate_structured_adapter(
    adapter: DecisionModelAdapter,
    dataset: BenchmarkDataset,
    *,
    timeout_s: float | None = None,
    measurement_kind: str = "real_model",
    configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate raw adapter output before FastClassifier fallback masking."""
    records = []
    for example in dataset.examples:
        request = DecisionRequest(
            user_text=example.utterance,
            context=dict(example.context),
            capabilities=example.capabilities,
        )
        record = _base_record(example)
        outcome, elapsed_ms = _timed_call(
            lambda request=request: adapter.predict(request), timeout_s
        )
        record["latency_ms"] = elapsed_ms
        if outcome[0] == "timeout":
            record.update(status="timeout", error=outcome[1])
        elif outcome[0] == "error":
            record.update(status="error", error=outcome[1])
        else:
            try:
                decision = ClassifierDecision.from_mapping(outcome[1])
                comparisons = _compare_structured(decision, example.expected)
                record.update(
                    status="success",
                    prediction=_decision_to_json(decision),
                    comparisons=comparisons,
                )
                timing = getattr(adapter, "last_inference_metrics", None)
                if isinstance(timing, Mapping):
                    record["inference_components_ms"] = {
                        key: float(value)
                        for key, value in timing.items()
                        if key.endswith("_ms") and isinstance(value, (int, float))
                    }
            except Exception as exc:
                record.update(status="invalid_output", error=_error_text(exc))
        records.append(record)

    return _build_report(
        dataset,
        records,
        adapter_id=adapter.adapter_id,
        adapter_kind="structured",
        measurement_kind=measurement_kind,
        configuration=configuration,
        unavailable_metrics={},
    )


def latency_statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    """Compute deterministic nearest-rank latency statistics."""
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean_ms": sum(ordered) / len(ordered),
        "p50_ms": _nearest_rank(ordered, 0.50),
        "p95_ms": _nearest_rank(ordered, 0.95),
    }


def render_summary(report: Mapping[str, Any]) -> str:
    """Create a compact human-readable summary from a machine report."""
    counts = report["counts"]
    latency = report["latency"]["successful_predictions"]
    lines = [
        f"Dataset {report['dataset']['schema_version']} | split={report['dataset']['split']}",
        (
            f"Adapter {report['adapter']['id']} ({report['adapter']['kind']}, "
            f"{report['adapter']['measurement_kind']})"
        ),
        (
            f"Examples={counts['total']} success={counts['successful_predictions']} "
            f"invalid={counts['invalid_outputs']} errors={counts['errors']} "
            f"timeouts={counts['timeouts']}"
        ),
    ]
    if counts.get("not_applicable_execution_paths"):
        lines.append(
            "Legacy-unrepresentable expected paths: "
            f"{counts['not_applicable_execution_paths']} (excluded from route accuracy)"
        )
    for name, metric in report["metrics"].items():
        if metric.get("available"):
            if "accuracy" in metric:
                accuracy = metric["accuracy"]
                value = "n/a" if accuracy is None else f"{accuracy:.3f}"
                lines.append(
                    f"{name}: {metric['correct']}/{metric['applicable']} accuracy={value}"
                )
            elif "rate" in metric:
                rate = metric["rate"]
                value = "n/a" if rate is None else f"{rate:.3f}"
                lines.append(
                    f"{name}: {metric['triggers']}/{metric['applicable']} rate={value}"
                )
            elif "macro_f1" in metric:
                lines.append(
                    f"{name}: precision={metric['precision']:.3f} "
                    f"recall={metric['recall']:.3f} macro_f1={metric['macro_f1']:.3f}"
                )
        else:
            lines.append(f"{name}: unavailable ({metric['reason']})")
    lines.append(
        "Latency success: "
        f"mean={_fmt_ms(latency['mean_ms'])} p50={_fmt_ms(latency['p50_ms'])} "
        f"p95={_fmt_ms(latency['p95_ms'])}"
    )
    for provenance, stats in report["latency"].get("by_inference_provenance", {}).items():
        lines.append(
            f"Latency {provenance}: n={stats['count']} mean={_fmt_ms(stats['mean_ms'])} "
            f"p50={_fmt_ms(stats['p50_ms'])} p95={_fmt_ms(stats['p95_ms'])}"
        )
    lines.append(
        "Cold-start is the first invocation only; warm statistics cover subsequent "
        "successful invocations and do not prove model residency."
    )
    return "\n".join(lines)


def _parse_example(raw: Any) -> BenchmarkExample:
    if not isinstance(raw, Mapping):
        raise DatasetValidationError("record must be an object")
    version = _required_string(raw, "schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise DatasetValidationError(
            f"unsupported schema_version {version!r}; expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)!r}"
        )
    if version in {"2.0", "3.0"}:
        _required_string(raw, "group_id")
        _required_string(raw, "review_status")
        _required_string(raw, "provenance")
    expected_raw = _required_mapping(raw, "expected")
    modes = _string_list(expected_raw, "request_modes", nonempty=True)
    try:
        request_modes = frozenset(RequestMode(mode) for mode in modes)
    except ValueError as exc:
        raise DatasetValidationError(f"unknown expected request mode: {exc}") from exc

    expected = ExpectedDecision(
        intent=_required_string(expected_raw, "intent"),
        execution_path=_required_string(expected_raw, "execution_path"),
        request_modes=request_modes,
        selected_capability=_optional_nullable_string(expected_raw, "selected_capability"),
        context_references=tuple(
            _parse_expected_reference(item)
            for item in _list(expected_raw, "context_references")
        ),
        missing_information=tuple(
            _parse_expected_missing(item)
            for item in _list(expected_raw, "missing_information")
        ),
        needs_clarification=_required_bool(expected_raw, "needs_clarification"),
        needs_escalation=_required_bool(expected_raw, "needs_escalation"),
    )
    if expected.needs_clarification and not expected.missing_information:
        raise DatasetValidationError(
            "expected clarification requires missing_information"
        )

    capabilities = tuple(_parse_capability(item) for item in _list(raw, "capabilities"))
    context = raw.get("context", {})
    if not isinstance(context, Mapping):
        raise DatasetValidationError("context must be an object")
    categories = tuple(_string_list(raw, "categories", nonempty=True))
    annotation_status = _required_string(raw, "annotation_status")
    if annotation_status not in {
        "reviewed", "synthetic_candidate", "pending_review", "adjudicated", "ambiguous"
    }:
        raise DatasetValidationError("unsupported annotation_status")

    return BenchmarkExample(
        schema_version=version,
        example_id=_required_string(raw, "id"),
        split=_required_string(raw, "split"),
        utterance=_required_string(raw, "utterance"),
        context=dict(context),
        capabilities=capabilities,
        expected=expected,
        language=_required_string(raw, "language"),
        categories=categories,
        annotation_status=annotation_status,
        source=_required_string(raw, "source"),
        notes=_required_string(raw, "notes"),
    )


def _parse_capability(raw: Any) -> Capability:
    item = _mapping(raw, "capability")
    return Capability(
        name=_required_string(item, "name"),
        description=_required_string(item, "description"),
        required_arguments=tuple(_string_list(item, "required_arguments")),
    )


def _parse_expected_reference(raw: Any) -> ContextReference:
    item = _mapping(raw, "context reference")
    return ContextReference(
        reference=_required_string(item, "reference"),
        kind=_required_string(item, "kind"),
        resolved_value=_optional_nullable_string(item, "resolved_value"),
    )


def _parse_expected_missing(raw: Any) -> MissingInformation:
    item = _mapping(raw, "missing information")
    return MissingInformation(
        name=_required_string(item, "name"),
        description=_optional_string(item, "description"),
    )


def _timed_call(call: Callable[[], Any], timeout_s: float | None) -> tuple[tuple[str, Any], float]:
    started_ns = time.perf_counter_ns()
    if timeout_s is None:
        try:
            value = call()
            outcome = ("success", value)
        except TimeoutError as exc:
            outcome = ("timeout", _error_text(exc))
        except Exception as exc:
            outcome = ("error", _error_text(exc))
    else:
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="classifier-benchmark")
        future = pool.submit(call)
        try:
            outcome = ("success", future.result(timeout=timeout_s))
        except FutureTimeout:
            future.cancel()
            outcome = ("timeout", f"evaluation exceeded {timeout_s:.3f}s")
        except TimeoutError as exc:
            outcome = ("timeout", _error_text(exc))
        except Exception as exc:
            outcome = ("error", _error_text(exc))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    return outcome, (time.perf_counter_ns() - started_ns) / 1_000_000.0


def _compare_structured(
    actual: ClassifierDecision, expected: ExpectedDecision
) -> dict[str, bool]:
    return {
        "intent": actual.intent == expected.intent,
        "execution_path": actual.execution_path == expected.execution_path,
        "request_modes": actual.request_modes == expected.request_modes,
        "selected_capability": actual.selected_capability == expected.selected_capability,
        "context_references": actual.context_references == expected.context_references,
        "missing_information": actual.missing_information == expected.missing_information,
        "needs_clarification": actual.needs_clarification == expected.needs_clarification,
        "needs_escalation": actual.needs_escalation == expected.needs_escalation,
    }


def _build_report(
    dataset: BenchmarkDataset,
    records: list[dict[str, Any]],
    *,
    adapter_id: str,
    adapter_kind: str,
    measurement_kind: str,
    configuration: Mapping[str, Any] | None,
    unavailable_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    statuses = Counter(record["status"] for record in records)
    metrics: dict[str, Any] = {}
    metric_names = ("execution_path",) if adapter_kind == "legacy" else STRUCTURED_METRICS
    for name in metric_names:
        values = [
            record["comparisons"][name]
            for record in records
            if record["status"] == "success" and name in record.get("comparisons", {})
        ]
        metrics[name] = {
            "available": True,
            "applicable": len(values),
            "correct": sum(values),
            "accuracy": (sum(values) / len(values)) if values else None,
        }
    metrics.update(unavailable_metrics)
    if adapter_kind == "structured":
        total = len(records)
        valid = statuses["success"]
        fallback_triggers = total - valid
        metrics["schema_validity"] = {
            "available": True,
            "applicable": total,
            "correct": valid,
            "accuracy": valid / total if total else None,
        }
        metrics["fallback_trigger_rate"] = {
            "available": True,
            "applicable": total,
            "triggers": fallback_triggers,
            "rate": fallback_triggers / total if total else None,
            "note": "Invalid outputs, adapter exceptions, and watchdog timeouts would trigger FastClassifier fallback.",
        }
        applicable_capabilities = [
            record["comparisons"]["selected_capability"]
            for record in records
            if record["status"] == "success"
            and record.get("expected_selected_capability") is not None
        ]
        metrics["capability_selection_applicable"] = {
            "available": True,
            "applicable": len(applicable_capabilities),
            "correct": sum(applicable_capabilities),
            "accuracy": (
                sum(applicable_capabilities) / len(applicable_capabilities)
                if applicable_capabilities
                else None
            ),
        }
        metrics["clarification_binary"] = _binary_metrics(
            records, "expected_needs_clarification", "needs_clarification"
        )
        metrics["escalation_binary"] = _binary_metrics(
            records, "expected_needs_escalation", "needs_escalation"
        )
        core_fields = (
            "execution_path", "selected_capability", "intent", "request_modes",
            "needs_clarification", "needs_escalation",
        )
        strict = [
            all(record.get("comparisons", {}).get(field, False) for field in core_fields)
            for record in records if record["status"] == "success"
        ]
        metrics["overall_strict"] = {
            "available": True,
            "definition": "Exact match on path, capability, intent, modes, clarification, and escalation.",
            "applicable": len(strict), "correct": sum(strict),
            "accuracy": sum(strict) / len(strict) if strict else None,
        }

    completed_latencies = [
        record["latency_ms"] for record in records if record["status"] == "success"
    ]
    cold = completed_latencies[:1]
    warm = completed_latencies[1:]
    failures = [
        {
            "id": record["id"],
            "status": record["status"],
            "expected_execution_path": record["expected_execution_path"],
            "prediction": record.get("prediction"),
            "failed_metrics": [
                name for name, passed in record.get("comparisons", {}).items() if not passed
            ],
            "error": record.get("error"),
        }
        for record in records
        if record["status"] != "success"
        or not all(record.get("comparisons", {}).values())
    ]

    return {
        "report_schema_version": "1.0",
        "dataset": {
            "schema_version": dataset.schema_version,
            "path": dataset.path,
            "split": _common_or_mixed(example.split for example in dataset.examples),
        },
        "adapter": {
            "id": adapter_id,
            "kind": adapter_kind,
            "measurement_kind": measurement_kind,
            "configuration": dict(configuration or {}),
        },
        "counts": {
            "total": len(records),
            "successful_predictions": statuses["success"],
            "invalid_outputs": statuses["invalid_output"],
            "errors": statuses["error"],
            "timeouts": statuses["timeout"],
            "not_applicable_execution_paths": sum(
                "execution_path" in record.get("not_applicable", []) for record in records
            ),
        },
        "coverage": {
            "languages": dict(sorted(Counter(e.language for e in dataset.examples).items())),
            "categories": dict(
                sorted(Counter(category for e in dataset.examples for category in e.categories).items())
            ),
        },
        "metrics": metrics,
        "group_metrics": _group_metrics(records),
        "execution_path_confusion": _path_confusion(records),
        "latency": {
            "timer": "time.perf_counter_ns",
            "method": "nearest-rank percentiles",
            "successful_predictions": latency_statistics(completed_latencies),
            "cold_first_invocation": latency_statistics(cold),
            "warm_subsequent_invocations": latency_statistics(warm),
            "process_startup_and_model_loading_ms": None,
            "by_inference_provenance": {
                provenance: latency_statistics(
                    [
                        record["latency_ms"]
                        for record in records
                        if record["status"] == "success"
                        and record.get("inference_provenance") == provenance
                    ]
                )
                for provenance in sorted(
                    {
                        record.get("inference_provenance")
                        for record in records
                        if record.get("inference_provenance")
                    }
                )
            },
            "components": {
                component: latency_statistics(
                    [
                        record["inference_components_ms"][component]
                        for record in records
                        if record["status"] == "success"
                        and component in record.get("inference_components_ms", {})
                    ]
                )
                for component in ("preprocessing_ms", "model_inference_ms", "postprocessing_ms")
            },
            "note": (
                "Cold is the first benchmark invocation only. It is not guaranteed to "
                "include process startup or model loading; those are not measured separately. "
                "Warm does not prove cache state."
            ),
        },
        "failures": failures[:25],
        "records": records,
        "limitations": [
            "Classification only; selected capabilities are never executed.",
            "Legacy classify() hides internal Ollama failures by returning GEMINI.",
            "Legacy semantic fields are unavailable and are not inferred from routes.",
            "Seed-set group metrics have small sample sizes and are descriptive only.",
        ],
    }


def _group_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, dict[str, list[bool]]]] = {
        "language": defaultdict(lambda: defaultdict(list)),
        "category": defaultdict(lambda: defaultdict(list)),
        "script": defaultdict(lambda: defaultdict(list)),
    }
    for record in records:
        comparisons = record.get("comparisons", {})
        if not comparisons:
            continue
        keys = (("language", record["language"]), ("script", record["script_group"]))
        keys += tuple(("category", category) for category in record["categories"])
        for dimension, key in keys:
            for metric, passed in comparisons.items():
                groups[dimension][key][metric].append(bool(passed))
    return {
        dimension: {
            key: {"examples": len(next(iter(metrics.values()))), **{
                f"{metric}_accuracy": sum(values) / len(values)
                for metric, values in sorted(metrics.items()) if values
            }} for key, metrics in sorted(items.items())
        }
        for dimension, items in groups.items()
    }


def _path_confusion(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        if record["status"] != "success":
            continue
        predicted = (record.get("prediction") or {}).get("execution_path")
        expected = record.get("expected_execution_path")
        if expected is not None and predicted is not None:
            matrix[str(expected)][str(predicted)] += 1
    return {gold: dict(sorted(counts.items())) for gold, counts in sorted(matrix.items())}


def _base_record(example: BenchmarkExample) -> dict[str, Any]:
    return {
        "id": example.example_id,
        "split": example.split,
        "language": example.language,
        "categories": list(example.categories),
        "expected_execution_path": example.expected.execution_path,
        "expected_selected_capability": example.expected.selected_capability,
        "expected_needs_clarification": example.expected.needs_clarification,
        "expected_needs_escalation": example.expected.needs_escalation,
        "script_group": _script_group(example),
    }


def _binary_metrics(
    records: Sequence[Mapping[str, Any]], expected_key: str, prediction_key: str
) -> dict[str, Any]:
    pairs = []
    for record in records:
        if record["status"] != "success":
            continue
        predicted = bool((record.get("prediction") or {}).get(prediction_key))
        pairs.append((bool(record[expected_key]), predicted))
    tp = sum(expected and predicted for expected, predicted in pairs)
    fp = sum(not expected and predicted for expected, predicted in pairs)
    fn = sum(expected and not predicted for expected, predicted in pairs)
    tn = sum(not expected and not predicted for expected, predicted in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    positive_f1 = _f1(precision, recall)
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    negative_f1 = _f1(negative_precision, negative_recall)
    return {
        "available": True,
        "applicable": len(pairs),
        "positive_examples": sum(expected for expected, _ in pairs),
        "precision": precision,
        "recall": recall,
        "macro_f1": (positive_f1 + negative_f1) / 2.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _script_group(example: BenchmarkExample) -> str:
    categories = set(example.categories)
    if "code_switching" in categories or example.language.startswith("mixed-"):
        return "code_switched"
    if "romanized" in categories or example.language.endswith("-Latn"):
        return "romanized"
    if "native_script" in categories or example.language in {"te", "hi", "ta"}:
        return "native_script"
    return "english_latin"


def _decision_to_json(decision: ClassifierDecision) -> dict[str, Any]:
    return {
        "intent": decision.intent,
        "execution_path": decision.execution_path,
        "request_modes": sorted(mode.value for mode in decision.request_modes),
        "selected_capability": decision.selected_capability,
        "context_references": [
            {
                "reference": ref.reference,
                "kind": ref.kind,
                "resolved_value": ref.resolved_value,
            }
            for ref in decision.context_references
        ],
        "missing_information": [
            {"name": item.name, "description": item.description}
            for item in decision.missing_information
        ],
        "confidence": decision.confidence,
        "needs_clarification": decision.needs_clarification,
        "needs_escalation": decision.needs_escalation,
        "evidence": [
            {"kind": item.kind, "reference": item.reference}
            for item in decision.evidence
        ],
    }


def _nearest_rank(ordered: Sequence[float], percentile: float) -> float:
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _common_or_mixed(values: Iterable[str]) -> str:
    unique = set(values)
    return next(iter(unique)) if len(unique) == 1 else "mixed"


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}ms"


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"{label} must be an object")
    return value


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in value:
        raise DatasetValidationError(f"missing required field {key!r}")
    return _mapping(value[key], key)


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise DatasetValidationError(f"{key} must be a non-empty string")
    return item.strip()


def _optional_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key, "")
    if not isinstance(item, str):
        raise DatasetValidationError(f"{key} must be a string")
    return item


def _optional_nullable_string(value: Mapping[str, Any], key: str) -> str | None:
    item = value.get(key)
    if item is not None and (not isinstance(item, str) or not item.strip()):
        raise DatasetValidationError(f"{key} must be null or a non-empty string")
    return item.strip() if item else None


def _required_bool(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise DatasetValidationError(f"{key} must be a boolean")
    return item


def _list(value: Mapping[str, Any], key: str) -> list[Any]:
    item = value.get(key)
    if not isinstance(item, list):
        raise DatasetValidationError(f"{key} must be an array")
    return item


def _string_list(
    value: Mapping[str, Any], key: str, *, nonempty: bool = False
) -> list[str]:
    items = _list(value, key)
    if nonempty and not items:
        raise DatasetValidationError(f"{key} must not be empty")
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise DatasetValidationError(f"{key} entries must be non-empty strings")
    return [item.strip() for item in items]


def _parse_ids(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="benchmarks/fast_classifier/v1/seed.jsonl",
        help="Versioned JSONL dataset path",
    )
    parser.add_argument("--adapter", choices=("legacy-qwen",), default="legacy-qwen")
    parser.add_argument("--split", help="Evaluate only this split")
    parser.add_argument("--ids", help="Comma-separated stable example IDs")
    parser.add_argument("--limit", type=int, help="Evaluate the first N selected records")
    parser.add_argument("--timeout", type=float, default=None, help="Per-example watchdog seconds")
    parser.add_argument("--output", help="Write the machine-readable JSON report here")
    args = parser.parse_args(argv)

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        example_ids=_parse_ids(args.ids),
        limit=args.limit,
    )

    from assistant import config
    from assistant.orchestration.router import QwenClassifier, fast_classify

    classifier = QwenClassifier()
    report = evaluate_legacy_classifier(
        classifier,
        dataset,
        adapter_id=f"legacy-qwen:{classifier.model}",
        timeout_s=args.timeout,
        measurement_kind="real_model_or_deterministic_fast_path",
        configuration={
            "model": classifier.model,
            "base_url": classifier.base_url,
            "classifier_timeout_s": classifier.timeout_s,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
        },
        provenance_resolver=lambda text: (
            "deterministic_fast_path" if fast_classify(text) is not None else "local_model"
        ),
    )
    print(render_summary(report))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
