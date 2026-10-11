"""Approval-gated isolated trainer. NOT RUN during dataset construction.

Rejects non-gold data before encoder loading; preserves per-example descriptors.
This is a new file; the original trainer and frozen artifact are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import psutil
import sklearn
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assistant.orchestration.compact_adapter import (
    DEFAULT_MODEL_ID,
    FrozenMultilingualEncoder,
    _pair_features,
    serialize_capability,
    serialize_capability_pair,
    serialize_request,
)
from assistant.orchestration.decision import DecisionRequest
from benchmarks.fast_classifier.evaluate import BenchmarkExample, load_dataset
from dataset_tools import load_verified_training, exact_pair_features


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(__file__).resolve().parent / "train.jsonl"
DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "future_approved_model" / "compact_heads.joblib"


def _classifier(c_value: float = 1.0) -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
        C=c_value,
    )


def _request(example: BenchmarkExample) -> DecisionRequest:
    return DecisionRequest(
        user_text=example.utterance,
        context=example.context,
        capabilities=example.capabilities,
    )


def _first_or_none(values, none="__none__"):
    return values[0] if values else none


def train(
    dataset_path: Path,
    artifact_path: Path,
    model_id: str,
    *,
    c_value: float = 1.0,
    capability_threshold: float = 0.5,
    mode_threshold: float = 0.5,
    capability_encoding: str = "independent",
) -> dict:
    dataset = load_verified_training(dataset_path)
    examples = list(dataset.examples)
    if any(getattr(example, "split") == "test" for example in examples):
        raise ValueError("sealed test examples must not be used for training")

    if artifact_path.exists() or artifact_path.with_suffix(".json").exists():
        raise ValueError("Refusing to overwrite existing model artifacts")

    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = time.perf_counter_ns()
    encoder = FrozenMultilingualEncoder(model_id=model_id, max_length=160)
    load_ms = encoder.load_ms
    if capability_encoding not in {"joint_pair", "independent"}:
        raise ValueError("capability_encoding must be joint_pair or independent")
    request_texts = [
        serialize_request(
            _request(example), include_capabilities=capability_encoding == "joint_pair"
        )
        for example in examples
    ]
    vectors = encoder.encode(request_texts, batch_size=16)

    execution = _classifier(c_value).fit(vectors, [e.expected.execution_path for e in examples])
    mode_heads = {}
    for mode in ("conversational", "memory", "research", "action"):
        labels = [int(any(item.value == mode for item in e.expected.request_modes)) for e in examples]
        mode_heads[mode] = _classifier(c_value).fit(vectors, labels)
    clarification = _classifier(c_value).fit(vectors, [int(e.expected.needs_clarification) for e in examples])
    escalation = _classifier(c_value).fit(vectors, [int(e.expected.needs_escalation) for e in examples])
    missing = _classifier(c_value).fit(
        vectors, [_first_or_none([item.name for item in e.expected.missing_information]) for e in examples]
    )
    reference_kind = _classifier(c_value).fit(
        vectors, [_first_or_none([item.kind for item in e.expected.context_references]) for e in examples]
    )
    reference_resolved = _classifier(c_value).fit(
        vectors,
        [int(bool(e.expected.context_references and e.expected.context_references[0].is_resolved)) for e in examples],
    )

    pair_labels = []
    if capability_encoding == "joint_pair":
        pair_texts = []
        for example in examples:
            request = _request(example)
            for cap in example.capabilities:
                pair_texts.append(serialize_capability_pair(request, cap))
                pair_labels.append(int(cap.name == example.expected.selected_capability))
        pair_features = encoder.encode(pair_texts, batch_size=16)
    else:
        pair_features, pair_labels, _descriptor_texts = exact_pair_features(
            examples, vectors, encoder
        )
    capability_matcher = _classifier(c_value).fit(pair_features, pair_labels)

    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    artifact = {
        "artifact_schema_version": "2.0",
        "adapter_id": "compact-mbert-frozen-v3",
        "model_id": model_id,
        "model_role": "frozen multilingual encoder; sentiment checkpoint classification head unused",
        "max_length": 160,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "training_split": "train",
        "training_examples": len(examples),
        "training_example_ids": [e.example_id for e in examples],
        "execution_path": execution,
        "request_modes": mode_heads,
        "clarification": clarification,
        "escalation": escalation,
        "missing_information": missing,
        "reference_kind": reference_kind,
        "reference_resolved": reference_resolved,
        "capability_matcher": capability_matcher,
        "capability_encoding": capability_encoding,
        "capability_threshold": capability_threshold,
        "mode_threshold": mode_threshold,
        "training_configuration": {
            "seed": 42,
            "logistic_c": c_value,
            "class_weight": "balanced",
            "capability_threshold": capability_threshold,
            "mode_threshold": mode_threshold,
            "encoder_frozen": True,
            "capability_encoding": capability_encoding,
        },
        "path_default_mode": {
            "direct_response": "conversational",
            "memory_recall": "memory",
            "research": "research",
            "single_action": "action",
            "multi_step_task": "action",
            "clarify": "action",
            "escalate": "action",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sklearn": sklearn.__version__,
            "encoder_load_ms": load_ms,
            "training_total_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": process.memory_info().rss,
        },
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path, compress=3)
    metadata = {key: value for key, value in artifact.items() if key not in {
        "execution_path", "request_modes", "clarification", "escalation",
        "missing_information", "reference_kind", "reference_resolved", "capability_matcher",
    }}
    metadata_path = artifact_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--capability-threshold", type=float, default=0.5)
    parser.add_argument("--mode-threshold", type=float, default=0.5)
    parser.add_argument(
        "--capability-encoding", choices=("joint_pair", "independent"),
        default="independent",
    )
    args = parser.parse_args()
    metadata = train(
        args.dataset, args.artifact, args.model_id,
        c_value=args.c,
        capability_threshold=args.capability_threshold,
        mode_threshold=args.mode_threshold,
        capability_encoding=args.capability_encoding,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
