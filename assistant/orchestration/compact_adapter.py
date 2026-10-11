"""Offline-only frozen multilingual encoder adapter for FastClassifier experiments.

This module is intentionally not imported by the production router. It performs
one batched encoder forward pass per decision and never executes capabilities.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from assistant.orchestration.decision import (
    ClassifierDecision,
    DecisionRequest,
)


DEFAULT_MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
ACTION_PATHS = frozenset({"research", "single_action", "multi_step_task"})


def serialize_request(request: DecisionRequest, *, include_capabilities: bool = False) -> str:
    """Create a bounded, deterministic encoder input without executable objects."""
    context = json.dumps(request.context, ensure_ascii=False, sort_keys=True, default=str)
    if not include_capabilities:
        return f"User request: {request.user_text}\nContext: {context[:2000]}"
    capabilities = " | ".join(
        serialize_capability(cap.name, cap.description, cap.required_arguments)
        for cap in request.capabilities
    ) or "none"
    return (
        f"[REQUEST] {request.user_text}\n"
        f"[CONTEXT] {context[:1600]}\n"
        f"[AVAILABLE CAPABILITIES] {capabilities[:1600]}"
    )


def serialize_capability(name: str, description: str, required: Sequence[str]) -> str:
    return f"Capability {name}: {description} Required arguments: {', '.join(required) or 'none'}"


def serialize_capability_pair(request: DecisionRequest, capability: Any) -> str:
    """Represent semantic request/capability compatibility for one encoder batch item."""
    context = json.dumps(request.context, ensure_ascii=False, sort_keys=True, default=str)
    return (
        f"[REQUEST] {request.user_text}\n"
        f"[CONTEXT] {context[:1000]}\n"
        f"[CANDIDATE] {serialize_capability(capability.name, capability.description, capability.required_arguments)}"
    )


class FrozenMultilingualEncoder:
    """Local-files-only Hugging Face encoder with mean pooling."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, max_length: int = 160) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModel, AutoTokenizer

        started = time.perf_counter_ns()
        self.torch = torch
        self.model_id = model_id
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_id, local_files_only=True)
        self.model.eval()
        self.load_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        self.forward_calls = 0
        self.last_timings: dict[str, float] = {}

    def encode(self, texts: Sequence[str], *, batch_size: int | None = None) -> np.ndarray:
        """Encode texts; each internal batch is one model forward call."""
        all_vectors = []
        size = batch_size or len(texts)
        for offset in range(0, len(texts), size):
            batch = list(texts[offset : offset + size])
            token_started = time.perf_counter_ns()
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            token_ms = (time.perf_counter_ns() - token_started) / 1_000_000.0
            model_started = time.perf_counter_ns()
            with self.torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            model_ms = (time.perf_counter_ns() - model_started) / 1_000_000.0
            self.forward_calls += 1
            self.last_timings = {"preprocessing_ms": token_ms, "model_inference_ms": model_ms}
            all_vectors.append(pooled.cpu().numpy())
        return np.concatenate(all_vectors, axis=0)


class CompactMultilingualAdapter:
    """Frozen multilingual encoder plus trained lightweight decision heads."""

    def __init__(
        self,
        artifact_path: str | Path,
        *,
        encoder: FrozenMultilingualEncoder | None = None,
    ) -> None:
        self.artifact_path = str(artifact_path)
        self.artifact = joblib.load(self.artifact_path)
        self.encoder = encoder or FrozenMultilingualEncoder(
            self.artifact["model_id"], self.artifact["max_length"]
        )
        self._adapter_id = self.artifact["adapter_id"]
        self.inference_count = 0
        self.last_inference_metrics: dict[str, float] = {}

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    def predict(self, request: DecisionRequest) -> Mapping[str, Any]:
        """Run exactly one batched encoder inference and return validated data."""
        started = time.perf_counter_ns()
        before_calls = self.encoder.forward_calls
        pair_encoding = self.artifact.get("capability_encoding", "independent") == "joint_pair"
        joint_encoding = self.artifact.get("capability_encoding") == "joint_pair"
        texts = [serialize_request(request, include_capabilities=joint_encoding)] + [
            (
                serialize_capability_pair(request, cap)
                if pair_encoding
                else serialize_capability(cap.name, cap.description, cap.required_arguments)
            )
            for cap in request.capabilities
        ]
        vectors = self.encoder.encode(texts)
        if self.encoder.forward_calls - before_calls != 1:
            raise RuntimeError("decision inference must use exactly one encoder forward pass")
        self.inference_count += 1

        request_vector = vectors[0:1]
        path_model = self.artifact["execution_path"]
        path = str(path_model.predict(request_vector)[0])
        path_probabilities = path_model.predict_proba(request_vector)[0]
        confidence = float(np.max(path_probabilities))

        modes = [
            mode
            for mode, model in self.artifact["request_modes"].items()
            if _positive_probability(model, request_vector) >= self.artifact["mode_threshold"]
        ]
        if not modes:
            modes = [self.artifact["path_default_mode"].get(path, "conversational")]

        clarification_probability = _positive_probability(
            self.artifact["clarification"], request_vector
        )
        escalation_probability = _positive_probability(
            self.artifact["escalation"], request_vector
        )
        needs_clarification = path == "clarify" or clarification_probability >= 0.5
        needs_escalation = path == "escalate" or escalation_probability >= 0.5
        if needs_clarification and needs_escalation:
            if clarification_probability >= escalation_probability:
                needs_escalation = False
                path = "clarify"
            else:
                needs_clarification = False
                path = "escalate"
        elif needs_clarification:
            path = "clarify"
        elif needs_escalation:
            path = "escalate"

        selected_capability, capability_score = self._select_capability(
            request, request_vector[0], vectors[1:]
        )
        if path not in ACTION_PATHS:
            selected_capability = None
        if path in ACTION_PATHS and selected_capability is None:
            if not request.capabilities:
                path = "escalate"
                needs_escalation = True
                needs_clarification = False
            else:
                path = "clarify"
                needs_clarification = True
                needs_escalation = False

        missing_information = []
        if needs_clarification:
            missing_name = str(self.artifact["missing_information"].predict(request_vector)[0])
            if missing_name == "__none__":
                missing_name = "required_information"
            missing_information = [
                {"name": missing_name, "description": "Required information is unresolved"}
            ]

        references = self._predict_references(request, request_vector)
        intent = _compose_intent(path, selected_capability)
        evidence = [{"kind": "input_span", "reference": request.user_text[:160]}]
        if selected_capability:
            evidence.append({"kind": "capability", "reference": selected_capability})

        raw = {
            "intent": intent,
            "execution_path": path,
            "request_modes": sorted(set(modes)),
            "selected_capability": selected_capability,
            "context_references": references,
            "missing_information": missing_information,
            "confidence": max(0.0, min(1.0, confidence)),
            "needs_clarification": needs_clarification,
            "needs_escalation": needs_escalation,
            "evidence": evidence,
        }
        decision = ClassifierDecision.from_mapping(raw)
        total_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        self.last_inference_metrics = {
            **self.encoder.last_timings,
            "postprocessing_ms": max(
                0.0,
                total_ms
                - self.encoder.last_timings.get("preprocessing_ms", 0.0)
                - self.encoder.last_timings.get("model_inference_ms", 0.0),
            ),
            "capability_score": capability_score,
        }
        return _decision_to_mapping(decision)

    def _select_capability(
        self,
        request: DecisionRequest,
        request_vector: np.ndarray,
        capability_vectors: np.ndarray,
    ) -> tuple[str | None, float]:
        if not request.capabilities:
            return None, 0.0
        features = (
            capability_vectors
            if self.artifact.get("capability_encoding") == "joint_pair"
            else np.stack(
                [_pair_features(request_vector, vector) for vector in capability_vectors]
            )
        )
        model = self.artifact["capability_matcher"]
        scores = model.predict_proba(features)[:, list(model.classes_).index(1)]
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score < float(self.artifact["capability_threshold"]):
            return None, score
        return request.capabilities[best].name, score

    def _predict_references(
        self, request: DecisionRequest, request_vector: np.ndarray
    ) -> list[dict[str, Any]]:
        kind = str(self.artifact["reference_kind"].predict(request_vector)[0])
        if kind == "__none__":
            return []
        resolved_probability = _positive_probability(
            self.artifact["reference_resolved"], request_vector
        )
        resolved = _first_context_value(request.context) if resolved_probability >= 0.5 else None
        return [
            {
                "reference": request.user_text,
                "kind": kind,
                "resolved_value": resolved,
            }
        ]


def _pair_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left, right, np.abs(left - right), left * right])


def _positive_probability(model: Any, vectors: np.ndarray) -> float:
    probabilities = model.predict_proba(vectors)[0]
    classes = list(model.classes_)
    return float(probabilities[classes.index(1)]) if 1 in classes else float(classes[0] == 1)


def _compose_intent(path: str, capability: str | None) -> str:
    if path == "direct_response":
        return "respond"
    if path == "memory_recall":
        return "recall"
    if path in {"research", "multi_step_task"}:
        return "research"
    if path == "single_action" and capability:
        return f"use:{capability}"
    if path == "clarify":
        return "clarify_request"
    return "unsupported_request"


def _first_context_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_context_value(item)
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            found = _first_context_value(item)
            if found:
                return found
    elif isinstance(value, (str, int, float, bool)):
        return str(value)
    return None


def _decision_to_mapping(decision: ClassifierDecision) -> dict[str, Any]:
    return {
        "intent": decision.intent,
        "execution_path": decision.execution_path,
        "request_modes": [mode.value for mode in decision.request_modes],
        "selected_capability": decision.selected_capability,
        "context_references": [
            {"reference": item.reference, "kind": item.kind, "resolved_value": item.resolved_value}
            for item in decision.context_references
        ],
        "missing_information": [
            {"name": item.name, "description": item.description}
            for item in decision.missing_information
        ],
        "confidence": decision.confidence,
        "needs_clarification": decision.needs_clarification,
        "needs_escalation": decision.needs_escalation,
        "evidence": [
            {"kind": item.kind, "reference": item.reference} for item in decision.evidence
        ],
    }


__all__ = [
    "CompactMultilingualAdapter",
    "DEFAULT_MODEL_ID",
    "FrozenMultilingualEncoder",
    "serialize_capability",
    "serialize_capability_pair",
    "serialize_request",
]
