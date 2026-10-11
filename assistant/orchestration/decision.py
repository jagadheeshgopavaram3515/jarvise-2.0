"""Structured, model-agnostic classification contracts for JARVIS.

This module interprets model output only. It deliberately has no dependency on
the tool executor, permission layer, or provider clients.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


class DecisionValidationError(ValueError):
    """Raised when a model response does not satisfy the decision contract."""


class RequestMode(str, Enum):
    """Stable, broad characteristics of a request; multiple may apply."""

    CONVERSATIONAL = "conversational"
    MEMORY = "memory"
    RESEARCH = "research"
    ACTION = "action"


@dataclass(frozen=True)
class ContextReference:
    reference: str
    kind: str = "unspecified"
    resolved_value: str | None = None

    @property
    def is_resolved(self) -> bool:
        return self.resolved_value is not None


@dataclass(frozen=True)
class MissingInformation:
    name: str
    description: str = ""


@dataclass(frozen=True)
class EvidenceReference:
    """Short, auditable output evidence, never hidden reasoning text."""

    kind: str
    reference: str


@dataclass(frozen=True)
class Capability:
    """A capability advertised to a model, without an executable handler."""

    name: str
    description: str
    required_arguments: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionRequest:
    user_text: str
    context: Mapping[str, Any] = field(default_factory=dict)
    capabilities: tuple[Capability, ...] = ()


@dataclass(frozen=True)
class ClassifierDecision:
    """Validated semantic decision produced by a classifier model."""

    intent: str
    execution_path: str
    request_modes: frozenset[RequestMode]
    selected_capability: str | None = None
    context_references: tuple[ContextReference, ...] = ()
    missing_information: tuple[MissingInformation, ...] = ()
    confidence: float = 0.0
    needs_clarification: bool = False
    needs_escalation: bool = False
    evidence: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        if not self.intent.strip():
            raise DecisionValidationError("intent must be a non-empty string")
        if not self.execution_path.strip():
            raise DecisionValidationError("execution_path must be a non-empty string")
        if not self.request_modes:
            raise DecisionValidationError("request_modes must contain at least one mode")
        if not 0.0 <= self.confidence <= 1.0:
            raise DecisionValidationError("confidence must be between 0 and 1")
        if self.needs_clarification and not self.missing_information:
            raise DecisionValidationError(
                "needs_clarification requires at least one missing_information item"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClassifierDecision":
        """Parse untrusted model output into the validated contract."""
        if not isinstance(value, Mapping):
            raise DecisionValidationError("model output must be an object")

        intent = _required_string(value, "intent")
        execution_path = _required_string(value, "execution_path")
        modes_raw = _required_sequence(value, "request_modes")
        try:
            modes = frozenset(RequestMode(_string_item(item, "request_modes")) for item in modes_raw)
        except ValueError as exc:
            raise DecisionValidationError(f"unknown request mode: {exc}") from exc

        capability = value.get("selected_capability")
        if capability is not None and (not isinstance(capability, str) or not capability.strip()):
            raise DecisionValidationError("selected_capability must be null or a non-empty string")

        confidence = value.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise DecisionValidationError("confidence must be a number")

        clarification = value.get("needs_clarification", False)
        escalation = value.get("needs_escalation", False)
        if not isinstance(clarification, bool) or not isinstance(escalation, bool):
            raise DecisionValidationError("uncertainty flags must be booleans")

        return cls(
            intent=intent,
            execution_path=execution_path,
            request_modes=modes,
            selected_capability=capability.strip() if capability else None,
            context_references=tuple(
                _parse_context_reference(item)
                for item in _optional_sequence(value, "context_references")
            ),
            missing_information=tuple(
                _parse_missing_information(item)
                for item in _optional_sequence(value, "missing_information")
            ),
            confidence=float(confidence),
            needs_clarification=clarification,
            needs_escalation=escalation,
            evidence=tuple(
                _parse_evidence(item) for item in _optional_sequence(value, "evidence")
            ),
        )


@runtime_checkable
class DecisionModelAdapter(Protocol):
    """Replaceable boundary for Qwen, compact, or fine-tuned classifiers."""

    @property
    def adapter_id(self) -> str: ...

    def predict(self, request: DecisionRequest) -> Mapping[str, Any]: ...


class LegacyClassifierAdapter:
    """Expose an existing ``classify(text)`` classifier through the new boundary."""

    def __init__(self, classifier: Any, adapter_id: str = "legacy-qwen") -> None:
        if not callable(getattr(classifier, "classify", None)):
            raise TypeError("classifier must provide classify(user_text)")
        self.classifier = classifier
        self._adapter_id = adapter_id

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    def predict(self, request: DecisionRequest) -> Mapping[str, Any]:
        route, _latency_ms = self.classifier.classify(request.user_text)
        route = str(route).upper()
        if route not in FastClassifier.LEGACY_ROUTES:
            raise DecisionValidationError(f"legacy classifier returned unsupported route {route!r}")
        return {
            "intent": "legacy_route_selection",
            "execution_path": route,
            "request_modes": [
                RequestMode.ACTION.value if route == "TASK" else RequestMode.CONVERSATIONAL.value
            ],
            "selected_capability": None,
            "context_references": [],
            "missing_information": [],
            "confidence": 1.0,
            "needs_clarification": False,
            "needs_escalation": False,
            "evidence": [{"kind": "adapter", "reference": self.adapter_id}],
        }


class FastClassifier:
    """Validate adapter output while preserving the legacy classifier interface."""

    LEGACY_ROUTES = frozenset({"LOCAL", "GEMINI", "TASK"})

    def __init__(self, adapter: DecisionModelAdapter) -> None:
        if not isinstance(adapter, DecisionModelAdapter):
            raise TypeError("adapter must implement DecisionModelAdapter")
        self.adapter = adapter

    def decide(
        self,
        user_text: str,
        *,
        context: Mapping[str, Any] | None = None,
        capabilities: Sequence[Capability] = (),
    ) -> ClassifierDecision:
        request = DecisionRequest(
            user_text=user_text,
            context=dict(context or {}),
            capabilities=tuple(capabilities),
        )
        try:
            raw = self.adapter.predict(request)
            return ClassifierDecision.from_mapping(raw)
        except Exception as exc:
            return self._safe_fallback(exc)

    def classify(self, user_text: str) -> tuple[str, float]:
        """Compatibility interface for current dispatcher/LangGraph callers."""
        started = time.perf_counter()
        decision = self.decide(user_text)
        route = decision.execution_path.upper()
        if route not in self.LEGACY_ROUTES:
            route = "GEMINI"
        return route, (time.perf_counter() - started) * 1000.0

    @staticmethod
    def _safe_fallback(error: Exception) -> ClassifierDecision:
        return ClassifierDecision(
            intent="classification_failed",
            execution_path="GEMINI",
            request_modes=frozenset({RequestMode.CONVERSATIONAL}),
            confidence=0.0,
            needs_escalation=True,
            evidence=(
                EvidenceReference(kind="validation_error", reference=type(error).__name__),
            ),
        )


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise DecisionValidationError(f"{key} must be a non-empty string")
    return item.strip()


def _required_sequence(value: Mapping[str, Any], key: str) -> Sequence[Any]:
    if key not in value:
        raise DecisionValidationError(f"missing required field: {key}")
    return _sequence(value[key], key)


def _optional_sequence(value: Mapping[str, Any], key: str) -> Sequence[Any]:
    return _sequence(value.get(key, ()), key)


def _sequence(value: Any, key: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DecisionValidationError(f"{key} must be an array")
    return value


def _string_item(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise DecisionValidationError(f"{key} entries must be strings")
    return value


def _object(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DecisionValidationError(f"{key} entries must be objects")
    return value


def _optional_string(value: Mapping[str, Any], key: str, default: str = "") -> str:
    item = value.get(key, default)
    if not isinstance(item, str):
        raise DecisionValidationError(f"{key} must be a string")
    return item


def _parse_context_reference(value: Any) -> ContextReference:
    item = _object(value, "context_references")
    resolved = item.get("resolved_value")
    if resolved is not None and not isinstance(resolved, str):
        raise DecisionValidationError("resolved_value must be null or a string")
    return ContextReference(
        reference=_required_string(item, "reference"),
        kind=_optional_string(item, "kind", "unspecified"),
        resolved_value=resolved,
    )


def _parse_missing_information(value: Any) -> MissingInformation:
    item = _object(value, "missing_information")
    return MissingInformation(
        name=_required_string(item, "name"),
        description=_optional_string(item, "description"),
    )


def _parse_evidence(value: Any) -> EvidenceReference:
    item = _object(value, "evidence")
    return EvidenceReference(
        kind=_required_string(item, "kind"),
        reference=_required_string(item, "reference"),
    )
