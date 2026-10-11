from __future__ import annotations

import pytest

from assistant.orchestration.fast_brain.adapter import FastBrainRequest, safe_decide
from assistant.orchestration.fast_brain.contract import (
    CapabilityFamily, Clarification, ContextRequirement, Escalation,
    EscalationReason, ExecutionPath, FastBrainDecision, IntentFamily,
    MissingInformation, Uncertainty,
)
from assistant.orchestration.fast_brain.deterministic import DeterministicFastBrain
from assistant.orchestration.fast_brain.shadow import observe
from assistant.orchestration.fast_brain.validator import DecisionValidationError, safe_fallback, validate_decision


def valid(**overrides):
    data = dict(
        intent_family=IntentFamily.LOCAL_TASK,
        execution_path=ExecutionPath.QWEN_AGENT,
        capability_family=CapabilityFamily.PROJECT,
        context_requirements=(ContextRequirement.PROJECT,),
        clarification=Clarification(),
        uncertainty=Uncertainty.HIGH,
        escalation=Escalation(),
    )
    data.update(overrides)
    return FastBrainDecision(**data)


def test_valid_decision_serializes_round_trip():
    decision = valid()
    assert FastBrainDecision.from_mapping(decision.to_dict()) == decision


@pytest.mark.parametrize("field,value", [
    ("intent_family", "not-real"),
    ("execution_path", "not-real"),
    ("capability_family", "not-real"),
    ("uncertainty", "not-real"),
])
def test_invalid_enum_rejected(field, value):
    payload = valid().to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        FastBrainDecision.from_mapping(payload)


def test_clarification_invariants():
    with pytest.raises(DecisionValidationError):
        validate_decision(valid(clarification=Clarification(required=True)))
    with pytest.raises(DecisionValidationError):
        validate_decision(valid(clarification=Clarification(missing=(MissingInformation.TARGET,))))


def test_escalation_invariants():
    with pytest.raises(DecisionValidationError):
        validate_decision(valid(escalation=Escalation(candidate=True)))
    with pytest.raises(DecisionValidationError):
        validate_decision(valid(escalation=Escalation(reason=EscalationReason.CONTEXT_OVERLOAD)))


def test_path_invariants():
    with pytest.raises(DecisionValidationError):
        validate_decision(valid(execution_path=ExecutionPath.FAST_PATH, capability_family=CapabilityFamily.UNKNOWN))
    with pytest.raises(DecisionValidationError):
        validate_decision(valid(execution_path=ExecutionPath.CLARIFY))


def test_safe_fallback_is_non_executing_clarification():
    fallback = safe_fallback()
    assert fallback.execution_path is ExecutionPath.CLARIFY
    assert fallback.clarification.required
    assert fallback.clarification.missing == (MissingInformation.REFERENCE,)


def test_deterministic_backend_examples_and_unknown_conservative_response():
    brain = DeterministicFastBrain()
    assert brain.decide(FastBrainRequest("what time is it")).execution_path is ExecutionPath.FAST_PATH
    assert brain.decide(FastBrainRequest("open chrome")).capability_family is CapabilityFamily.APPLICATION
    unknown = brain.decide(FastBrainRequest("explain quantum consciousness")).execution_path
    assert unknown is ExecutionPath.RESPOND


def test_adapter_failure_is_safe():
    class Broken:
        backend_name = "broken"
        def decide(self, request):
            raise RuntimeError("offline")
    decision = safe_decide(Broken(), FastBrainRequest("Open it"))
    assert decision.execution_path is ExecutionPath.CLARIFY


def test_shadow_preserves_existing_route_and_measures_latency():
    called = []
    result = observe(DeterministicFastBrain(), "open chrome", existing_route="GEMINI",
                     route_callback=lambda text: called.append(text))
    assert result.existing_route == "GEMINI"
    assert called == []
    assert result.validation_ok
    assert result.elapsed_ms >= 0
    assert result.decision["execution_path"] == "fast_path"
