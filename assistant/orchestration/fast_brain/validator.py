from __future__ import annotations

from dataclasses import replace

from .contract import (
    CapabilityFamily, Clarification, ContextRequirement, Escalation,
    ExecutionPath, FastBrainDecision, IntentFamily, MissingInformation,
    Uncertainty,
)


class DecisionValidationError(ValueError):
    pass


def validate_decision(decision: FastBrainDecision) -> FastBrainDecision:
    if not isinstance(decision, FastBrainDecision):
        raise DecisionValidationError("decision must be FastBrainDecision")
    if not decision.context_requirements:
        raise DecisionValidationError("context_requirements must not be empty")
    if ContextRequirement.NONE in decision.context_requirements and len(decision.context_requirements) > 1:
        raise DecisionValidationError("none cannot be combined with other context requirements")
    if decision.clarification.required and not decision.clarification.missing:
        raise DecisionValidationError("required clarification needs missing information")
    if not decision.clarification.required and decision.clarification.missing:
        raise DecisionValidationError("non-required clarification cannot have missing information")
    if decision.escalation.candidate and decision.escalation.reason is None:
        raise DecisionValidationError("escalation candidate needs a reason")
    if not decision.escalation.candidate and decision.escalation.reason is not None:
        raise DecisionValidationError("non-candidate escalation cannot have a reason")
    if decision.execution_path is ExecutionPath.FAST_PATH and decision.capability_family is CapabilityFamily.UNKNOWN:
        raise DecisionValidationError("fast_path cannot use unknown capability")
    if decision.execution_path is ExecutionPath.CLARIFY and not decision.clarification.required:
        raise DecisionValidationError("clarify requires clarification.required")
    if decision.intent_family is IntentFamily.CLARIFICATION and not decision.clarification.required:
        raise DecisionValidationError("clarification intent requires clarification.required")
    return decision


def safe_fallback() -> FastBrainDecision:
    """Conservative, non-executing fallback for adapter failures."""
    return FastBrainDecision(
        intent_family=IntentFamily.CLARIFICATION,
        execution_path=ExecutionPath.CLARIFY,
        capability_family=CapabilityFamily.UNKNOWN,
        context_requirements=(ContextRequirement.CONVERSATION,),
        clarification=Clarification(required=True, missing=(MissingInformation.REFERENCE,)),
        uncertainty=Uncertainty.AMBIGUOUS,
        escalation=Escalation(),
    )
