"""Shadow-only Fast Brain foundation.

This package is deliberately not imported by the production dispatcher or
router. It provides the frozen decision contract, validation, and a pure
deterministic baseline for future shadow evaluation.
"""

from .adapter import FastBrainAdapter, FastBrainRequest
from .contract import (
    CapabilityFamily,
    Clarification,
    ContextRequirement,
    Escalation,
    ExecutionPath,
    FastBrainDecision,
    IntentFamily,
    Uncertainty,
)
from .deterministic import DeterministicFastBrain
from .shadow import ShadowDecision, observe
from .validator import DecisionValidationError, validate_decision

__all__ = [
    "FastBrainAdapter", "FastBrainRequest", "FastBrainDecision",
    "DeterministicFastBrain", "ShadowDecision", "observe",
    "validate_decision", "DecisionValidationError", "IntentFamily",
    "ExecutionPath", "CapabilityFamily", "ContextRequirement",
    "Clarification", "Uncertainty", "Escalation",
]
