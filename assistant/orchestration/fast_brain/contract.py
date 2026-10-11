from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class IntentFamily(_ValueEnum):
    CONVERSATION = "conversation"
    INFORMATION = "information"
    DETERMINISTIC_ACTION = "deterministic_action"
    LOCAL_TASK = "local_task"
    RESEARCH = "research"
    MEMORY = "memory"
    MULTI_STEP_TASK = "multi_step_task"
    CLARIFICATION = "clarification"
    UNSUPPORTED = "unsupported"


class ExecutionPath(_ValueEnum):
    FAST_PATH = "fast_path"
    RESPOND = "respond"
    QWEN_AGENT = "qwen_agent"
    MEMORY = "memory"
    RESEARCH = "research"
    CLARIFY = "clarify"
    ESCALATE = "escalate"


class CapabilityFamily(_ValueEnum):
    NONE = "none"
    APPLICATION = "application"
    BROWSER = "browser"
    FILE = "file"
    PROJECT = "project"
    WEATHER = "weather"
    LOCATION = "location"
    REMINDER = "reminder"
    MEMORY = "memory"
    RESEARCH = "research"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ContextRequirement(_ValueEnum):
    CONVERSATION = "conversation"
    MEMORY = "memory"
    PROJECT = "project"
    CURRENT_INFORMATION = "current_information"
    OBSERVATIONS = "observations"
    ACTIVE_TASK = "active_task"
    NONE = "none"


class Uncertainty(_ValueEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    AMBIGUOUS = "ambiguous"
    OUTSIDE_ENVELOPE = "outside_envelope"


class MissingInformation(_ValueEnum):
    TARGET = "target"
    FILE = "file"
    APPLICATION = "application"
    URL = "url"
    QUERY = "query"
    LOCATION = "location"
    TIME = "time"
    SCOPE = "scope"
    PERMISSION = "permission"
    REFERENCE = "reference"


class EscalationReason(_ValueEnum):
    CURRENT_INFORMATION_REQUIRED = "current_information_required"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    OUTSIDE_QWEN_ENVELOPE = "outside_qwen_envelope"
    REPEATED_LOCAL_FAILURE = "repeated_local_failure"
    CONTEXT_OVERLOAD = "context_overload"
    EXPLICIT_CLOUD_REQUEST = "explicit_cloud_request"


@dataclass(frozen=True)
class Clarification:
    required: bool = False
    missing: tuple[MissingInformation, ...] = ()


@dataclass(frozen=True)
class Escalation:
    candidate: bool = False
    reason: EscalationReason | None = None


@dataclass(frozen=True)
class FastBrainDecision:
    """Frozen seven-concept Fast Brain decision.

    This is advisory data. It has no execution method and no provider/model
    dependency. Validation is explicit so malformed adapter output cannot be
    mistaken for an operational route.
    """

    intent_family: IntentFamily
    execution_path: ExecutionPath
    capability_family: CapabilityFamily
    context_requirements: tuple[ContextRequirement, ...] = (ContextRequirement.NONE,)
    clarification: Clarification = field(default_factory=Clarification)
    uncertainty: Uncertainty = Uncertainty.MEDIUM
    escalation: Escalation = field(default_factory=Escalation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_family": self.intent_family.value,
            "execution_path": self.execution_path.value,
            "capability_family": self.capability_family.value,
            "context_requirements": [x.value for x in self.context_requirements],
            "clarification": {
                "required": self.clarification.required,
                "missing": [x.value for x in self.clarification.missing],
            },
            "uncertainty": self.uncertainty.value,
            "escalation": {
                "candidate": self.escalation.candidate,
                "reason": self.escalation.reason.value if self.escalation.reason else None,
            },
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FastBrainDecision":
        """Strictly deserialize untrusted adapter output."""
        try:
            clarification = value["clarification"]
            escalation = value["escalation"]
            decision = cls(
                intent_family=IntentFamily(value["intent_family"]),
                execution_path=ExecutionPath(value["execution_path"]),
                capability_family=CapabilityFamily(value["capability_family"]),
                context_requirements=tuple(ContextRequirement(x) for x in value["context_requirements"]),
                clarification=Clarification(
                    required=clarification["required"],
                    missing=tuple(MissingInformation(x) for x in clarification["missing"]),
                ),
                uncertainty=Uncertainty(value["uncertainty"]),
                escalation=Escalation(
                    candidate=escalation["candidate"],
                    reason=EscalationReason(escalation["reason"]) if escalation["reason"] is not None else None,
                ),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"invalid Fast Brain decision: {exc}") from exc
        from .validator import validate_decision
        return validate_decision(decision)
