from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ReplayCase:
    id: str
    language: str
    user_input: str
    context: dict[str, Any] = field(default_factory=dict)
    available_capabilities: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    expected: dict[str, Any] = field(default_factory=dict)
    fixtures: dict[str, Any] = field(default_factory=dict)
    group: str = ""
    review_status: str = "synthetic"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReplayCase":
        required = ("id", "language", "user_input", "expected")
        for key in ("id", "language", "user_input"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f"{key} must be a non-empty string")
        if not isinstance(value.get("expected"), Mapping):
            raise ValueError("expected must be an object")
        return cls(
            id=value["id"], language=value["language"], user_input=value["user_input"],
            context=dict(value.get("context") or {}),
            available_capabilities=list(value.get("available_capabilities") or []),
            observations=list(value.get("observations") or []),
            expected=dict(value["expected"]), fixtures=dict(value.get("fixtures") or {}),
            group=value.get("group", value["id"]), review_status=value.get("review_status", "synthetic"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_input(case: ReplayCase, system: str, decision: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build model input. Expected labels are deliberately never consulted."""
    payload: dict[str, Any] = {
        "request": case.user_input,
        "context": case.context,
        "capabilities": case.available_capabilities,
        "observations": case.observations,
        "system": system,
    }
    if decision is not None:
        payload["decision"] = dict(decision)
    return payload
