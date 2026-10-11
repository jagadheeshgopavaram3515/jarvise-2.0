from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .contract import FastBrainDecision
from .validator import safe_fallback, validate_decision


@dataclass(frozen=True)
class FastBrainRequest:
    transcript: str
    context: Mapping[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()


@runtime_checkable
class FastBrainAdapter(Protocol):
    @property
    def backend_name(self) -> str: ...

    def decide(self, request: FastBrainRequest) -> FastBrainDecision: ...


def safe_decide(adapter: FastBrainAdapter, request: FastBrainRequest) -> FastBrainDecision:
    """Adapter boundary helper; errors become conservative clarification."""
    try:
        return validate_decision(adapter.decide(request))
    except Exception:
        return safe_fallback()
