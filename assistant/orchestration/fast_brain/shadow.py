from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .adapter import FastBrainAdapter, FastBrainRequest, safe_decide


@dataclass(frozen=True)
class ShadowDecision:
    transcript: str
    backend_name: str
    decision: dict[str, Any]
    existing_route: Any
    validation_ok: bool
    elapsed_ms: float
    error: str | None = None


def observe(
    adapter: FastBrainAdapter,
    transcript: str,
    *,
    existing_route: Any = None,
    context: Mapping[str, Any] | None = None,
    capabilities: tuple[str, ...] = (),
    route_callback: Callable[[str], Any] | None = None,
) -> ShadowDecision:
    """Run the Fast Brain beside existing routing without invoking it.

    `route_callback` is intentionally optional and is never called with the
    Fast Brain decision. It may be used by a future observer to record the
    already-authoritative route, but this function never changes that route.
    """
    started = time.perf_counter()
    error = None
    validation_ok = True
    try:
        decision = safe_decide(adapter, FastBrainRequest(transcript, context or {}, capabilities))
    except Exception as exc:  # defensive: shadow mode must never affect caller
        validation_ok = False
        error = repr(exc)
        decision = safe_decide(adapter, FastBrainRequest(transcript))
    return ShadowDecision(
        transcript=transcript,
        backend_name=getattr(adapter, "backend_name", type(adapter).__name__),
        decision=decision.to_dict(),
        existing_route=existing_route,
        validation_ok=validation_ok,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        error=error,
    )
