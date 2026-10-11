from __future__ import annotations

import re

from .adapter import FastBrainAdapter, FastBrainRequest
from .contract import (
    CapabilityFamily, Clarification, ContextRequirement, Escalation,
    ExecutionPath, FastBrainDecision, IntentFamily, Uncertainty,
)


class DeterministicFastBrain:
    """Small conservative baseline; it only recognizes unambiguous phrases."""

    backend_name = "deterministic-v1"

    _STOP = re.compile(r"^(stop|cancel|pause|wait|quiet|halt|shush)[.!]?$")
    _TIME = re.compile(r"^(what(?:'s| is) the time|what time is it|current time)[?!.]?$", re.I)
    _DATE = re.compile(r"^(what(?:'s| is) today's date|what(?:'s| is) the date|current date)[?!.]?$", re.I)
    _OPEN_APP = re.compile(r"^(?:open|launch|start)\s+(chrome|youtube)$", re.I)

    def decide(self, request: FastBrainRequest) -> FastBrainDecision:
        text = request.transcript.strip()
        if self._STOP.match(text):
            return self._fast(CapabilityFamily.SYSTEM)
        if self._TIME.match(text) or self._DATE.match(text):
            return self._fast(CapabilityFamily.SYSTEM)
        if self._OPEN_APP.match(text):
            return self._fast(CapabilityFamily.APPLICATION)
        return FastBrainDecision(
            intent_family=IntentFamily.INFORMATION,
            execution_path=ExecutionPath.RESPOND,
            capability_family=CapabilityFamily.NONE,
            context_requirements=(ContextRequirement.NONE,),
            clarification=Clarification(),
            uncertainty=Uncertainty.LOW,
            escalation=Escalation(),
        )

    @staticmethod
    def _fast(capability: CapabilityFamily) -> FastBrainDecision:
        return FastBrainDecision(
            intent_family=IntentFamily.DETERMINISTIC_ACTION,
            execution_path=ExecutionPath.FAST_PATH,
            capability_family=capability,
            context_requirements=(ContextRequirement.NONE,),
            clarification=Clarification(),
            uncertainty=Uncertainty.HIGH,
            escalation=Escalation(),
        )
