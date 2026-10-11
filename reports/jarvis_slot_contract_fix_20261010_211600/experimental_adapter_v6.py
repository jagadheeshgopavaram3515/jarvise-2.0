"""Experimental v6 Adapter with Canonical Slot Contract Resolver.

Preserves the validated v5 feature extraction, script-aware calibration, and capability ranking,
while adding canonical argument schema resolution, strict decision contract validation,
and safe capability admissibility filtering.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from assistant.orchestration.decision import (
    ClassifierDecision,
    DecisionRequest,
)
from assistant.orchestration.compact_adapter import (
    _decision_to_mapping,
)
from experimental_adapter_v5 import (
    ExperimentalV5Adapter,
)
from slot_contract_resolver import (
    CanonicalArgumentContract,
    SlotContractResolver,
)


class ExperimentalV6Adapter(ExperimentalV5Adapter):
    """v6 Adapter integrating the canonical slot contract resolver."""

    def __init__(self, artifact_path: str | Path, encoder: Any) -> None:
        super().__init__(artifact_path, encoder)
        self._adapter_id = "jarvis-compact-bert-v6-slot-contract"
        self.contract_resolver = SlotContractResolver()
        self.last_contract_metrics: dict[str, Any] = {}

    def predict(self, request: DecisionRequest) -> Mapping[str, Any]:
        """Runs v5 inference and passes the output through the canonical slot contract resolver."""
        started = time.perf_counter_ns()
        
        # 1. Run raw v5 prediction
        raw_v5 = super().predict(request)
        
        # 2. Apply canonical slot contract resolution
        contract_start = time.perf_counter_ns()
        resolved = self.contract_resolver.resolve_decision(raw_v5, request)
        contract_ms = (time.perf_counter_ns() - contract_start) / 1_000_000.0
        
        # 3. Validate contract compliance via ClassifierDecision
        validated_decision = ClassifierDecision.from_mapping(resolved)
        total_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        
        self.last_contract_metrics = {
            "contract_resolution_ms": contract_ms,
            "total_ms": total_ms,
            "contract_valid": True,
        }
        
        return _decision_to_mapping(validated_decision)
