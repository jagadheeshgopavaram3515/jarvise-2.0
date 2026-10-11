from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from assistant.orchestration.fast_brain.contract import (
    CapabilityFamily, IntentFamily, MissingInformation,
)


PROVENANCE = frozenset({"human_reviewed", "expert_authored", "teacher_generated", "synthetic", "historical_synthetic"})
REVIEW_STATUS = frozenset({"reviewed", "pending_review", "rejected"})
SPLITS = frozenset({"train", "development", "sealed_candidate"})
LANGUAGE_GROUPS = frozenset({"english", "telugu", "hindi", "tamil", "romanized_telugu", "romanized_hindi", "code_switch"})
SCRIPT_GROUPS = frozenset({"latin", "telugu", "devanagari", "tamil", "mixed", "unknown"})
NOISE_TYPES = frozenset({"clean", "stt_variant"})


class DatasetValidationError(ValueError):
    pass


@dataclass(frozen=True)
class FastBrainRecord:
    id: str
    text: str
    intent_family: str
    capability_family: str
    clarification_required: bool
    missing_information: tuple[str, ...]
    provenance: str
    group_id: str
    split: str
    review_status: str
    language_group: str = "english"
    script_group: str = "latin"
    noise_type: str = "clean"
    hard_negative_group: str | None = None
    context_resolvable: bool | None = None
    semantic_language: str | None = None
    input_script: str | None = None
    response_language: str | None = None
    original_dataset: str | None = None
    original_id: str | None = None
    original_label: str | None = None
    original_provenance: str | None = None
    original_review_status: str | None = None
    rationale: str | None = None
    generation_reason: str | None = None
    source_type: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FastBrainRecord":
        required = ("id", "text", "intent_family", "capability_family", "clarification_required",
                    "missing_information", "provenance", "group_id", "split", "review_status")
        missing = [key for key in required if key not in raw]
        if missing:
            raise DatasetValidationError(f"missing fields: {', '.join(missing)}")
        record = cls(
            id=str(raw["id"]), text=str(raw["text"]), intent_family=str(raw["intent_family"]),
            capability_family=str(raw["capability_family"]), clarification_required=raw["clarification_required"],
            missing_information=tuple(str(x) for x in raw["missing_information"]),
            provenance=str(raw["provenance"]), group_id=str(raw["group_id"]), split=str(raw["split"]),
            review_status=str(raw["review_status"]), language_group=str(raw.get("language_group", "english")),
            script_group=str(raw.get("script_group", "latin")), noise_type=str(raw.get("noise_type", "clean")),
            hard_negative_group=raw.get("hard_negative_group"), context_resolvable=raw.get("context_resolvable"),
            semantic_language=raw.get("semantic_language"), input_script=raw.get("input_script"),
            response_language=raw.get("response_language"), original_dataset=raw.get("original_dataset"),
            original_id=raw.get("original_id"), original_label=raw.get("original_label"),
            original_provenance=raw.get("original_provenance"), original_review_status=raw.get("original_review_status"),
            rationale=raw.get("rationale"), generation_reason=raw.get("generation_reason"),
            source_type=raw.get("source_type"),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not self.id or not self.text.strip() or not self.group_id:
            raise DatasetValidationError(f"{self.id or '<unknown>'}: id, text, and group_id are required")
        if self.intent_family not in {x.value for x in IntentFamily}:
            raise DatasetValidationError(f"{self.id}: unknown intent_family {self.intent_family}")
        if self.capability_family not in {x.value for x in CapabilityFamily}:
            raise DatasetValidationError(f"{self.id}: unknown capability_family {self.capability_family}")
        if self.provenance not in PROVENANCE or self.review_status not in REVIEW_STATUS:
            raise DatasetValidationError(f"{self.id}: invalid provenance or review_status")
        if self.split not in SPLITS or self.language_group not in LANGUAGE_GROUPS:
            raise DatasetValidationError(f"{self.id}: invalid split or language_group")
        if self.script_group not in SCRIPT_GROUPS or self.noise_type not in NOISE_TYPES:
            raise DatasetValidationError(f"{self.id}: invalid script_group or noise_type")
        if not isinstance(self.clarification_required, bool):
            raise DatasetValidationError(f"{self.id}: clarification_required must be boolean")
        allowed_missing = {x.value for x in MissingInformation}
        if any(x not in allowed_missing for x in self.missing_information):
            raise DatasetValidationError(f"{self.id}: unknown missing_information label")
        if self.clarification_required != bool(self.missing_information):
            raise DatasetValidationError(f"{self.id}: clarification/missing_information invariant violated")

