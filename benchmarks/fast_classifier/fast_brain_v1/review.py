from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any, Mapping

from .schema import DatasetValidationError, FastBrainRecord
from assistant.orchestration.fast_brain.contract import FastBrainDecision


REVIEW_DECISIONS = frozenset({"accept", "correct", "reject", "ambiguous", "skip"})
ADJUDICATION_STATUS = frozenset({"not_required", "requires_adjudication", "adjudicated"})
READINESS_STATES = frozenset({"CANDIDATE", "REVIEW_IN_PROGRESS", "REVIEWED_NOT_ADJUDICATED", "TRAINING_READY", "SEALED_TEST_READY"})
EXPERIMENT_READINESS = frozenset({"READY_FOR_EXPERIMENTS", "BLOCKED_INSUFFICIENT_HUMAN_LABELS", "BLOCKED_LABEL_CONFLICTS", "BLOCKED_DATA_QUALITY"})


@dataclass(frozen=True)
class ReviewDecision:
    record_id: str
    reviewer_id: str
    decision: str
    reviewed_label: Mapping[str, Any] | None = None
    reviewer_notes: str = ""
    disagreement_fields: tuple[str, ...] = ()
    adjudication_status: str = "not_required"
    reviewed_at: str | None = None

    def validate(self) -> None:
        if not self.record_id or not self.reviewer_id:
            raise DatasetValidationError("record_id and reviewer_id are required")
        if self.decision not in REVIEW_DECISIONS:
            raise DatasetValidationError(f"invalid reviewer decision: {self.decision}")
        if self.adjudication_status not in ADJUDICATION_STATUS:
            raise DatasetValidationError(f"invalid adjudication status: {self.adjudication_status}")
        if self.decision == "correct" and not self.reviewed_label:
            raise DatasetValidationError("correct decisions require reviewed_label")
        if self.adjudication_status == "adjudicated" and not self.reviewed_label:
            raise DatasetValidationError("adjudicated decisions require reviewed_label")
        if self.reviewed_label is not None:
            try:
                FastBrainDecision.from_mapping(self.reviewed_label)
            except (TypeError, ValueError) as exc:
                raise DatasetValidationError(f"invalid reviewed_label: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "record_id": self.record_id, "reviewer_id": self.reviewer_id,
            "reviewer_decision": self.decision, "reviewed_label": self.reviewed_label,
            "reviewer_notes": self.reviewer_notes, "disagreement_fields": list(self.disagreement_fields),
            "adjudication_status": self.adjudication_status,
            "reviewed_at": self.reviewed_at or datetime.now(timezone.utc).isoformat(),
        }


def review_priority(record: FastBrainRecord) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if record.language_group in {"telugu", "hindi", "code_switch"} or record.script_group in {"telugu", "devanagari", "mixed"}:
        reasons.append("multilingual_metadata")
    if record.original_review_status not in {"pending_review", "assistant_reviewed_pending_human_review"}:
        reasons.append("historical_review_status_unclear")
    if record.clarification_required or record.intent_family in {"clarification", "unsupported"}:
        reasons.append("clarification_or_unsupported_boundary")
    if record.hard_negative_group:
        reasons.append("hard_negative")
    if record.context_resolvable is not None:
        reasons.append("context_dependent")
    if record.capability_family in {"unknown", "project", "research", "memory"}:
        reasons.append("high_risk_capability")
    if record.noise_type == "stt_variant":
        reasons.append("stt_noise")
    priority = "critical" if any(x in reasons for x in ("multilingual_metadata", "clarification_or_unsupported_boundary")) else (
        "high" if reasons else "normal")
    return priority, reasons


def validate_review_set(records: list[FastBrainRecord], reviewed: list[Mapping[str, Any]]) -> None:
    """Validate review artifacts before they can affect readiness."""
    known = {record.id for record in records}
    seen: set[str] = set()
    for raw in reviewed:
        record_id = raw.get("record_id")
        if record_id in seen:
            raise DatasetValidationError(f"duplicate review record: {record_id}")
        if record_id not in known:
            raise DatasetValidationError(f"review for unknown record: {record_id}")
        seen.add(record_id)
        decision = ReviewDecision(
            record_id=str(record_id), reviewer_id=raw.get("reviewer_id") or "",
            decision=raw.get("reviewer_decision") or "", reviewed_label=raw.get("reviewed_label"),
            reviewer_notes=raw.get("reviewer_notes", ""),
            disagreement_fields=tuple(raw.get("disagreement_fields", [])),
            adjudication_status=raw.get("adjudication_status", ""), reviewed_at=raw.get("reviewed_at"),
        )
        decision.validate()


def readiness(records: list[FastBrainRecord], reviewed: list[Mapping[str, Any]] | None = None) -> str:
    reviewed = reviewed or []
    if not records:
        return "CANDIDATE"
    try:
        validate_review_set(records, reviewed)
    except DatasetValidationError:
        return "CANDIDATE"
    by_id = {x["record_id"]: x for x in reviewed}
    if not by_id:
        return "CANDIDATE"
    if len(by_id) < len(records):
        return "REVIEW_IN_PROGRESS"
    unresolved = [x for x in reviewed if x["reviewer_decision"] not in {"accept", "correct"}
                  or x.get("adjudication_status") == "requires_adjudication"]
    if unresolved:
        return "REVIEWED_NOT_ADJUDICATED"
    return "TRAINING_READY"


def experiment_readiness_report(records: list[FastBrainRecord], reviewed: list[Mapping[str, Any]],
                               *, required_languages: tuple[str, ...] = ("english", "telugu", "hindi", "tamil", "code_switch"),
                               min_examples_per_language: int = 1) -> dict[str, Any]:
    """Return a deterministic, non-mutating gate for supervised experiments.

    Only accepted/correct decisions with a valid reviewed_label are eligible.
    The default language requirement is deliberately minimal coverage, not a
    claim that one example per language is statistically sufficient.
    """
    from collections import Counter
    by_id = {record.id: record for record in records}
    counts = Counter((item.get("reviewer_decision") or "pending") for item in reviewed)
    eligible: list[tuple[FastBrainRecord, Mapping[str, Any]]] = []
    conflicts: list[str] = []
    quality: list[str] = []
    seen_labels: dict[str, str] = {}
    seen_text_groups: dict[str, str] = {}
    for record in records:
        previous_group = seen_text_groups.get(record.text)
        if previous_group is not None and previous_group != record.group_id:
            quality.append(f"duplicate text crosses groups: {record.id}")
        seen_text_groups[record.text] = record.group_id
    try:
        validate_review_set(records, reviewed)
    except DatasetValidationError as exc:
        quality.append(str(exc))
    for item in reviewed:
        record = by_id.get(item.get("record_id"))
        if record is None or item.get("reviewer_decision") not in {"accept", "correct"}:
            continue
        label = item.get("reviewed_label") or {
            "intent_family": record.intent_family, "execution_path": "respond",
            "capability_family": record.capability_family, "context_requirements": ["active_task"],
            "clarification": {"required": record.clarification_required, "missing": list(record.missing_information)},
            "uncertainty": "medium", "escalation": {"candidate": record.intent_family == "unsupported",
            "reason": "unsupported_capability" if record.intent_family == "unsupported" else None}}
        try:
            normalized = FastBrainDecision.from_mapping(label).to_dict()
        except (TypeError, ValueError) as exc:
            quality.append(f"{item.get('record_id')}: invalid reviewed label: {exc}")
            continue
        fingerprint = json.dumps(normalized, sort_keys=True)
        previous = seen_labels.get(record.text)
        if previous is not None and previous != fingerprint:
            conflicts.append(record.id)
        seen_labels[record.text] = fingerprint
        eligible.append((record, normalized))
    language_counts = Counter(record.language_group for record, _ in eligible)
    missing_languages = [x for x in required_languages if language_counts[x] < min_examples_per_language]
    usable = len(eligible)
    status = ("BLOCKED_DATA_QUALITY" if quality else
              "BLOCKED_LABEL_CONFLICTS" if conflicts else
              "BLOCKED_INSUFFICIENT_HUMAN_LABELS" if missing_languages or usable == 0 else
              "READY_FOR_EXPERIMENTS")
    return {"status": status, "usable_sample_count": usable,
            "decision_counts": dict(sorted(counts.items())),
            "eligible_languages": dict(sorted(language_counts.items())),
            "missing_languages": missing_languages, "label_conflicts": conflicts,
            "data_quality_errors": quality,
            "eligible_record_ids": [record.id for record, _ in eligible]}


def review_artifacts_are_protected(queue_text: str, adjudication_text: str) -> tuple[bool, str | None]:
    """Return whether regeneration must refuse to overwrite review work."""
    if adjudication_text.strip():
        return True, "adjudication_queue.jsonl is non-empty"
    for line in queue_text.splitlines():
        if not line.strip():
            continue
        item = __import__("json").loads(line)
        if (item.get("reviewer_decision") or item.get("reviewed_label") or item.get("reviewer_id")
                or item.get("reviewed_at") or item.get("reviewer_notes")
                or item.get("disagreement_fields") or item.get("adjudication_status") == "adjudicated"):
            return True, "review_queue.jsonl contains human-review data"
    return False, None

