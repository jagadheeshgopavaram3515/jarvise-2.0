"""Small, local, append-only human reviewer for the Fast Brain candidate queue."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

from .fast_brain_v1.review import ReviewDecision, readiness, validate_review_set
from .fast_brain_v1.schema import DatasetValidationError

ROOT = Path(__file__).resolve().parent
DEFAULT_QUEUE = ROOT / "fast_brain_v1" / "review_queue.jsonl"
DEFAULT_DECISIONS = ROOT / "fast_brain_v1" / "review_decisions.jsonl"
AI_MODES = frozenset({"AUTO_CONFIDENT", "AI_CORRECTION", "HUMAN_REVIEW", "SOURCE_CONFLICT"})
AI_CONFIDENCES = frozenset({"high", "medium", "low"})
LEGACY_EXECUTION_PATHS = {
    "direct_response": "respond",
}


def _original_mapping(record: Any) -> dict[str, Any] | None:
    value = getattr(record, "original_label", None)
    if not value:
        return None
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_legacy_label(original: dict[str, Any]) -> dict[str, Any]:
    """Normalize only documented vocabulary aliases; preserve all semantics."""
    normalized = dict(original)
    path = normalized.get("execution_path")
    if path in LEGACY_EXECUTION_PATHS:
        normalized["execution_path"] = LEGACY_EXECUTION_PATHS[path]
    return normalized


def source_conflict_evidence(record: Any, canonical: dict[str, Any]) -> tuple[bool, str]:
    """Return a conflict only when source metadata materially contradicts canonical data."""
    original = _original_mapping(record)
    if original is None:
        return False, "original source label/provenance is missing or unreadable"
    normalized = normalize_legacy_label(original)
    source_intent = original.get("intent_family")
    source_intent = source_intent or ({
        "respond": "conversation",
        "research": "research",
        "unsupported_request": "unsupported",
    }.get(original.get("intent")))
    # A legacy spelling difference is explicitly non-conflicting, but only
    # after checking semantic intent and escalation evidence.
    if (original.get("execution_path") == "multi_step_task"
            and original.get("intent") == "research"
            and canonical.get("intent_family") == "multi_step_task"
            and canonical.get("capability_family") not in {"none", "unknown"}):
        return False, "multi-step research source semantics support the canonical agent route"
    if (source_intent and source_intent != canonical.get("intent_family")):
        return True, f"source intent {source_intent!r} contradicts canonical intent {canonical.get('intent_family')!r}"
    if (original.get("intent") == "unsupported_request"
            and original.get("needs_escalation") is True
            and canonical.get("intent_family") == "unsupported"
            and canonical.get("escalation", {}).get("candidate") is True):
        return False, "source unsupported/escalation semantics support canonical unsupported label"
    if normalized.get("execution_path") == canonical.get("execution_path"):
        return False, "legacy/current execution-path vocabulary is compatible"
    # Do not call single_action/qwen_agent equivalent without evidence from the
    # source record.  A mismatch is still insufficient for SOURCE_CONFLICT;
    # the human must adjudicate incomplete evidence.
    return False, "source and canonical labels require human adjudication"


def _compatibility_proposal(record: Any, canonical: dict[str, Any]) -> AIReviewProposal | None:
    """Allow only an explicit, semantically complete legacy alias to auto-confirm."""
    original = _original_mapping(record)
    if not original or original.get("execution_path") != "direct_response":
        return None
    if original.get("intent") != "respond" or canonical.get("intent_family") != "conversation":
        return None
    return AIReviewProposal(
        "AUTO_CONFIDENT", canonical, "high",
        "Original conversational source label direct_response maps explicitly to current respond.",
        ("Original intent=respond and execution_path=direct_response; canonical intent=conversation and execution_path=respond.",),
        False)


def _requires_human_for_canonical(canonical: dict[str, Any]) -> bool:
    """Unsupported/escalation candidates always require a human decision."""
    return (canonical.get("intent_family") == "unsupported"
            or canonical.get("escalation", {}).get("candidate") is True)


def _diagnostic(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "request timeout"
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
        return "model not found"
    if isinstance(exc, urllib.error.URLError):
        return "Ollama connection refused/unavailable"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid JSON"
    if isinstance(exc, DatasetValidationError):
        return "invalid contract"
    return "invalid Ollama response"


@dataclass(frozen=True)
class AIReviewProposal:
    mode: str
    proposed_label: dict[str, Any] | None
    confidence: str
    rationale: str
    source_evidence: tuple[str, ...] = ()
    human_attention: bool = False

    def validate(self) -> None:
        if self.mode not in AI_MODES:
            raise DatasetValidationError(f"invalid AI review mode: {self.mode}")
        if self.confidence not in AI_CONFIDENCES:
            raise DatasetValidationError(f"invalid AI confidence: {self.confidence}")
        if not self.rationale.strip():
            raise DatasetValidationError("AI rationale is required")
        if not isinstance(self.source_evidence, tuple) or any(not x.strip() for x in self.source_evidence):
            raise DatasetValidationError("source_evidence must contain non-empty strings")
        if self.mode in {"HUMAN_REVIEW", "SOURCE_CONFLICT"} and self.proposed_label is not None:
            raise DatasetValidationError("uncertain AI proposals cannot include a proposed_label")
        if self.mode in {"AUTO_CONFIDENT", "AI_CORRECTION"} and self.proposed_label is None:
            raise DatasetValidationError("confident AI proposals require proposed_label")
        if self.proposed_label is not None:
            try:
                from assistant.orchestration.fast_brain.contract import FastBrainDecision
                FastBrainDecision.from_mapping(self.proposed_label)
            except (TypeError, ValueError) as exc:
                raise DatasetValidationError(f"invalid AI proposed_label: {exc}") from exc
        if self.mode in {"HUMAN_REVIEW", "SOURCE_CONFLICT"} and not self.human_attention:
            raise DatasetValidationError("uncertain AI proposals require human attention")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"mode": self.mode, "proposed_label": self.proposed_label,
                "confidence": self.confidence, "rationale": self.rationale,
                "source_evidence": list(self.source_evidence),
                "human_attention": self.human_attention}


def validate_ai_proposal(value: Any) -> AIReviewProposal:
    if isinstance(value, AIReviewProposal):
        value.validate()
        return value
    if not isinstance(value, dict):
        raise DatasetValidationError("AI proposal must be an object")
    proposal = AIReviewProposal(
        mode=value.get("mode", ""), proposed_label=value.get("proposed_label"),
        confidence=value.get("confidence", ""), rationale=value.get("rationale", ""),
        source_evidence=tuple(value.get("source_evidence", ())),
        human_attention=value.get("human_attention", False),
    )
    proposal.validate()
    return proposal


class OllamaReviewProposer:
    """Resident local proposer. It only returns advisory data; it never persists reviews."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5-coder:3b",
                 timeout: float = 20.0) -> None:
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout

    def propose(self, record: Any, canonical: dict[str, Any]) -> AIReviewProposal:
        prompt = _ai_prompt(record, canonical)
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "keep_alive": "10m", "format": "json",
                   "options": {"temperature": 0.0, "num_predict": 700}}
        try:
            request = urllib.request.Request(
                f"{self.base_url}/api/generate", data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "JARVIS-review"})
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            raw = body.get("response", body)
            if isinstance(raw, str):
                raw = json.loads(raw)
            proposal = validate_ai_proposal(raw)
            if (proposal.mode in {"AUTO_CONFIDENT", "AI_CORRECTION"}
                    and _requires_human_for_canonical(canonical)):
                return AIReviewProposal(
                    "HUMAN_REVIEW", None, "low",
                    "Unsupported or escalation candidates require explicit human review.",
                    ("Canonical escalation policy requires human review.",), True)
            if (proposal.mode in {"AUTO_CONFIDENT", "AI_CORRECTION"}
                    and proposal.proposed_label["clarification"]["required"]):
                # A valid clarification label still describes unresolved evidence.
                # Inspect the proposed label, so a supported correction that resolves
                # an incorrectly ambiguous candidate can still be reviewed normally.
                return AIReviewProposal(
                    "HUMAN_REVIEW", None, "low",
                    "The proposed label still requires missing information; human review is required.",
                    ("Unresolved missing information: " + ", ".join(
                        proposal.proposed_label["clarification"]["missing"]),), True)
            if proposal.mode == "SOURCE_CONFLICT":
                conflict, evidence = source_conflict_evidence(record, canonical)
                if not conflict:
                    compatible = _compatibility_proposal(record, canonical)
                    if compatible is not None:
                        return compatible
                    return AIReviewProposal(
                        "HUMAN_REVIEW", None, "low",
                        f"insufficient source-conflict evidence: {evidence}",
                        (evidence,), True)
            return proposal
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                TypeError, ValueError, DatasetValidationError) as exc:
            return AIReviewProposal("HUMAN_REVIEW", None, "low",
                                    f"{_diagnostic(exc)}: {exc}",
                                    ("No reliable AI output was available.",), True)


def _ai_prompt(record: Any, canonical: dict[str, Any]) -> str:
    original = getattr(record, "original_label", None) or "unavailable"
    provenance = getattr(record, "provenance", "unavailable")
    source_metadata = getattr(record, "source_metadata", {})
    return ("This is a voice-assistant semantic classification task. The utterance is natural language, "
            "NOT a contract document. The candidate label is the structured interpretation of what the "
            "user wants. Compare MEANING, never compare the literal text with JSON keys. A request can "
            "correctly mean research or an application action without mentioning any label field.\n\n"
            "You are an evidence-bound human-review assistant. Do not use outside knowledge. "
            "First perform semantic label review: compare the candidate text with the canonical "
            "seven-field label and original source metadata. If they are consistent, return "
            "AUTO_CONFIDENT or AI_CORRECTION with a COMPLETE proposed_label containing every "
            "field of the frozen Fast Brain contract. Never return SOURCE_CONFLICT merely because "
            "legacy/current vocabulary, intent/capability abstraction, or uncertainty differs. "
            "Interpret natural language, transliterated Hindi/Telugu/Tamil, and code-switched "
            "technical terms by meaning; the user text will not literally contain contract field "
            "names such as intent_family or execution_path. Do not require those field names in "
            "the text. "
            "Use SOURCE_CONFLICT only for a concrete contradiction directly evidenced by the "
            "original label/provenance, and cite that exact evidence. Use HUMAN_REVIEW when source "
            "evidence is missing/incomplete, a legacy mapping is not explicit, or the canonical "
            "label is unsupported/escalation. Use only the candidate text, canonical label, "
            "original label, and provenance. "
            "Return JSON with exactly: mode (AUTO_CONFIDENT|AI_CORRECTION|HUMAN_REVIEW|SOURCE_CONFLICT), "
            "proposed_label (full Fast Brain contract object or null), confidence (high|medium|low), "
            "rationale (short), source_evidence (array of short strings), human_attention (boolean). "
            "For HUMAN_REVIEW or SOURCE_CONFLICT, proposed_label must be null and human_attention true. "
            "Never invent missing context or capabilities.\n\n"
            f"TEXT:\n{record.text}\n\nCANONICAL LABEL:\n{json.dumps(canonical, ensure_ascii=False)}\n"
            f"ORIGINAL LABEL:\n{original}\nPROVENANCE:\n{provenance}\n"
            f"SOURCE METADATA:\n{json.dumps(source_metadata, ensure_ascii=False)}")


def _load_queue(path: Path) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    records: list[Any] = []
    labels: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            record_id = str(raw["record_id"] if "record_id" in raw else raw["id"])
            canonical = raw.get("canonical_label") or _label_from_raw(raw)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise DatasetValidationError(f"queue line {line_number}: malformed review record") from exc
        from types import SimpleNamespace
        labels[record_id] = canonical
        records.append(SimpleNamespace(
            id=record_id, text=str(raw.get("text", "")), intent_family=canonical["intent_family"],
            capability_family=canonical["capability_family"], clarification_required=canonical["clarification"]["required"],
            missing_information=tuple(canonical["clarification"]["missing"]), language_group=raw.get("language", {}).get("semantic_language", "english"),
            script_group=raw.get("language", {}).get("input_script", "unknown"), semantic_language=raw.get("language", {}).get("semantic_language"),
            input_script=raw.get("language", {}).get("input_script"), response_language=raw.get("language", {}).get("response_language"),
            review_priority=raw.get("review_priority", "normal"), review_reasons=raw.get("review_reasons", []),
            original_label=raw.get("original_label"), provenance=raw.get("provenance", "unavailable"),
            source_metadata={key: raw[key] for key in ("original_dataset", "original_id", "source_type", "source_file") if key in raw}))
    return records, labels


def _label_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    return {"intent_family": raw.get("intent_family", "conversation"), "execution_path": raw.get("execution_path", "respond"),
            "capability_family": raw.get("capability_family", "none"), "context_requirements": ["none"],
            "clarification": {"required": bool(raw.get("clarification_required", False)), "missing": list(raw.get("missing_information", []))},
            "uncertainty": "medium", "escalation": {"candidate": False, "reason": None}}


def load_decisions(path: str | Path, records: list[Any]) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    decisions: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"decision line {line_number}: malformed JSON") from exc
        if not isinstance(value, dict):
            raise DatasetValidationError(f"decision line {line_number}: object required")
        decisions.append(value)
    validate_review_set(records, decisions)
    return decisions


class ReviewStore:
    def __init__(self, queue_path: str | Path = DEFAULT_QUEUE,
                 decisions_path: str | Path = DEFAULT_DECISIONS) -> None:
        self.queue_path = Path(queue_path)
        self.decisions_path = Path(decisions_path)
        self.records, self._canonical_labels = _load_queue(self.queue_path)
        self.decisions = load_decisions(self.decisions_path, self.records)
        self._ids = {x["record_id"] for x in self.decisions}

    def canonical_label(self, record: Any) -> dict[str, Any]:
        return self._canonical_labels[record.id]

    @property
    def pending(self) -> list[Any]:
        return [record for record in self.records if record.id not in self._ids]

    def append(self, decision: ReviewDecision) -> dict[str, Any]:
        decision.validate()
        if decision.record_id not in {record.id for record in self.records}:
            raise DatasetValidationError(f"review for unknown record: {decision.record_id}")
        if decision.record_id in self._ids:
            raise DatasetValidationError(f"duplicate review record: {decision.record_id}")
        value = decision.to_dict()
        self.decisions.append(value)
        self._ids.add(decision.record_id)
        self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with self.decisions_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        return value


def _label_from_record(record: Any) -> dict[str, Any]:
    return {
        "intent_family": record.intent_family,
        "execution_path": "research" if record.intent_family == "research" else "respond",
        "capability_family": record.capability_family,
        "context_requirements": ["active_task"],
        "clarification": {"required": record.clarification_required, "missing": list(record.missing_information)},
        "uncertainty": "medium",
        "escalation": {"candidate": record.intent_family == "unsupported", "reason": "unsupported_capability" if record.intent_family == "unsupported" else None},
    }


def _show(record: Any, canonical_label: dict[str, Any], out: TextIO) -> None:
    print(f"\nrecord_id: {record.id}\ntext: {record.text}", file=out)
    print(json.dumps({"canonical_label": canonical_label, "language": record.language_group,
                      "script": record.script_group, "semantic_language": record.semantic_language,
                      "input_script": record.input_script, "response_language": record.response_language,
                      "review_priority": record.review_priority, "review_reasons": record.review_reasons},
                     ensure_ascii=False, indent=2), file=out)


def _show_ai(proposal: AIReviewProposal, out: TextIO) -> None:
    print("\nAI PROPOSAL", file=out)
    print(json.dumps(proposal.to_dict(), ensure_ascii=False, indent=2), file=out)
    if proposal.mode in {"HUMAN_REVIEW", "SOURCE_CONFLICT"}:
        print("AI could not safely resolve this record; human review is required.", file=out)


def _save_human_choice(store: ReviewStore, record: Any, reviewer_id: str,
                       choice: str, input_fn: Callable[[str], str], out: TextIO,
                       proposed_label: dict[str, Any] | None = None) -> bool:
    if choice == "q":
        return False
    decision = {"a": "accept", "c": "correct", "r": "reject", "m": "ambiguous", "s": "skip"}.get(choice)
    if decision is None:
        print("Invalid choice.", file=out)
        return True
    reviewed_label = proposed_label if decision == "accept" else None
    if decision == "correct":
        while True:
            raw_label = input_fn("Complete reviewed_label JSON (single line, q to cancel): ")
            if raw_label.strip().lower() == "q":
                return True
            try:
                candidate_label = json.loads(raw_label)
                if not isinstance(candidate_label, dict):
                    raise ValueError("reviewed_label must be an object")
                ReviewDecision(record_id=record.id, reviewer_id=reviewer_id,
                               decision="correct", reviewed_label=candidate_label).validate()
                reviewed_label = candidate_label
                break
            except (json.JSONDecodeError, ValueError, DatasetValidationError) as exc:
                print(f"Invalid reviewed_label: {exc}", file=out)
    try:
        store.append(ReviewDecision(record_id=record.id, reviewer_id=reviewer_id,
                                    decision=decision, reviewed_label=reviewed_label,
                                    reviewed_at=datetime.now(timezone.utc).isoformat()))
    except DatasetValidationError as exc:
        print(f"Review rejected: {exc}", file=out)
    else:
        print(f"Saved {decision}. {len(store.records) - len(store.decisions)} pending; readiness={readiness(store.records, store.decisions)}", file=out)
    return True


def run_ai_cli(store: ReviewStore, reviewer_id: str, proposer: Any,
               input_fn: Callable[[str], str] = input, out: TextIO = sys.stdout) -> None:
    while store.pending:
        record = store.pending[0]
        canonical = store.canonical_label(record)
        _show(record, canonical, out)
        proposal = validate_ai_proposal(proposer.propose(record, canonical))
        _show_ai(proposal, out)
        can_accept = proposal.mode in {"AUTO_CONFIDENT", "AI_CORRECTION"} and proposal.proposed_label is not None
        prompt = "[a] Accept AI proposal [c] Correct manually [r] Reject [m] Ambiguous [s] Skip [q] Quit: " if can_accept else "[c] Correct manually [r] Reject [m] Ambiguous [s] Skip [q] Quit: "
        choice = input_fn(prompt).strip().lower()
        if choice == "a" and not can_accept:
            print("AI proposal is not safe to accept automatically; choose manual review.", file=out)
            continue
        if not _save_human_choice(store, record, reviewer_id, choice, input_fn, out,
                                  proposal.proposed_label if can_accept else None):
            return


def run_cli(store: ReviewStore, reviewer_id: str, input_fn: Callable[[str], str] = input,
            out: TextIO = sys.stdout) -> None:
    while store.pending:
        record = store.pending[0]
        _show(record, store.canonical_label(record), out)
        choice = input_fn("[a]ccept [c]orrect [r]eject [m]biguous [s]kip [q]uit: ").strip().lower()
        if choice == "q":
            return
        decision = {"a": "accept", "c": "correct", "r": "reject", "m": "ambiguous", "s": "skip"}.get(choice)
        if decision is None:
            print("Invalid choice.", file=out)
            continue
        reviewed_label = None
        if decision == "accept":
            reviewed_label = store.canonical_label(record)
        elif decision == "correct":
            while True:
                raw_label = input_fn("Complete reviewed_label JSON (single line, q to cancel): ")
                if raw_label.strip().lower() == "q":
                    reviewed_label = None
                    break
                try:
                    candidate_label = json.loads(raw_label)
                    if not isinstance(candidate_label, dict):
                        raise ValueError("reviewed_label must be an object")
                    ReviewDecision(record_id=record.id, reviewer_id=reviewer_id,
                                   decision="correct", reviewed_label=candidate_label).validate()
                    reviewed_label = candidate_label
                    break
                except (json.JSONDecodeError, ValueError, DatasetValidationError) as exc:
                    print(f"Invalid reviewed_label: {exc}", file=out)
            if reviewed_label is None:
                continue
        try:
            store.append(ReviewDecision(record_id=record.id, reviewer_id=reviewer_id,
                                        decision=decision, reviewed_label=reviewed_label,
                                        reviewed_at=datetime.now(timezone.utc).isoformat()))
        except DatasetValidationError as exc:
            print(f"Review rejected: {exc}", file=out)
            continue
        print(f"Saved {decision}. {len(store.records) - len(store.decisions)} pending; readiness={readiness(store.records, store.decisions)}", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    parser.add_argument("--ai-assist", action="store_true", help="show local AI proposals; human approval is still required")
    parser.add_argument("--ai-timeout", type=float, default=20.0)
    args = parser.parse_args(argv)
    try:
        store = ReviewStore(args.queue, args.decisions)
        if args.ai_assist:
            run_ai_cli(store, args.reviewer_id, OllamaReviewProposer(timeout=args.ai_timeout))
        else:
            run_cli(store, args.reviewer_id)
    except DatasetValidationError as exc:
        print(f"Review artifact invalid: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
