from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .schema import DatasetValidationError, FastBrainRecord


def load_jsonl(path: str | Path, *, allow_text_conflicts: bool = False) -> list[FastBrainRecord]:
    records: list[FastBrainRecord] = []
    seen: set[str] = set()
    seen_text: dict[str, str] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = FastBrainRecord.from_mapping(json.loads(line))
        except (json.JSONDecodeError, DatasetValidationError) as exc:
            raise DatasetValidationError(f"line {line_number}: {exc}") from exc
        if record.id in seen:
            raise DatasetValidationError(f"duplicate id: {record.id}")
        if not allow_text_conflicts and record.text in seen_text and seen_text[record.text] != record.group_id:
            raise DatasetValidationError(f"duplicate text crosses groups: {record.id}")
        seen.add(record.id)
        seen_text[record.text] = record.group_id
        records.append(record)
    return records


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance_counts(records: Iterable[FastBrainRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.provenance] = counts.get(record.provenance, 0) + 1
    return dict(sorted(counts.items()))


def analysis(records: Iterable[FastBrainRecord]) -> dict:
    records = list(records)
    from collections import Counter
    def count(field):
        return dict(sorted(Counter(getattr(r, field) for r in records).items()))
    return {
        "records": len(records), "provenance": provenance_counts(records),
        "review_status": count("review_status"), "intent": count("intent_family"),
        "capability": count("capability_family"), "language": count("language_group"),
        "script": count("script_group"), "response_language": count("response_language"),
        "clarification": count("clarification_required"),
        "missing_information": {label: sum(label in r.missing_information for r in records)
                                 for label in sorted({x for r in records for x in r.missing_information})},
        "groups": len({r.group_id for r in records}),
        "hard_negative_groups": len({r.hard_negative_group for r in records if r.hard_negative_group}),
        "stt_variants": sum(r.noise_type == "stt_variant" for r in records),
        "context_resolvable": sum(r.context_resolvable is True for r in records),
    }

