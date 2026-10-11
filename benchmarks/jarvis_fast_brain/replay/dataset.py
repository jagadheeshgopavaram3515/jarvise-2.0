from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .schema import ReplayCase


def load_jsonl(path: str | Path) -> list[ReplayCase]:
    cases = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            cases.append(ReplayCase.from_mapping(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid replay case at {path}:{line_no}: {exc}") from exc
    validate_groups(cases)
    return cases


def validate_groups(cases: Iterable[ReplayCase]) -> None:
    seen: dict[str, str] = {}
    for case in cases:
        if case.id in seen:
            raise ValueError(f"duplicate case id: {case.id}")
        seen[case.id] = case.group


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
