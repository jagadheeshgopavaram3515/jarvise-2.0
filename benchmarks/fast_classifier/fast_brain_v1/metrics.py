from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

from .schema import FastBrainRecord


def accuracy(expected: Iterable[str], predicted: Iterable[str]) -> float | None:
    pairs = list(zip(expected, predicted))
    return None if not pairs else sum(a == b for a, b in pairs) / len(pairs)


def binary_metrics(expected: Iterable[bool], predicted: Iterable[bool]) -> dict[str, float | None]:
    pairs = list(zip(expected, predicted))
    if not pairs:
        return {"precision": None, "recall": None, "f1": None}
    tp = sum(a and b for a, b in pairs); fp = sum((not a) and b for a, b in pairs); fn = sum(a and (not b) for a, b in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def class_counts(records: Iterable[FastBrainRecord]) -> dict[str, dict[str, int]]:
    records = list(records)
    return {field: dict(Counter(getattr(r, field) for r in records))
            for field in ("intent_family", "capability_family", "provenance", "language_group", "split")}

