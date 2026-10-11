from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict
from typing import Iterable

from .schema import DatasetValidationError, FastBrainRecord, SPLITS


def validate_group_leakage(records: Iterable[FastBrainRecord]) -> None:
    groups: dict[str, str] = {}
    for record in records:
        previous = groups.setdefault(record.group_id, record.split)
        if previous != record.split:
            raise DatasetValidationError(f"group crosses splits: {record.group_id}")


def grouped_split(records: list[FastBrainRecord], *, seed: int = 42,
                  ratios: tuple[float, float, float] = (.70, .15, .15)) -> list[FastBrainRecord]:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to one")
    groups: dict[str, list[FastBrainRecord]] = defaultdict(list)
    for record in records:
        groups[record.group_id].append(record)
    names = list(groups)
    random.Random(seed).shuffle(names)
    targets = [len(records) * ratio for ratio in ratios]
    buckets: list[list[FastBrainRecord]] = [[], [], []]
    for name in names:
        index = min(range(3), key=lambda i: len(buckets[i]) / max(targets[i], 1))
        buckets[index].extend(groups[name])
    split_names = ("train", "development", "sealed_candidate")
    return [FastBrainRecord(**{**asdict(record), "split": split_names[i]})
            for i, bucket in enumerate(buckets) for record in bucket]


def manifest(records: Iterable[FastBrainRecord], *, seed: int = 42) -> dict:
    records = list(records)
    validate_group_leakage(records)
    return {
        "version": "fast-brain-v1",
        "seed": seed,
        "records": len(records),
        "splits": {split: [r.id for r in records if r.split == split] for split in SPLITS},
        "groups": {split: sorted({r.group_id for r in records if r.split == split}) for split in SPLITS},
    }


def manifest_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

