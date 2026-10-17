"""Offline Reviewed-Label Evaluation Engine for JARVIS Shadow Mode.

Joins raw shadow events against independently reviewed human ground truth.
Strictly separates model agreement from true accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

FIELDS = (
    "intent",
    "execution_path",
    "request_modes",
    "selected_capability",
    "needs_clarification",
    "needs_escalation",
    "missing_information",
    "context_references",
)


def evaluate_reviewed_shadow_data(
    events_path: str | Path,
    reviewed_labels_path: str | Path,
) -> dict[str, Any]:
    """Join shadow events and reviewed labels by request_id and compute accuracy metrics.

    If reviewed_labels_path does not exist or has 0 valid reviews, returns
    status="insufficient_reviewed_samples" without synthesizing fake metrics.
    """
    events_file = Path(events_path)
    reviews_file = Path(reviewed_labels_path)

    events_by_id: dict[str, dict[str, Any]] = {}
    if events_file.exists():
        for line in events_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                req_id = ev.get("request_id")
                if req_id:
                    events_by_id[req_id] = ev
            except Exception:
                continue

    reviews_by_id: dict[str, dict[str, Any]] = {}
    if reviews_file.exists():
        for line in reviews_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rev = json.loads(line)
                req_id = rev.get("request_id")
                status = rev.get("review_status", "").lower()
                if req_id and status in ("reviewed", "verified", "gold"):
                    reviews_by_id[req_id] = rev
            except Exception:
                continue

    total_events = len(events_by_id)
    reviewed_count = len(reviews_by_id)

    if reviewed_count == 0:
        return {
            "status": "insufficient_reviewed_samples",
            "message": "No independently human-reviewed labels available for accuracy calculation.",
            "total_observed_events": total_events,
            "reviewed_records_count": 0,
            "unreviewed_records_excluded": total_events,
            "accuracy_evaluation_available": False,
        }

    # Match events with reviews
    matched_ids = [rid for rid in reviews_by_id if rid in events_by_id]
    if not matched_ids:
        return {
            "status": "insufficient_reviewed_samples",
            "message": "Reviewed labels provided do not match any captured shadow event request IDs.",
            "total_observed_events": total_events,
            "reviewed_records_count": reviewed_count,
            "matched_records_count": 0,
            "unreviewed_records_excluded": total_events,
            "accuracy_evaluation_available": False,
        }

    # Metric tracking
    n = len(matched_ids)
    prod_exact = 0
    prod_cap = 0
    v8_exact = 0
    v8_cap = 0

    v8_field_counts = {f: 0 for f in FIELDS}
    prod_field_counts = {f: 0 for f in FIELDS}

    v8_schema_invalid = 0
    prod_schema_invalid = 0

    wins_v8 = 0
    wins_prod = 0
    ties = 0

    # Language breakdowns
    lang_stats: dict[str, dict[str, Any]] = {}

    for req_id in matched_ids:
        ev = events_by_id[req_id]
        rev = reviews_by_id[req_id]
        expected = rev.get("expected_output") or {}

        v8_pred = (ev.get("v8") or {}).get("prediction") or {}
        v8_valid = (ev.get("v8") or {}).get("schema_valid", False)
        if not v8_valid:
            v8_schema_invalid += 1

        # Production mapping
        prod_route = (ev.get("production") or {}).get("route", "")
        prod_mapped = _map_production_route_to_contract(prod_route)

        lang_cat = (ev.get("language_classification") or {}).get("primary_category", "unknown")
        if lang_cat not in lang_stats:
            lang_stats[lang_cat] = {
                "total": 0,
                "v8_exact": 0,
                "v8_cap": 0,
                "prod_exact": 0,
                "prod_cap": 0,
            }
        lang_stats[lang_cat]["total"] += 1

        # Check V8 accuracy
        v8_mismatches = _check_mismatches(v8_pred, expected)
        is_v8_exact = len(v8_mismatches) == 0 and v8_valid
        is_v8_cap = (v8_pred.get("selected_capability") == expected.get("selected_capability"))

        if is_v8_exact:
            v8_exact += 1
            lang_stats[lang_cat]["v8_exact"] += 1
        if is_v8_cap:
            v8_cap += 1
            lang_stats[lang_cat]["v8_cap"] += 1

        for f in FIELDS:
            if f not in v8_mismatches:
                v8_field_counts[f] += 1

        # Check Production accuracy
        prod_mismatches = _check_mismatches(prod_mapped, expected)
        is_prod_exact = len(prod_mismatches) == 0
        is_prod_cap = (prod_mapped.get("selected_capability") == expected.get("selected_capability"))

        if is_prod_exact:
            prod_exact += 1
            lang_stats[lang_cat]["prod_exact"] += 1
        if is_prod_cap:
            prod_cap += 1
            lang_stats[lang_cat]["prod_cap"] += 1

        for f in FIELDS:
            if f not in prod_mismatches:
                prod_field_counts[f] += 1

        # Paired wins / losses
        if is_v8_exact and not is_prod_exact:
            wins_v8 += 1
        elif is_prod_exact and not is_v8_exact:
            wins_prod += 1
        else:
            ties += 1

    return {
        "status": "evaluated",
        "accuracy_evaluation_available": True,
        "sample_counts": {
            "total_observed_events": total_events,
            "reviewed_records_count": reviewed_count,
            "evaluated_samples": n,
            "unreviewed_records_excluded": total_events - n,
        },
        "overall_metrics": {
            "production": {
                "strict_exact_matches": prod_exact,
                "strict_exact_accuracy_pct": round(prod_exact / n * 100.0, 2),
                "capability_top1_matches": prod_cap,
                "capability_top1_accuracy_pct": round(prod_cap / n * 100.0, 2),
                "schema_invalid_count": prod_schema_invalid,
                "per_field_accuracy": {
                    f: {"count": prod_field_counts[f], "pct": round(prod_field_counts[f] / n * 100.0, 2)}
                    for f in FIELDS
                },
            },
            "v8": {
                "strict_exact_matches": v8_exact,
                "strict_exact_accuracy_pct": round(v8_exact / n * 100.0, 2),
                "capability_top1_matches": v8_cap,
                "capability_top1_accuracy_pct": round(v8_cap / n * 100.0, 2),
                "schema_invalid_count": v8_schema_invalid,
                "per_field_accuracy": {
                    f: {"count": v8_field_counts[f], "pct": round(v8_field_counts[f] / n * 100.0, 2)}
                    for f in FIELDS
                },
            },
            "paired_comparison": {
                "v8_wins": wins_v8,
                "production_wins": wins_prod,
                "ties": ties,
                "net_v8_gain": wins_v8 - wins_prod,
            },
        },
        "language_breakdown": lang_stats,
    }


def _check_mismatches(pred: Mapping[str, Any], exp: Mapping[str, Any]) -> list[str]:
    mismatches = []
    for f in FIELDS:
        if f == "request_modes":
            p_modes = sorted(list(pred.get(f) or []))
            e_modes = sorted(list(exp.get(f) or []))
            if p_modes != e_modes:
                mismatches.append(f)
        elif f == "context_references":
            p_refs = pred.get(f) or []
            e_refs = exp.get(f) or []
            if p_refs != e_refs:
                mismatches.append(f)
        elif f == "missing_information":
            p_miss = pred.get(f) or []
            e_miss = exp.get(f) or []
            if p_miss != e_miss:
                mismatches.append(f)
        else:
            if pred.get(f) != exp.get(f):
                mismatches.append(f)
    return mismatches


def _map_production_route_to_contract(route: str) -> dict[str, Any]:
    r = (route or "").upper()
    if "TASK" in r:
        return {
            "intent": "execute_task",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": None,
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": [],
        }
    return {
        "intent": "respond",
        "execution_path": "direct_response",
        "request_modes": ["conversational"],
        "selected_capability": None,
        "needs_clarification": False,
        "needs_escalation": False,
        "missing_information": [],
        "context_references": [],
    }

