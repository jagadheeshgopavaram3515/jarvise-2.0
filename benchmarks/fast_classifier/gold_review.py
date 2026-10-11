"""Offline gold-review gate and scorer. Never promotes historical/AI labels.

The real frozen source is loaded in an isolated package to avoid the production
orchestration initializer. No provider, logger, application or model is started.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import types

FIELDS = ("intent_family", "capability_family", "execution_path", "context_requirements",
          "clarification.required", "clarification.missing", "escalation.candidate",
          "escalation.reason", "uncertainty")


def baseline_modules():
    name = "_jarvis_gold_offline_baseline"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[2] / "assistant/orchestration/fast_brain")]
        sys.modules[name] = package
    return {part: importlib.import_module(name + "." + part)
            for part in ("contract", "validator", "adapter", "deterministic")}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def input_hash(record):
    return hashlib.sha256(canonical({key: record[key] for key in
        ("review_id", "utterance", "context", "capabilities", "language", "review_requirements")}).encode()).hexdigest()


def field_value(label, field):
    value = label
    for part in field.split("."):
        value = value[part]
    return value


def verify_inputs(records, manifest):
    expected = manifest["review_inputs"]
    if len(records) != len(expected) or {r["review_id"] for r in records} != set(expected):
        raise ValueError("review population differs from frozen manifest")
    for record in records:
        if input_hash(record) != expected[record["review_id"]]:
            raise ValueError("blinded input changed: " + record["review_id"])
        if record["rubric_sha256"] != manifest["rubric_sha256"]:
            raise ValueError("review rubric differs from frozen manifest")


def validate_label(label):
    """Actual contract plus explicit annotation shape, without changing runtime."""
    baseline_modules()["contract"].FastBrainDecision.from_mapping(label)
    expected = {"intent_family", "capability_family", "execution_path", "context_requirements",
                "clarification", "escalation", "uncertainty"}
    if set(label) != expected:
        raise ValueError("annotation must contain exactly seven decision fields")
    if set(label["clarification"]) != {"required", "missing"} or set(label["escalation"]) != {"candidate", "reason"}:
        raise ValueError("annotation nested fields must match the frozen shape")
    if type(label["clarification"]["required"]) is not bool or type(label["escalation"]["candidate"]) is not bool:
        raise ValueError("annotation flags must be JSON booleans")
    if any(not isinstance(field_value(label, field), list) for field in
           ("context_requirements", "clarification.missing")):
        raise ValueError("annotation context and missing fields must be arrays")


def check_judgment(judgment, record, *, initial=True):
    if judgment.get("judgment") != "labeled":
        raise ValueError("judgment is ambiguous, unresolved or pending")
    if judgment.get("review_kind") != "independent" or judgment.get("reviewer_type") != "human":
        raise ValueError("self/AI/historical review is not independent human gold")
    if not isinstance(judgment.get("reviewer_id"), str) or not judgment["reviewer_id"].strip():
        raise ValueError("anonymized reviewer identity required")
    if initial and judgment.get("prediction_blinded") is not True:
        raise ValueError("initial independent judgment was not prediction-blinded")
    if judgment.get("input_sha256") != input_hash(record):
        raise ValueError("judgment input hash mismatch")
    requirements = record["review_requirements"]
    for key in ("languages", "scripts"):
        if not set(requirements["required_" + key]).issubset(set(judgment.get("qualification", {}).get(key, []))):
            raise ValueError("qualified language/script review missing")
    if requirements.get("transliteration_expertise_required") and judgment.get("qualification", {}).get("transliteration") is not True:
        raise ValueError("transliteration expertise not attested")
    if judgment.get("rubric_sha256") != record["rubric_sha256"]:
        raise ValueError("review rubric version mismatch")
    rationale = judgment.get("field_rationale", {})
    if set(rationale) != set(FIELDS) or any(not isinstance(v, str) or not v.strip() for v in rationale.values()):
        raise ValueError("rationale required for each of nine decision dimensions")
    timestamp = dt.datetime.fromisoformat(judgment.get("timestamp", "").replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("review timestamp must include timezone")
    validate_label(judgment.get("label"))


def verified_gold(record):
    """Fail closed on missing expertise, self-review or unadjudicated disagreement."""
    try:
        judgments = record.get("judgments", [])
        if not judgments:
            raise ValueError("awaiting independent human judgment")
        for judgment in judgments:
            check_judgment(judgment, record)
        if len({j["reviewer_id"] for j in judgments}) != len(judgments):
            raise ValueError("duplicate reviewer identities")
        labels = {canonical(j["label"]) for j in judgments}
        if len(labels) == 1:
            return judgments[0]["label"], None
        resolution = record.get("adjudication")
        if not resolution:
            raise ValueError("reviewer disagreement requires adjudication")
        check_judgment(resolution, record, initial=False)
        if resolution["reviewer_id"] in {j["reviewer_id"] for j in judgments}:
            raise ValueError("disagreement requires a separate adjudicator")
        if not isinstance(resolution.get("notes"), str) or not resolution["notes"].strip():
            raise ValueError("adjudication explanation required")
        return resolution["label"], None
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return None, str(error)


def score(records, predictions):
    """Failures count as incorrect; unresolved reference labels are excluded."""
    if len({r["review_id"] for r in records}) != len(records):
        raise ValueError("duplicate review IDs")
    eligible = [(record, *verified_gold(record)) for record in records]
    gold = [(record, label) for record, label, error in eligible if error is None]
    fields = {field: 0 for field in FIELDS}
    confusion = {field: {} for field in FIELDS}
    comparisons = []
    failures = exact = valid = 0
    for record, label in gold:
        prediction = predictions.get(record["review_id"])
        try:
            validate_label(prediction)
            valid += 1
            matches = {field: canonical(field_value(label, field)) == canonical(field_value(prediction, field)) for field in FIELDS}
        except (ValueError, TypeError, KeyError, AttributeError):
            failures += 1
            matches = dict.fromkeys(FIELDS, False)
        exact += all(matches.values())
        for field, equal in matches.items():
            fields[field] += equal
            a = canonical(field_value(label, field))
            b = canonical(field_value(prediction, field)) if prediction is not None and valid_label(prediction) else "__FAILED_PREDICTION__"
            row = confusion[field].setdefault(a, {})
            row[b] = row.get(b, 0) + 1
        comparisons.append({"review_id": record["review_id"], "gold_label": label,
                            "prediction": prediction, "field_correctness": matches,
                            "exact_match": all(matches.values())})
    def metric(n):
        return {"status": "MEASURED" if gold else "NOT_MEASURED", "numerator": n if gold else None,
                "denominator": len(gold), "value": n / len(gold) if gold else None}
    return {"selected": len(records), "independently_adjudicated": len(gold),
            "unresolved": [{"review_id": r["review_id"], "reason": e} for r, _, e in eligible if e],
            "exact_match": metric(exact), "per_field": {field: metric(n) for field, n in fields.items()},
            "schema_valid": metric(valid), "execution_failure": metric(failures),
            "confusion": confusion, "comparisons": comparisons}


def valid_label(value):
    try:
        validate_label(value)
        return True
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def evaluate_records(records):
    """Execute the selected real baseline only after independent gold is eligible."""
    modules = baseline_modules()
    backend = modules["deterministic"].DeterministicFastBrain()
    predictions = {}
    executions = {}
    for record in records:
        if verified_gold(record)[1] is not None:
            continue
        started = time.perf_counter_ns()
        stage = "backend"
        try:
            request = modules["adapter"].FastBrainRequest(record["utterance"], record["context"],
                tuple(cap["name"] for cap in record["capabilities"]))
            decision = backend.decide(request)
            stage = "schema_validation"
            modules["validator"].validate_decision(decision)
            predictions[record["review_id"]] = decision.to_dict()
            executions[record["review_id"]] = {"status": "success", "error": None}
        except Exception as error:
            predictions[record["review_id"]] = None
            executions[record["review_id"]] = {"status": "failure", "stage": stage,
                "error": {"type": type(error).__name__, "message": str(error)}}
        executions[record["review_id"]]["elapsed_ms"] = (time.perf_counter_ns() - started) / 1_000_000
    metrics = score(records, predictions)
    metrics["prediction_executions"] = executions
    return predictions, metrics


def main():
    parser = argparse.ArgumentParser(description="Offline independently reviewed gold benchmark; no network/model.")
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--manifest", required=True, help="frozen steward manifest")
    parser.add_argument("--out", required=True, help="new output directory; existing paths are refused")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.reviews).read_text(encoding="utf-8").splitlines() if line.strip()]
    verify_inputs(records, json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    required_sources = ["assistant/orchestration/fast_brain/" + part + ".py"
                        for part in ("contract", "validator", "adapter", "deterministic")]
    for source in required_sources:
        if hashlib.sha256((root / source).read_bytes()).hexdigest() != manifest["source_hashes"][source]:
            raise ValueError("baseline source changed since benchmark plan: " + source)
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=False)
    predictions, metrics = evaluate_records(records)
    metrics["baseline_source_hashes"] = {source: manifest["source_hashes"][source] for source in required_sources}
    metrics["review_file_sha256"] = hashlib.sha256(Path(args.reviews).read_bytes()).hexdigest()
    metrics["review_rubric_sha256"] = manifest["rubric_sha256"]
    metrics["review_process_limit"] = "Human independence/qualifications are attestations requiring steward verification, not authenticated by code."
    (destination / "gold-labels.jsonl").write_text("".join(json.dumps({"review_id": record["review_id"],
        "gold_label": verified_gold(record)[0], "judgments": record["judgments"],
        "adjudication": record.get("adjudication")}, ensure_ascii=False) + "\n"
        for record in records if verified_gold(record)[1] is None), encoding="utf-8")
    (destination / "comparisons.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
        for row in metrics["comparisons"]), encoding="utf-8")
    (destination / "predictions.jsonl").write_text("".join(json.dumps({"review_id": key, "prediction": value}, ensure_ascii=False) + "\n" for key, value in predictions.items()), encoding="utf-8")
    (destination / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"independently_adjudicated": metrics["independently_adjudicated"], "accuracy": metrics["exact_match"]}))


if __name__ == "__main__":
    main()
