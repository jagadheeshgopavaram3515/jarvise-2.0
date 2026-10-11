"""Consolidate historical classifier data into Fast Brain v1 candidates.

Historical records remain pending review and retain their source metadata.
This builder never labels records human-reviewed and never generates examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from .fast_brain_v1.dataset import analysis, sha256_file
from .fast_brain_v1.schema import DatasetValidationError, FastBrainRecord
from .fast_brain_v1.split import grouped_split, manifest, manifest_hash, validate_group_leakage


def language_meta(raw: dict) -> tuple[str, str, str]:
    language = str(raw.get("language", "en"))
    if language.startswith("te"):
        semantic = "telugu"
    elif language.startswith("hi"):
        semantic = "hindi"
    elif language.startswith("ta"):
        semantic = "tamil"
    elif "mixed" in language:
        semantic = "code_switch"
    else:
        semantic = "english"
    script = "telugu" if semantic == "telugu" and "Latn" not in language else (
        "devanagari" if semantic == "hindi" and "Latn" not in language else (
            "tamil" if semantic == "tamil" and "Latn" not in language else (
                "mixed" if "mixed" in language else "latin")))
    return semantic, script, semantic


def capability(raw: dict, categories: list[str]) -> str:
    selected = str(raw.get("expected", {}).get("selected_capability") or "")
    text = " ".join(categories).lower()
    if selected in {"google_search", "youtube_search", "open_url", "read_page", "get_page_title"} or "browser" in text:
        return "research" if "research" in text else "browser"
    if selected in {"read_file", "open_file", "find_file", "search_files", "list_directory"} or "file" in text:
        return "project" if "project" in text or "code" in text else "file"
    if selected in {"weather"}: return "weather"
    if selected in {"location"}: return "location"
    if "memory" in text or raw.get("expected", {}).get("intent") == "recall": return "memory"
    if "application" in text or selected in {"open_app", "close_app"}: return "application"
    if "system" in text: return "system"
    if "research" in text: return "research"
    return "none"


def intent(raw: dict, categories: list[str], cap: str) -> str:
    old = str(raw.get("expected", {}).get("intent") or "")
    text = " ".join(categories).lower()
    if old == "recall" or "memory" in text: return "memory"
    if "unsupported" in old or "unsupported" in text: return "unsupported"
    if "clarif" in text or raw.get("expected", {}).get("needs_clarification"): return "clarification"
    if "multi" in text or "multi_step" in str(raw.get("expected", {}).get("execution_path")): return "multi_step_task"
    if "research" in text or raw.get("expected", {}).get("execution_path") == "research": return "research"
    if cap != "none":
        return "local_task" if cap in {"file", "project", "application", "browser"} else "deterministic_action"
    if old in {"respond", "conversation"}:
        return "information" if "knowledge" in text or "information" in text else "conversation"
    return "information"


def normalize(raw: dict, source: str) -> FastBrainRecord:
    expected = raw.get("expected", {})
    categories = list(raw.get("categories", []))
    cap = capability(raw, categories)
    semantic, script, response = language_meta(raw)
    missing = tuple(str(x) for x in expected.get("missing_information", []))
    clarify = bool(expected.get("needs_clarification"))
    if clarify and not missing:
        missing = ("reference",)
    if not clarify:
        missing = ()
    original_id = str(raw.get("id", ""))
    split = {"train": "train", "dev": "development", "test": "sealed_candidate"}.get(raw.get("split"), "train")
    record_id = hashlib.sha256(f"{source}:{original_id}".encode()).hexdigest()[:16]
    return FastBrainRecord(
        id=f"hist-{record_id}", text=str(raw.get("utterance", "")), intent_family=intent(raw, categories, cap),
        capability_family=cap, clarification_required=clarify, missing_information=missing,
        provenance="historical_synthetic", group_id=str(raw.get("group_id") or original_id), split=split,
        review_status="pending_review", language_group=semantic, script_group=script,
        semantic_language=semantic, input_script=script, response_language=response,
        noise_type="clean",
        context_resolvable=bool(raw.get("context")), original_dataset=source, original_id=original_id,
        original_label=json.dumps(expected, ensure_ascii=False, sort_keys=True),
        original_provenance=str(raw.get("provenance", "unknown")),
        original_review_status=str(raw.get("review_status", raw.get("annotation_status", "unknown"))),
        rationale=str(raw.get("notes", "")), generation_reason="historical consolidation",
        source_type="historical",
    )


def write_jsonl(path: Path, records: list[FastBrainRecord]) -> None:
    path.write_text("".join(json.dumps(r.__dict__, ensure_ascii=False, sort_keys=True) + "\n" for r in records), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "fast_brain_v1"))
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    records, rejected, discovered = [], [], []
    seen = set()
    for source in args.source:
        source_path = Path(source); discovered.append({"path": str(source_path), "sha256": sha256_file(source_path)})
        for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip(): continue
            raw = json.loads(line)
            try:
                record = normalize(raw, str(source_path))
                if record.id in seen: raise DatasetValidationError("duplicate canonical id")
                record.validate(); seen.add(record.id); records.append(record)
            except Exception as exc:
                rejected.append({"source": str(source_path), "line": line_number, "original_id": raw.get("id"), "rejection_reason": str(exc)})
    records = grouped_split(records, seed=42)
    validate_group_leakage(records)
    write_jsonl(output / "dataset.jsonl", records)
    write_jsonl(output / "review_queue.jsonl", records)
    (output / "rejected_records.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rejected), encoding="utf-8")
    report = analysis(records)
    report.update({"historical_sources": discovered, "records_discovered": sum(1 for _ in records) + len(rejected),
                   "records_imported": len(records), "records_quarantined": len(rejected),
                   "records_newly_generated": 0, "dataset_sha256": sha256_file(output / "dataset.jsonl"),
                   "human_review_required": len(records), "model_downloaded": False, "model_trained": False,
                   "network_used": False, "production_routing_changed": False})
    (output / "dataset_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    split = manifest(records)
    (output / "split_manifest.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
