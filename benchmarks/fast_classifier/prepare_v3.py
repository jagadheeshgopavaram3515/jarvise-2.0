"""Derive the reviewed Phase 3C dataset without modifying v2 artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "benchmarks" / "fast_classifier" / "v2" / "dataset.jsonl"
OUT_DIR = ROOT / "benchmarks" / "fast_classifier" / "v3"
OUT = OUT_DIR / "dataset.jsonl"
MANIFEST = OUT_DIR / "split_manifest.json"

CAPS = {
    "open_app": ("Open an allow-listed desktop application: Chrome, Edge, VS Code, Notepad, Calculator, or Explorer.", ["app"]),
    "close_app": ("Close an allow-listed running desktop application.", ["app"]),
    "find_file": ("Find a file by filename in approved locations.", ["filename"]),
    "open_file": ("Open a resolved file path with its default application.", ["path"]),
    "open_folder": ("Open a resolved folder path in Windows Explorer.", ["path"]),
    "google_search": ("Search the web for a supplied query.", ["query"]),
    "read_page": ("Read text from a supplied web page URL.", ["url"]),
    "open_url": ("Open a supplied URL in the default browser.", ["url"]),
}


def _cap(name: str) -> dict[str, Any]:
    description, required = CAPS[name]
    return {"name": name, "description": description, "required_arguments": required}


def _expected(path: str, capability: str | None = None, *, resolved: str | None = None) -> dict[str, Any]:
    modes = {
        "direct_response": ["conversational"], "memory_recall": ["memory"],
        "research": ["research", "action"], "single_action": ["action"],
        "multi_step_task": ["research", "action"], "clarify": ["action"],
        "escalate": ["action"],
    }[path]
    intent = {
        "direct_response": "respond", "memory_recall": "recall", "research": "research",
        "multi_step_task": "research", "clarify": "clarify_request",
        "escalate": "unsupported_request",
    }.get(path, f"use:{capability}")
    reference = []
    missing = []
    if path == "memory_recall":
        reference = [{"reference": "prior discussion", "kind": "prior_conversation", "resolved_value": resolved}]
    if path == "clarify":
        reference = [{"reference": "it", "kind": "item", "resolved_value": None}]
        missing = [{"name": "target", "description": "The requested target is not identifiable"}]
    return {
        "intent": intent, "execution_path": path, "request_modes": modes,
        "selected_capability": capability, "context_references": reference,
        "missing_information": missing, "needs_clarification": path == "clarify",
        "needs_escalation": path == "escalate",
    }


SCENARIOS = [
    ("train_casual_checkin", "train", "direct_response", None, [], {}, {
        "en": "I had a long day and could use a quick chat.",
        "te-Latn": "ivala chala tiring ga undi, konchem matladadama",
        "hi-Latn": "aaj din lamba tha, thodi baat karte hain",
        "ta-Latn": "inniku romba tiring-a irundhuchu, konjam pesalama",
    }),
    ("train_memory_meeting", "train", "memory_recall", None, [], {"recent_topics": [{"topic": "design meeting", "decision": "ship the API first"}]}, {
        "en": "What did we settle on in the design meeting?",
        "te": "డిజైన్ మీటింగ్‌లో మనం ఏం నిర్ణయించాం?",
        "hi": "डिज़ाइन मीटिंग में हमने क्या तय किया था?",
        "ta": "வடிவமைப்பு கூட்டத்தில் நாம் என்ன முடிவு செய்தோம்?",
    }),
    ("train_research_market", "train", "research", "google_search", ["google_search", "find_file"], {}, {
        "en": "Check today's rupee to dollar exchange rate.",
        "mixed-en-te": "today rupee dollar exchange rate entha check cheyyi",
        "mixed-en-hi": "aaj ka rupee dollar exchange rate check karo",
        "mixed-en-ta": "today rupee dollar exchange rate enna check pannu",
    }),
    ("train_open_known_url", "train", "single_action", "open_url", ["open_url", "google_search"], {}, {
        "en": "Open https://docs.python.org in my browser.",
        "te-Latn": "https://docs.python.org browser lo open cheyyi",
        "hi-Latn": "https://docs.python.org browser mein kholo",
        "ta-Latn": "https://docs.python.org browser-la open pannu",
    }),
    ("train_multistep_comparison", "train", "multi_step_task", "google_search", ["google_search", "read_page"], {}, {
        "en": "Find the current Rust and Go releases, then compare their highlights.",
        "mixed-en-te": "current Rust Go releases find chesi highlights compare cheyyi",
        "mixed-en-hi": "current Rust aur Go releases dhoondh kar highlights compare karo",
        "mixed-en-ta": "current Rust Go releases thedi highlights compare pannu",
    }),
    ("train_clarify_missing_target", "train", "clarify", None, ["open_file", "open_url"], {}, {
        "en": "Open the thing I mentioned.", "te-Latn": "nenu cheppina danini open cheyyi",
        "hi-Latn": "jo maine kaha tha use kholo", "ta-Latn": "naan sonnatha open pannu",
    }),
    ("train_escalate_calendar", "train", "escalate", None, ["open_app", "find_file"], {}, {
        "en": "Schedule a calendar meeting with the supplier.",
        "te-Latn": "supplier tho calendar meeting schedule cheyyi",
        "hi-Latn": "supplier ke saath calendar meeting schedule karo",
        "ta-Latn": "supplier-oda calendar meeting schedule pannu",
    }),
    ("train_escalate_lights", "train", "escalate", None, ["open_app", "google_search"], {}, {
        "en": "Turn off the kitchen lights.", "te-Latn": "kitchen lights off cheyyi",
        "hi-Latn": "kitchen lights band karo", "ta-Latn": "kitchen lights-ai off pannu",
    }),
    ("train_escalate_email", "train", "escalate", None, ["open_file", "google_search"], {}, {
        "en": "Email the invoice to the accountant.", "te-Latn": "invoice accountant ki email cheyyi",
        "hi-Latn": "invoice accountant ko email karo", "ta-Latn": "invoice-ai accountant-ku email pannu",
    }),
    ("train_escalate_temperature", "train", "escalate", None, ["open_app", "google_search"], {}, {
        "en": "Lower the office thermostat.", "te-Latn": "office thermostat tagginchu",
        "hi-Latn": "office thermostat kam karo", "ta-Latn": "office thermostat-ai kurai",
    }),
    ("train_close_editor", "train", "single_action", "close_app", ["close_app", "open_app"], {}, {
        "en": "Close Notepad.", "te-Latn": "Notepad close cheyyi",
        "hi-Latn": "Notepad band karo", "ta-Latn": "Notepad-ai close pannu",
    }),
    ("train_open_downloads", "train", "single_action", "open_folder", ["open_folder", "find_file"], {}, {
        "en": "Open my Downloads folder.", "te-Latn": "Downloads folder open cheyyi",
        "hi-Latn": "Downloads folder kholo", "ta-Latn": "Downloads folder-ai open pannu",
    }),
    ("train_find_resume", "train", "single_action", "find_file", ["find_file", "google_search"], {}, {
        "en": "Find my resume file.", "te-Latn": "naa resume file vetuku",
        "hi-Latn": "mera resume file dhoondo", "ta-Latn": "en resume file-ai thedu",
    }),
    ("train_clarify_search", "train", "clarify", None, ["google_search", "read_page"], {}, {
        "en": "Search for that topic.", "te-Latn": "aa topic search cheyyi",
        "hi-Latn": "us topic ko search karo", "ta-Latn": "antha topic-ai search pannu",
    }),
    ("dev_stable_knowledge", "dev", "direct_response", None, [], {}, {
        "en": "Why do leaves change colour?", "te": "ఆకుల రంగు ఎందుకు మారుతుంది?",
        "hi": "पत्तों का रंग क्यों बदलता है?", "ta": "இலைகளின் நிறம் ஏன் மாறுகிறது?",
    }),
    ("dev_memory_project", "dev", "memory_recall", None, [], {"projects": [{"name": "Orion", "decision": "use a local cache"}]}, {
        "en": "Remind me what we chose for the Orion project.",
        "te-Latn": "Orion project ki em choose chesamo gurthu cheyyi",
        "hi-Latn": "Orion project ke liye kya chuna tha yaad dilao",
        "ta-Latn": "Orion project-ku enna choose pannom nyabagam pannu",
    }),
    ("dev_research_transit", "dev", "research", "google_search", ["google_search", "open_app"], {}, {
        "en": "Find the current metro service status.",
        "mixed-en-te": "current metro service status vetuku",
        "mixed-en-hi": "current metro service status dhoondo",
        "mixed-en-ta": "current metro service status thedu",
    }),
    ("dev_close_browser", "dev", "single_action", "close_app", ["close_app", "open_app"], {}, {
        "en": "Close Chrome for me.", "te": "క్రోమ్‌ను మూసివేయి.",
        "hi": "क्रोम बंद कर दो।", "ta": "குரோமை மூடு.",
    }),
    ("dev_multistep_guides", "dev", "multi_step_task", "google_search", ["google_search", "read_page"], {}, {
        "en": "Locate two official accessibility guides, read them, and summarize the differences.",
        "mixed-en-te": "two official accessibility guides find chesi differences summarize cheyyi",
        "mixed-en-hi": "do official accessibility guides dhoondh kar differences summarize karo",
        "mixed-en-ta": "two official accessibility guides thedi differences summarize pannu",
    }),
    ("dev_clarify_unresolved", "dev", "clarify", None, ["open_file", "open_folder"], {}, {
        "en": "Bring it up again.", "te-Latn": "adi malli open cheyyi",
        "hi-Latn": "use phir se kholo", "ta-Latn": "atha marubadiyum open pannu",
    }),
    ("dev_escalate_thermostat", "dev", "escalate", None, ["open_app", "google_search"], {}, {
        "en": "Set the bedroom thermostat to twenty degrees.",
        "te-Latn": "bedroom thermostat twenty degrees ki set cheyyi",
        "hi-Latn": "bedroom thermostat ko twenty degrees par set karo",
        "ta-Latn": "bedroom thermostat-ai twenty degrees-ku set pannu",
    }),
]


def _new_records() -> list[dict[str, Any]]:
    rows = []
    for group, split, path, selected, cap_names, context, phrases in SCENARIOS:
        for language, utterance in phrases.items():
            resolved = "ship the API first" if group == "train_memory_meeting" else "use a local cache"
            rows.append({
                "schema_version": "3.0", "id": f"{group}-{language}", "split": split,
                "utterance": utterance, "context": context,
                "capabilities": [_cap(name) for name in cap_names],
                "expected": _expected(path, selected, resolved=resolved), "language": language,
                "categories": [path, "phase3c_targeted", "native_script" if language in {"te", "hi", "ta"} else "romanized" if language.endswith("Latn") else "code_switching" if language.startswith("mixed-") else "english"],
                "annotation_status": "synthetic_candidate", "source": "phase3c_targeted_manual_authoring",
                "notes": "Synthetic Phase 3C coverage candidate; assistant-reviewed, not human-reviewed.",
                "group_id": group, "provenance": "phase3c_targeted_synthetic",
                "review_status": "assistant_reviewed_pending_human_review",
            })
    return rows


def _derive_v2() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, changes = [], []
    for line in V2.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        item = json.loads(line); item["schema_version"] = "3.0"
        item["derived_from_v2_id"] = item["id"]
        item["review_status"] = "assistant_reviewed_pending_human_review"
        for cap in item["capabilities"]:
            if cap["name"] == "find_file" and cap.get("required_arguments") == ["name"]:
                cap["required_arguments"] = ["filename"]
                changes.append({"id": item["id"], "field": "capabilities.find_file.required_arguments", "from": ["name"], "to": ["filename"]})
        if item["id"] == "cap-unsupported-email-001":
            item["expected"]["missing_information"] = []
            changes.append({"id": item["id"], "field": "expected.missing_information", "from": ["recipient_address"], "to": []})
        if item["id"] == "en-music-001":
            old = {k: item["expected"][k] for k in ("intent", "execution_path", "selected_capability", "needs_escalation")}
            item["expected"].update(intent="unsupported_request", execution_path="escalate", selected_capability=None, needs_escalation=True)
            changes.append({"id": item["id"], "field": "expected semantic decision", "from": old, "to": {k: item["expected"][k] for k in old}})
        rows.append(item)
    return rows, changes


def main() -> None:
    rows, changes = _derive_v2(); rows.extend(_new_records())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n"
    OUT.write_text(content, encoding="utf-8")
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = groups.setdefault(row["group_id"], {"split": row["split"], "example_ids": []})
        if entry["split"] != row["split"]: raise ValueError(f"group {row['group_id']} crosses splits")
        entry["example_ids"].append(row["id"])
    manifest = {
        "schema_version": "3.0", "dataset": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "source_v2_sha256": hashlib.sha256(V2.read_bytes()).hexdigest(),
        "split_method": "v2 assignments preserved; Phase 3C scenario groups assigned wholly to train or dev",
        "sealed_test_policy": "the 18 v2-derived test examples and labels remain sealed and unchanged",
        "label_changes": changes, "groups": groups,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} examples and {len(changes)} traceable field changes")


if __name__ == "__main__": main()
