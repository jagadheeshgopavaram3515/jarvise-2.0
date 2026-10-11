"""Build the Phase 4 comparison dataset without modifying earlier versions."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "benchmarks" / "fast_classifier" / "v3" / "dataset.jsonl"
OUT_DIR = ROOT / "benchmarks" / "fast_classifier" / "v4"
OUT = OUT_DIR / "dataset.jsonl"
MANIFEST = OUT_DIR / "split_manifest.json"

CAPS = {
    "location": ("Determine the user's current geographic location.", []),
    "weather": ("Fetch current weather for coordinates or a city.", ["city"]),
    "open_app": ("Open an allow-listed desktop application.", ["app"]),
    "close_app": ("Close an allow-listed running desktop application.", ["app"]),
    "find_file": ("Find a file by filename in approved locations.", ["filename"]),
    "search_files": ("Search approved folders for files matching a query.", ["query"]),
    "open_file": ("Open a resolved file path with its default application.", ["path"]),
    "open_folder": ("Open a resolved folder path in Windows Explorer.", ["path"]),
    "recent_files": ("List recently modified files in approved locations.", []),
    "read_file": ("Read a text file from an approved path.", ["path"]),
    "list_directory": ("List files and folders in an approved directory.", ["path"]),
    "grep_code": ("Search source files for a text or regular-expression pattern.", ["query"]),
    "run_project_tests": ("Run the configured test suite for an approved project.", ["project_path"]),
    "open_url": ("Open a supplied URL in the default browser.", ["url"]),
    "google_search": ("Search the web for a supplied query.", ["query"]),
    "youtube_search": ("Search YouTube for a supplied query.", ["query"]),
    "read_page": ("Read text from a supplied web page URL.", ["url"]),
    "get_page_title": ("Get the title of a supplied web page URL.", ["url"]),
}

LANGUAGES = ("en", "te", "hi", "te-Latn", "hi-Latn", "mixed-en-te", "mixed-en-hi")


def cap(name: str) -> dict[str, Any]:
    description, required = CAPS[name]
    return {"name": name, "description": description, "required_arguments": required}


def expected(
    path: str,
    capability: str | None = None,
    *,
    intent: str | None = None,
    reference: dict[str, Any] | None = None,
    missing: tuple[str, str] | None = None,
) -> dict[str, Any]:
    modes = {
        "direct_response": ["conversational"],
        "memory_recall": ["memory"],
        "research": ["research", "action"],
        "single_action": ["action"],
        "multi_step_task": ["research", "action"],
        "clarify": ["action"],
        "escalate": ["action"],
    }[path]
    default_intent = {
        "direct_response": "respond",
        "memory_recall": "recall",
        "research": "research",
        "multi_step_task": "research",
        "clarify": "clarify_request",
        "escalate": "unsupported_request",
    }.get(path, f"use:{capability}")
    return {
        "intent": intent or default_intent,
        "execution_path": path,
        "request_modes": modes,
        "selected_capability": capability,
        "context_references": [reference] if reference else [],
        "missing_information": (
            [{"name": missing[0], "description": missing[1]}] if missing else []
        ),
        "needs_clarification": path == "clarify",
        "needs_escalation": path == "escalate",
    }


# Each tuple is one semantic family. All seven language variants remain in test.
SCENARIOS: list[dict[str, Any]] = [
    {"group": "v4_casual_weekend", "path": "direct_response", "caps": [], "text": {
        "en": "I had a rough week. Can we talk for a bit?",
        "te": "ఈ వారం చాలా కష్టంగా గడిచింది. కాసేపు మాట్లాడుకుందామా?",
        "hi": "यह हफ्ता बहुत मुश्किल था। थोड़ी देर बात करें?",
        "te-Latn": "ee week chala kashtanga undi, kasepu matladadama?",
        "hi-Latn": "ye hafta bahut mushkil tha, thodi der baat karein?",
        "mixed-en-te": "This week చాలా tough గా ఉంది, కాసేపు talk చేద్దామా?",
        "mixed-en-hi": "This week बहुत tough था, थोड़ी देर talk करें?",
    }},
    {"group": "v4_knowledge_rainbow", "path": "direct_response", "caps": [], "text": {
        "en": "Why does a rainbow appear after rain?",
        "te": "వర్షం తర్వాత ఇంద్రధనస్సు ఎందుకు కనిపిస్తుంది?",
        "hi": "बारिश के बाद इंद्रधनुष क्यों दिखाई देता है?",
        "te-Latn": "varsham tarvata rainbow enduku kanipistundi?",
        "hi-Latn": "barish ke baad rainbow kyon dikhai deta hai?",
        "mixed-en-te": "Rain తర్వాత rainbow ఎందుకు కనిపిస్తుంది?",
        "mixed-en-hi": "Rain के बाद rainbow क्यों दिखता है?",
    }},
    {"group": "v4_memory_release", "path": "memory_recall", "caps": [],
     "context": {"recent_topics": [{"topic": "release window", "decision": "deploy on Tuesday evening"}]},
     "reference": {"reference": "release window decision", "kind": "prior_conversation", "resolved_value": "deploy on Tuesday evening"},
     "text": {
        "en": "What did we decide about the release window?",
        "te": "రిలీజ్ సమయం గురించి మనం ఏం నిర్ణయించాం?",
        "hi": "रिलीज़ के समय के बारे में हमने क्या तय किया था?",
        "te-Latn": "release time gurinchi manam em decide chesam?",
        "hi-Latn": "release time ke baare mein humne kya decide kiya tha?",
        "mixed-en-te": "Release window గురించి మన decision ఏంటి?",
        "mixed-en-hi": "Release window के बारे में हमारा decision क्या था?",
    }},
    {"group": "v4_followup_unresolved", "path": "clarify", "caps": ["open_file", "open_url"],
     "reference": {"reference": "that one", "kind": "item", "resolved_value": None},
     "missing": ("target", "The referenced item is not present in the supplied context."), "text": {
        "en": "Open that one again.", "te": "దాన్ని మళ్లీ తెరువు.", "hi": "उसे फिर से खोलो।",
        "te-Latn": "adi malli open cheyyi.", "hi-Latn": "use phir se kholo.",
        "mixed-en-te": "That one మళ్లీ open చెయ్యి.", "mixed-en-hi": "That one फिर से open करो.",
    }},
    {"group": "v4_weather_pune", "path": "research", "selected": "weather", "caps": ["weather", "location"], "text": {
        "en": "What is the weather in Pune right now?", "te": "ఇప్పుడు పుణెలో వాతావరణం ఎలా ఉంది?",
        "hi": "अभी पुणे में मौसम कैसा है?", "te-Latn": "ippudu Pune lo weather ela undi?",
        "hi-Latn": "abhi Pune mein weather kaisa hai?", "mixed-en-te": "Pune లో current weather చెప్పు.",
        "mixed-en-hi": "Pune में current weather बताओ.",
    }},
    {"group": "v4_current_location", "path": "single_action", "selected": "location", "caps": ["location", "weather"], "text": {
        "en": "Tell me which city I am in.", "te": "నేను ఏ నగరంలో ఉన్నానో చెప్పు.",
        "hi": "बताओ मैं किस शहर में हूँ।", "te-Latn": "nenu ye city lo unnano cheppu.",
        "hi-Latn": "batao main kis city mein hoon.", "mixed-en-te": "My current city ఏదో చెప్పు.",
        "mixed-en-hi": "My current city कौन सा है बताओ.",
    }},
    {"group": "v4_open_calculator", "path": "single_action", "selected": "open_app", "caps": ["open_app", "open_file"], "text": {
        "en": "Launch Calculator.", "te": "కాలిక్యులేటర్ తెరువు.", "hi": "कैलकुलेटर खोलो।",
        "te-Latn": "calculator open cheyyi.", "hi-Latn": "calculator kholo.",
        "mixed-en-te": "Calculator ని open చెయ్యి.", "mixed-en-hi": "Calculator को open करो.",
    }},
    {"group": "v4_close_notepad", "path": "single_action", "selected": "close_app", "caps": ["close_app", "open_app"], "text": {
        "en": "Please close Notepad.", "te": "నోట్‌ప్యాడ్ మూసివేయి.", "hi": "नोटपैड बंद कर दो।",
        "te-Latn": "Notepad close cheyyi.", "hi-Latn": "Notepad band karo.",
        "mixed-en-te": "Notepad ని close చెయ్యి.", "mixed-en-hi": "Notepad को close कर दो.",
    }},
    {"group": "v4_find_invoice", "path": "single_action", "selected": "find_file", "caps": ["find_file", "google_search"], "text": {
        "en": "Find the file named invoice-july.pdf.", "te": "invoice-july.pdf అనే ఫైల్‌ను కనుగొను.",
        "hi": "invoice-july.pdf नाम की फ़ाइल ढूँढो।", "te-Latn": "invoice-july.pdf ane file ni vetuku.",
        "hi-Latn": "invoice-july.pdf naam ki file dhoondo.", "mixed-en-te": "invoice-july.pdf file ని find చెయ్యి.",
        "mixed-en-hi": "invoice-july.pdf file को find करो.",
    }},
    {"group": "v4_open_downloads", "path": "single_action", "selected": "open_folder", "caps": ["open_folder", "open_file"], "text": {
        "en": "Open my Downloads folder.", "te": "నా డౌన్‌లోడ్స్ ఫోల్డర్ తెరువు.",
        "hi": "मेरा डाउनलोड्स फ़ोल्डर खोलो।", "te-Latn": "naa Downloads folder open cheyyi.",
        "hi-Latn": "mera Downloads folder kholo.", "mixed-en-te": "My Downloads folder తెరువు.",
        "mixed-en-hi": "My Downloads folder खोलो.",
    }},
    {"group": "v4_open_python_docs", "path": "single_action", "selected": "open_url", "caps": ["open_url", "google_search"], "text": {
        "en": "Open https://docs.python.org in the browser.", "te": "బ్రౌజర్‌లో https://docs.python.org తెరువు.",
        "hi": "ब्राउज़र में https://docs.python.org खोलो।", "te-Latn": "browser lo https://docs.python.org open cheyyi.",
        "hi-Latn": "browser mein https://docs.python.org kholo.", "mixed-en-te": "Browser లో https://docs.python.org open చెయ్యి.",
        "mixed-en-hi": "Browser में https://docs.python.org open करो.",
    }},
    {"group": "v4_research_ai_news", "path": "research", "selected": "google_search", "caps": ["google_search", "open_url"], "text": {
        "en": "Search for today's major AI news.", "te": "ఈ రోజు ముఖ్యమైన ఏఐ వార్తల కోసం వెతుకు.",
        "hi": "आज की प्रमुख एआई खबरें खोजो।", "te-Latn": "ivala main AI news kosam search cheyyi.",
        "hi-Latn": "aaj ki main AI news search karo.", "mixed-en-te": "Today's latest AI news search చెయ్యి.",
        "mixed-en-hi": "Today's latest AI news search करो.",
    }},
    {"group": "v4_music_focus", "path": "single_action", "selected": "youtube_search", "caps": ["youtube_search", "google_search"], "text": {
        "en": "Find focus music on YouTube.", "te": "యూట్యూబ్‌లో పని చేసేటప్పుడు వినే సంగీతం వెతుకు.",
        "hi": "यूट्यूब पर काम के लिए संगीत खोजो।", "te-Latn": "YouTube lo focus music vetuku.",
        "hi-Latn": "YouTube par focus music dhoondo.", "mixed-en-te": "YouTube లో focus music search చెయ్యి.",
        "mixed-en-hi": "YouTube पर focus music search करो.",
    }},
    {"group": "v4_reminder_unsupported", "path": "escalate", "caps": ["open_app", "weather"], "text": {
        "en": "Remind me at six to call the dentist.", "te": "ఆరు గంటలకు దంత వైద్యుడికి ఫోన్ చేయమని గుర్తు చేయి.",
        "hi": "छह बजे दाँत के डॉक्टर को फोन करने की याद दिलाना।", "te-Latn": "six ki dentist ki call cheyamani remind cheyyi.",
        "hi-Latn": "six baje dentist ko call karne ka reminder laga do.", "mixed-en-te": "6 PM కి dentist call reminder పెట్టు.",
        "mixed-en-hi": "6 PM पर dentist call reminder लगा दो.",
    }},
    {"group": "v4_ambiguous_file", "path": "clarify", "caps": ["open_file", "find_file"],
     "reference": {"reference": "the report", "kind": "file", "resolved_value": None},
     "missing": ("path", "Several reports may exist and no file path is supplied."), "text": {
        "en": "Open the report.", "te": "రిపోర్ట్ తెరువు.", "hi": "रिपोर्ट खोलो।",
        "te-Latn": "report open cheyyi.", "hi-Latn": "report kholo.",
        "mixed-en-te": "The report ని open చెయ్యి.", "mixed-en-hi": "The report को open करो.",
    }},
    {"group": "v4_search_missing_query", "path": "clarify", "caps": ["google_search", "open_url"],
     "reference": {"reference": "that", "kind": "topic", "resolved_value": None},
     "missing": ("query", "The search topic is absent from the supplied context."), "text": {
        "en": "Search for that.", "te": "దాని కోసం వెతుకు.", "hi": "उसके बारे में खोजो।",
        "te-Latn": "dani gurinchi search cheyyi.", "hi-Latn": "uske baare mein search karo.",
        "mixed-en-te": "That topic ని search చెయ్యి.", "mixed-en-hi": "That topic को search करो.",
    }},
    {"group": "v4_multistep_ai_news", "path": "multi_step_task", "selected": "google_search", "caps": ["google_search", "read_page"], "text": {
        "en": "Find two current AI stories, read both sources, and summarize the differences.",
        "te": "ప్రస్తుత ఏఐ వార్తల్లో రెండు కనుగొని, రెండు మూలాలను చదివి, తేడాలు చెప్పు.",
        "hi": "एआई की दो मौजूदा खबरें खोजो, दोनों स्रोत पढ़ो और अंतर बताओ।",
        "te-Latn": "two current AI stories find chesi, rendu sources chadivi differences summarize cheyyi.",
        "hi-Latn": "do current AI stories dhoondh kar dono sources padho aur differences summarize karo.",
        "mixed-en-te": "Two current AI stories find చేసి, sources చదివి differences summarize చెయ్యి.",
        "mixed-en-hi": "Two current AI stories find करके, sources पढ़ो और differences summarize करो.",
    }},
    {"group": "v4_project_todo_inspection", "path": "multi_step_task", "selected": "grep_code", "caps": ["grep_code", "read_file", "run_project_tests"], "text": {
        "en": "Search the project for TODO comments and summarize the files that need attention.",
        "te": "ప్రాజెక్ట్‌లో TODO వ్యాఖ్యలు వెతికి, చూడాల్సిన ఫైళ్లను సంక్షిప్తంగా చెప్పు.",
        "hi": "प्रोजेक्ट में TODO टिप्पणियाँ खोजो और जिन फ़ाइलों पर ध्यान चाहिए उनका सार बताओ।",
        "te-Latn": "project lo TODO comments search chesi attention kavalsina files summarize cheyyi.",
        "hi-Latn": "project mein TODO comments search karke attention wali files summarize karo.",
        "mixed-en-te": "Project లో TODO comments grep చేసి files summarize చెయ్యి.",
        "mixed-en-hi": "Project में TODO comments grep करके files summarize करो.",
    }},
    {"group": "v4_recent_files", "path": "single_action", "selected": "recent_files", "caps": ["recent_files", "search_files"], "text": {
        "en": "Show me the files I changed recently.", "te": "నేను ఇటీవల మార్చిన ఫైళ్లను చూపించు.",
        "hi": "मैंने हाल में जो फ़ाइलें बदली हैं वे दिखाओ।", "te-Latn": "nenu recent ga change chesina files chupinchu.",
        "hi-Latn": "maine recently change ki files dikhao.", "mixed-en-te": "Recently changed files నాకు చూపించు.",
        "mixed-en-hi": "Recently changed files मुझे दिखाओ.",
    }},
    {"group": "v4_email_unsupported", "path": "escalate", "caps": ["open_file", "google_search"], "text": {
        "en": "Email the budget sheet to Priya.", "te": "బడ్జెట్ షీట్‌ను ప్రియకు ఈమెయిల్ చేయి.",
        "hi": "बजट शीट प्रिया को ईमेल करो।", "te-Latn": "budget sheet Priya ki email cheyyi.",
        "hi-Latn": "budget sheet Priya ko email karo.", "mixed-en-te": "Budget sheet ని Priya కి email చెయ్యి.",
        "mixed-en-hi": "Budget sheet को Priya को email करो.",
    }},
]


def build_rows() -> list[dict[str, Any]]:
    rows = []
    for line in V3.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["split"] == "test":
            continue
        row["schema_version"] = "4.0"
        row["derived_from_v3_id"] = row["id"]
        row["annotation_status"] = "pending_review"
        row["review_status"] = "pending_review"
        rows.append(row)

    for scenario in SCENARIOS:
        for language in LANGUAGES:
            reference = scenario.get("reference")
            rows.append({
                "schema_version": "4.0",
                "id": f"{scenario['group']}-{language}",
                "split": "test",
                "utterance": scenario["text"][language],
                "context": scenario.get("context", {}),
                "capabilities": [cap(name) for name in scenario["caps"]],
                "expected": expected(
                    scenario["path"], scenario.get("selected"),
                    reference=reference, missing=scenario.get("missing"),
                ),
                "language": language,
                "categories": [
                    scenario["path"],
                    "phase4_comparison",
                    "native_script" if language in {"te", "hi"} else
                    "romanized" if language.endswith("Latn") else
                    "code_switching" if language.startswith("mixed-") else "english",
                ],
                "annotation_status": "pending_review",
                "source": "phase4_manual_synthetic_candidate",
                "notes": "Author-created comparison candidate; independent human/native-speaker review has not occurred.",
                "group_id": scenario["group"],
                "provenance": "phase4_targeted_synthetic",
                "review_status": "pending_review",
            })
    return rows


def main() -> None:
    rows = build_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        group = groups.setdefault(row["group_id"], {"split": row["split"], "example_ids": []})
        if group["split"] != row["split"]:
            raise ValueError(f"group {row['group_id']} crosses splits")
        group["example_ids"].append(row["id"])
    split_counts = Counter(row["split"] for row in rows)
    test_languages = Counter(row["language"] for row in rows if row["split"] == "test")
    manifest = {
        "schema_version": "4.0",
        "dataset": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "source_v3_sha256": hashlib.sha256(V3.read_bytes()).hexdigest(),
        "split_method": "Retain v3 train/dev; replace v3 test with new Phase 4 groups held wholly in sealed test.",
        "sealed_test_policy": "Phase 4 test groups were not used by Phase 3 training or development. Do not train or tune on them.",
        "review_policy": "All records are pending independent human review; no human review is claimed.",
        "counts": {"total": len(rows), "by_split": dict(split_counts), "test_by_language": dict(test_languages)},
        "groups": groups,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
