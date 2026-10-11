"""Build the deterministic v2 candidate dataset from immutable v1 plus synthetic candidates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "benchmarks" / "fast_classifier" / "v1" / "seed.jsonl"
V2_DIR = ROOT / "benchmarks" / "fast_classifier" / "v2"
V2 = V2_DIR / "dataset.jsonl"
MANIFEST = V2_DIR / "split_manifest.json"

CAPABILITIES = {
    "open_app": {"name": "open_app", "description": "Open an installed desktop application.", "required_arguments": ["app"]},
    "find_file": {"name": "find_file", "description": "Find a file by name in approved locations.", "required_arguments": ["name"]},
    "open_folder": {"name": "open_folder", "description": "Open an approved folder.", "required_arguments": ["path"]},
    "google_search": {"name": "google_search", "description": "Search the web for a query.", "required_arguments": ["query"]},
    "read_page": {"name": "read_page", "description": "Read text from a web page.", "required_arguments": ["url"]},
    "open_url": {"name": "open_url", "description": "Open a URL in the default browser.", "required_arguments": ["url"]},
}

V1_GROUPS = {
    "en-conversation-001": ("conversation_general", "test"),
    "te-native-conversation-001": ("conversation_general", "test"),
    "hi-native-conversation-001": ("conversation_general", "test"),
    "ta-native-conversation-001": ("conversation_general", "test"),
    "en-knowledge-001": ("thermal_sensation", "test"),
    "te-romanized-knowledge-001": ("thermal_sensation", "test"),
    "en-research-001": ("python_release_research", "test"),
    "te-codeswitch-research-001": ("python_release_research", "test"),
    "ta-codeswitch-research-001": ("python_release_research", "test"),
    "en-desktop-001": ("calculator_action", "test"),
    "te-native-action-001": ("calculator_action", "test"),
    "hi-romanized-action-001": ("calculator_action", "test"),
    "ta-romanized-action-001": ("calculator_action", "test"),
    "stt-noisy-action-001": ("calculator_action", "test"),
    "ctx-continue-resolved-001": ("continue_reference", "test"),
    "ctx-continue-unresolved-001": ("continue_reference", "test"),
    "cap-unsupported-email-001": ("unsupported_external_action", "test"),
    "cap-unsupported-home-001": ("unsupported_external_action", "test"),
    "en-memory-001": ("resolved_memory", "dev"),
    "ctx-hologram-001": ("resolved_memory", "dev"),
    "hi-codeswitch-memory-001": ("resolved_memory", "dev"),
    "en-browser-001": ("documentation_browser", "dev"),
    "en-music-001": ("music_search", "dev"),
    "en-multistep-001": ("multistep_research", "dev"),
    "ctx-open-missing-001": ("missing_open_target", "dev"),
    "cap-browser-missing-001": ("missing_search_topic", "dev"),
    "mixed-conversation-action-001": ("mixed_conversation_action", "dev"),
    "ctx-browser-resolved-001": ("resolved_browser_reference", "dev"),
}

SCENARIOS = [
    {
        "group": "synth_open_editor",
        "split": "train",
        "intent": "open:application",
        "path": "single_action",
        "modes": ["action"],
        "capability": "open_app",
        "capabilities": ["open_app", "find_file", "google_search"],
        "phrases": {
            "en": ["Launch the text editor.", "Bring up a text editor.", "Start my editor.", "Can you open the editor?", "I need the text editor running."],
            "te": ["టెక్స్ట్ ఎడిటర్ తెరువు.", "ఎడిటర్‌ని ప్రారంభించు.", "నాకు టెక్స్ట్ ఎడిటర్ ఓపెన్ చేయి.", "ఎడిటర్ తెరవగలవా?", "టెక్స్ట్ ఎడిటర్ రన్ చేయి."],
            "hi": ["टेक्स्ट एडिटर खोलो।", "एडिटर शुरू कर दो।", "मेरा टेक्स्ट एडिटर खोलो।", "क्या एडिटर खोल सकते हो?", "टेक्स्ट एडिटर चला दो।"],
            "ta": ["டெக்ஸ்ட் எடிட்டரை திற.", "எடிட்டரை தொடங்கு.", "என் டெக்ஸ்ட் எடிட்டரை திறக்கவும்.", "எடிட்டரை திறக்க முடியுமா?", "டெக்ஸ்ட் எடிட்டரை இயக்கவும்."],
        },
    },
    {
        "group": "synth_find_budget_file",
        "split": "train",
        "intent": "find:file",
        "path": "single_action",
        "modes": ["action"],
        "capability": "find_file",
        "capabilities": ["find_file", "open_folder", "google_search"],
        "phrases": {
            "en": ["Find my budget spreadsheet.", "Locate the budget workbook.", "Where is the budget file?", "Search my files for the budget sheet.", "Help me find the spending spreadsheet."],
            "te-Latn": ["naa budget spreadsheet kanipettu", "budget workbook ekkada undo choodu", "budget file vetuku", "files lo budget sheet search cheyyi", "spending spreadsheet dorikinchu"],
            "hi-Latn": ["mera budget spreadsheet dhoondo", "budget workbook kahan hai", "budget file search karo", "files mein budget sheet dhoondo", "spending spreadsheet locate karo"],
            "ta-Latn": ["en budget spreadsheet-ai kandupidi", "budget workbook enga irukku", "budget file search pannu", "files-la budget sheet thedu", "spending spreadsheet kandupidi"],
        },
    },
    {
        "group": "synth_stable_knowledge",
        "split": "train",
        "intent": "answer:stable_knowledge",
        "path": "direct_response",
        "modes": ["conversational"],
        "capability": None,
        "capabilities": [],
        "phrases": {
            "en": ["How do rainbows form?", "Explain why the sky looks blue.", "Why does ice float?", "What causes an echo?", "How do magnets attract metal?"],
            "te": ["ఇంద్రధనస్సు ఎలా ఏర్పడుతుంది?", "ఆకాశం నీలంగా ఎందుకు కనిపిస్తుంది?", "మంచు నీటిపై ఎందుకు తేలుతుంది?", "ప్రతిధ్వని ఎలా వస్తుంది?", "అయస్కాంతం లోహాన్ని ఎలా ఆకర్షిస్తుంది?"],
            "hi": ["इंद्रधनुष कैसे बनता है?", "आसमान नीला क्यों दिखता है?", "बर्फ पानी पर क्यों तैरती है?", "गूंज कैसे बनती है?", "चुंबक धातु को क्यों खींचता है?"],
            "ta": ["வானவில் எப்படி உருவாகிறது?", "வானம் ஏன் நீலமாக தெரிகிறது?", "பனி ஏன் நீரில் மிதக்கிறது?", "எதிரொலி எப்படி உருவாகிறது?", "காந்தம் உலோகத்தை எப்படி ஈர்க்கிறது?"],
        },
    },
    {
        "group": "synth_recall_trip",
        "split": "train",
        "intent": "recall:prior_decision",
        "path": "memory_recall",
        "modes": ["memory"],
        "capability": None,
        "capabilities": [],
        "context": {"recent_topics": [{"topic": "team trip", "decision": "travel on Friday morning"}]},
        "reference_kind": "prior_conversation",
        "phrases": {
            "en": ["What did we decide about the team trip?", "Remind me of our travel decision.", "When did we agree to leave for the trip?", "Recall the plan we made for the team outing.", "What was our conclusion about travel?"],
            "te-Latn": ["team trip gurinchi em decide chesam", "mana travel decision gurthu cheyyi", "trip ki eppudu vellali anukunnam", "team outing plan recall cheyyi", "travel gurinchi mana conclusion enti"],
            "hi-Latn": ["team trip ke baare mein kya decide kiya tha", "travel decision yaad dilao", "trip ke liye kab nikalna tha", "team outing ka plan recall karo", "travel par hamara conclusion kya tha"],
            "ta-Latn": ["team trip pathi enna decide pannom", "travel decision nyabagam pannu", "trip-ku eppo kelamba mudivu pannom", "team outing plan recall pannu", "travel pathi namma conclusion enna"],
        },
    },
    {
        "group": "synth_current_release_research",
        "split": "train",
        "intent": "research:current_information",
        "path": "research",
        "modes": ["research", "action"],
        "capability": "google_search",
        "capabilities": ["google_search", "read_page", "find_file"],
        "phrases": {
            "en": ["Check the current stable Rust release.", "Find today's exchange rate for yen.", "Look up the latest browser security update.", "Research the newest Node.js LTS version.", "Find the current train schedule."],
            "mixed-en-te": ["current Rust stable release enti check cheyyi", "today yen exchange rate entha", "latest browser security update choodu", "newest Node.js LTS version vetuku", "current train schedule kanipettu"],
            "mixed-en-hi": ["current Rust stable release check karo", "aaj yen exchange rate kya hai", "latest browser security update dekho", "newest Node.js LTS version dhoondo", "current train schedule batao"],
            "mixed-en-ta": ["current Rust stable release enna check pannu", "today yen exchange rate enna", "latest browser security update paaru", "newest Node.js LTS version thedu", "current train schedule kandupidi"],
        },
    },
    {
        "group": "synth_multistep_article",
        "split": "train",
        "intent": "research:summarize_source",
        "path": "multi_step_task",
        "modes": ["research", "action"],
        "capability": "google_search",
        "capabilities": ["google_search", "read_page", "open_app"],
        "phrases": {
            "en": ["Find the official WebAssembly guide, read it, and summarize it.", "Locate the Kubernetes overview and extract its main ideas.", "Search for the official SQLite architecture page and brief me.", "Find the accessibility standard, read it, then list key points.", "Research the official Git documentation and summarize the workflow."],
            "mixed-en-te": ["official WebAssembly guide find chesi read chesi summarize cheyyi", "Kubernetes overview vetiki main ideas cheppu", "official SQLite architecture page search chesi brief ivvu", "accessibility standard find chesi key points cheppu", "official Git docs research chesi workflow summarize cheyyi"],
            "mixed-en-hi": ["official WebAssembly guide dhoondh kar padh ke summarize karo", "Kubernetes overview se main ideas nikalo", "official SQLite architecture page search karke brief do", "accessibility standard padh kar key points batao", "official Git docs research karke workflow summarize karo"],
            "mixed-en-ta": ["official WebAssembly guide thedi padichu summarize pannu", "Kubernetes overview-la main ideas sollu", "official SQLite architecture page search panni brief kudu", "accessibility standard padichu key points sollu", "official Git docs research panni workflow summarize pannu"],
        },
    },
    {
        "group": "synth_ambiguous_open",
        "split": "train",
        "intent": "open:referenced_item",
        "path": "clarify",
        "modes": ["action"],
        "capability": None,
        "capabilities": ["open_app", "open_folder", "find_file"],
        "missing": "target",
        "reference_kind": "item",
        "phrases": {
            "en": ["Open that.", "Bring it up.", "Can you launch the thing?", "Open the one I meant.", "Start it for me."],
            "te-Latn": ["adi open cheyyi", "daanini paiki teesuku ra", "aa thing launch cheyyi", "nenu annadi open cheyyi", "adi start cheyyi"],
            "hi-Latn": ["woh kholo", "use saamne lao", "us cheez ko launch karo", "jo maine kaha woh kholo", "use start kar do"],
            "ta-Latn": ["atha open pannu", "atha kondu vaa", "antha thing launch pannu", "naan sonnadha open pannu", "atha start pannu"],
        },
    },
    {
        "group": "synth_unsupported_message",
        "split": "train",
        "intent": "send:external_message",
        "path": "escalate",
        "modes": ["action"],
        "capability": None,
        "capabilities": ["open_app", "find_file"],
        "phrases": {
            "en": ["Send a chat message to the supplier.", "Post this update to our team channel.", "Message the landlord for me.", "Send the photo through a messaging app.", "Notify the customer by chat."],
            "te-Latn": ["supplier ki chat message pampu", "team channel lo update post cheyyi", "landlord ki message cheyyi", "photo messaging app lo pampu", "customer ki chat lo notify cheyyi"],
            "hi-Latn": ["supplier ko chat message bhejo", "team channel par update post karo", "landlord ko message karo", "photo messaging app se bhejo", "customer ko chat par notify karo"],
            "ta-Latn": ["supplier-ku chat message anuppu", "team channel-la update post pannu", "landlord-ku message pannu", "photo messaging app-la anuppu", "customer-ku chat-la notify pannu"],
        },
    },
    {
        "group": "synth_open_archive_folder",
        "split": "dev",
        "intent": "open:folder",
        "path": "single_action",
        "modes": ["action"],
        "capability": "open_folder",
        "capabilities": ["open_folder", "find_file", "google_search"],
        "phrases": {
            "en": ["Open the archive folder.", "Show me the archived files directory.", "Bring up the archive directory.", "Open my old-records folder.", "Take me to the archive folder."],
            "te-Latn": ["archive folder open cheyyi", "archived files directory chupinchu", "archive directory paiki teesuku ra", "old records folder open cheyyi", "archive folder ki teesuku vellu"],
            "hi-Latn": ["archive folder kholo", "archived files directory dikhao", "archive directory saamne lao", "old records folder kholo", "archive folder par le chalo"],
            "ta-Latn": ["archive folder-ai open pannu", "archived files directory kaatu", "archive directory kondu vaa", "old records folder-ai open pannu", "archive folder-ku kondu po"],
        },
    },
    {
        "group": "synth_unresolved_followup",
        "split": "dev",
        "intent": "continue:prior_request",
        "path": "clarify",
        "modes": ["action"],
        "capability": None,
        "capabilities": ["find_file", "google_search", "open_app"],
        "missing": "prior_request",
        "reference_kind": "prior_request",
        "phrases": {
            "en": ["Do the next part.", "Carry on from there.", "Finish what we started.", "Continue with the previous thing.", "Take the next step."],
            "te-Latn": ["next part cheyyi", "akkadi nunchi continue cheyyi", "manam start chesindi finish cheyyi", "previous thing continue cheyyi", "next step teesuko"],
            "hi-Latn": ["agla part karo", "wahan se continue karo", "jo shuru kiya use finish karo", "previous thing continue karo", "agla step lo"],
            "ta-Latn": ["next part pannu", "anga irundhu continue pannu", "start pannadhe finish pannu", "previous thing continue pannu", "next step edu"],
        },
    },
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _semantic_path(record: dict[str, Any]) -> str:
    expected = record["expected"]
    if expected["execution_path"] in {"clarify", "escalate"}:
        return expected["execution_path"]
    modes = set(expected["request_modes"])
    if "memory" in modes:
        return "memory_recall"
    if record["id"] == "en-multistep-001":
        return "multi_step_task"
    if "research" in modes:
        return "research"
    if "action" in modes:
        return "single_action"
    return "direct_response"


def _adjudicate_v1(record: dict[str, Any]) -> dict[str, Any]:
    item = json.loads(json.dumps(record, ensure_ascii=False))
    item["schema_version"] = "2.0"
    item["derived_from_id"] = record["id"]
    item["group_id"], item["split"] = V1_GROUPS[record["id"]]
    item["provenance"] = "v1_adjudication"
    item["review_status"] = "pending_human_review"
    item["annotation_status"] = "synthetic_candidate"
    item["source"] = "v1_labels_with_phase_3b_adjudication"
    item["expected"]["execution_path"] = _semantic_path(record)

    if record["id"] == "en-browser-001":
        item["capabilities"] = [CAPABILITIES["google_search"], CAPABILITIES["open_url"]]
        item["expected"]["intent"] = "research:open_documentation"
        item["expected"]["execution_path"] = "research"
        item["expected"]["request_modes"] = ["research", "action"]
        item["expected"]["selected_capability"] = "google_search"
    elif record["id"] == "en-music-001":
        item["expected"]["intent"] = "find:focus_music"
    elif record["id"] == "cap-unsupported-email-001":
        item["expected"]["missing_information"] = [
            {"name": "recipient_address", "description": "An address for the intended recipient"}
        ]
    item["expected"]["intent"] = _compositional_intent(item["expected"])
    return item


def _compositional_intent(expected: dict[str, Any]) -> str:
    path = expected["execution_path"]
    if path == "direct_response":
        return "respond"
    if path == "memory_recall":
        return "recall"
    if path in {"research", "multi_step_task"}:
        return "research"
    if path == "single_action":
        return f"use:{expected['selected_capability']}"
    if path == "clarify":
        return "clarify_request"
    if path == "escalate":
        return "unsupported_request"
    raise ValueError(f"unsupported semantic path {path!r}")


def _synthetic_records() -> list[dict[str, Any]]:
    records = []
    for scenario in SCENARIOS:
        for language, phrases in scenario["phrases"].items():
            for index, utterance in enumerate(phrases, start=1):
                context = scenario.get("context", {})
                reference_kind = scenario.get("reference_kind")
                resolved = None
                if reference_kind and context:
                    resolved = "travel on Friday morning"
                references = []
                if reference_kind:
                    references = [{"reference": utterance, "kind": reference_kind, "resolved_value": resolved}]
                missing = []
                if scenario.get("missing"):
                    missing = [{"name": scenario["missing"], "description": "Required information is not available in the supplied request or context"}]
                records.append({
                    "schema_version": "2.0",
                    "id": f"{scenario['group']}-{language}-{index:02d}",
                    "group_id": scenario["group"],
                    "split": scenario["split"],
                    "utterance": utterance,
                    "context": context,
                    "capabilities": [CAPABILITIES[name] for name in scenario["capabilities"]],
                    "expected": {
                        "intent": _compositional_intent({
                            "execution_path": scenario["path"],
                            "selected_capability": scenario["capability"],
                        }),
                        "execution_path": scenario["path"],
                        "request_modes": scenario["modes"],
                        "selected_capability": scenario["capability"],
                        "context_references": references,
                        "missing_information": missing,
                        "needs_clarification": scenario["path"] == "clarify",
                        "needs_escalation": scenario["path"] == "escalate",
                    },
                    "language": language,
                    "categories": [scenario["path"], "synthetic_training_candidate"],
                    "annotation_status": "synthetic_candidate",
                    "review_status": "pending_human_review",
                    "provenance": "template_generated",
                    "source": "phase_3b_synthetic_candidate",
                    "notes": "Generated candidate; requires human language and label review before authoritative evaluation use.",
                })
    return records


def main() -> None:
    v1_records = _read_jsonl(V1)
    records = [_adjudicate_v1(record) for record in v1_records] + _synthetic_records()
    V2_DIR.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records) + "\n"
    V2.write_text(content, encoding="utf-8")
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        group = groups.setdefault(record["group_id"], {"split": record["split"], "example_ids": []})
        if group["split"] != record["split"]:
            raise ValueError(f"group {record['group_id']} crosses splits")
        group["example_ids"].append(record["id"])
    manifest = {
        "schema_version": "2.0",
        "dataset": str(V2.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": hashlib.sha256(V2.read_bytes()).hexdigest(),
        "split_method": "scenario-group assignment; no group appears in more than one split",
        "sealed_test_policy": "v1-derived test groups only; no synthetic candidate is assigned to test",
        "source_v1_sha256": hashlib.sha256(V1.read_bytes()).hexdigest(),
        "groups": groups,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} examples ({len(v1_records)} v1-derived, {len(records)-len(v1_records)} synthetic)")


if __name__ == "__main__":
    main()
