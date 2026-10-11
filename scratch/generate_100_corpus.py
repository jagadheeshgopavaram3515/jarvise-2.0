"""Script to construct and validate the 100-request frozen evaluation corpus.
Constructed with diverse scenario templates and realistic multilingual variations.
"""
import json
from pathlib import Path

OUT = Path('D:/JARVIS/reports/classifier_v4_frozen_100_eval_20261010_180014')

# Standard capability definitions
CAP_OPEN_FOLDER = {"name": "open_folder", "description": "Open a specified folder or directory in file explorer.", "required_arguments": ["path"]}
CAP_FIND_FILE = {"name": "find_file", "description": "Find local files or documents matching a query.", "required_arguments": ["query"]}
CAP_SEARCH = {"name": "google_search", "description": "Perform web search for information or tutorials.", "required_arguments": ["query"]}
CAP_READ_PAGE = {"name": "read_page", "description": "Read content from a specified URL.", "required_arguments": ["url"]}
CAP_OPEN_APP = {"name": "open_app", "description": "Launch or open an installed desktop application.", "required_arguments": ["name"]}
CAP_CLOSE_APP = {"name": "close_app", "description": "Close a running application.", "required_arguments": ["name"]}
CAP_DELETE_FILE = {"name": "delete_file", "description": "Delete a specified file permanently.", "required_arguments": ["path"]}
CAP_DELETE_FOLDER = {"name": "delete_folder", "description": "Delete a specified directory permanently.", "required_arguments": ["path"]}

RECORDS = []

def add_record(rec_id, lang, script, group, text, context, caps, exp_path, exp_modes, exp_cap, exp_clar, exp_esc, exp_miss, exp_refs, exp_intent=None):
    if exp_intent is None:
        if exp_path == 'single_action' and exp_cap:
            exp_intent = f"use:{exp_cap}"
        elif exp_path == 'multi_step_task' and exp_cap:
            exp_intent = f"use:{exp_cap}"
        elif exp_path == 'direct_response':
            exp_intent = "respond"
        elif exp_path == 'memory_recall':
            exp_intent = "recall"
        elif exp_path == 'clarify':
            exp_intent = "clarify_request"
        elif exp_path == 'escalate':
            exp_intent = "unsupported_request"
        else:
            exp_intent = "unsupported_request"

    rec = {
        "id": f"frozen100-{rec_id:03d}",
        "language": lang,
        "script": script,
        "scenario_group": group,
        "utterance": text,
        "context": context,
        "capabilities": caps,
        "provenance": "authored_frozen_provisional_v4",
        "expected": {
            "intent": exp_intent,
            "execution_path": exp_path,
            "request_modes": sorted(exp_modes),
            "selected_capability": exp_cap,
            "needs_clarification": exp_clar,
            "needs_escalation": exp_esc,
            "missing_information": exp_miss,
            "context_references": exp_refs
        }
    }
    RECORDS.append(rec)

# ==============================================================================
# GROUP 1: Container / Folder Operations (12 records)
# ==============================================================================
# 001: en - open documents directory
add_record(1, "en", "latin", "corpus100-folder", "Open my documents directory please", {}, [CAP_OPEN_FOLDER, CAP_FIND_FILE], "single_action", ["action"], "open_folder", False, False, [], [])
# 002: en - open logs directory
add_record(2, "en", "latin", "corpus100-folder", "Browse to the server logs folder", {}, [CAP_OPEN_FOLDER, CAP_SEARCH], "single_action", ["action"], "open_folder", False, False, [], [])
# 003: hi - फ़ोल्डर खोलो (Devanagari)
add_record(3, "hi", "devanagari", "corpus100-folder", "डाउनलोड्स फ़ोल्डर खोलो", {}, [CAP_OPEN_FOLDER, CAP_FIND_FILE], "single_action", ["action"], "open_folder", False, False, [], [])
# 004: hi - बैकअप डायरेक्टरी खोलो (Devanagari)
add_record(4, "hi", "devanagari", "corpus100-folder", "बैकअप डायरेक्टरी स्क्रीन पर दिखाओ", {}, [CAP_OPEN_FOLDER, CAP_SEARCH], "single_action", ["action"], "open_folder", False, False, [], [])
# 005: hi-Latn - photos folder kholo
add_record(5, "hi-Latn", "latin", "corpus100-folder", "photos folder khol kar dikhao", {}, [CAP_OPEN_FOLDER, CAP_FIND_FILE], "single_action", ["action"], "open_folder", False, False, [], [])
# 006: hi-Latn - project directory show karo
add_record(6, "hi-Latn", "latin", "corpus100-folder", "client project directory open karo", {}, [CAP_OPEN_FOLDER, CAP_OPEN_APP], "single_action", ["action"], "open_folder", False, False, [], [])
# 007: te - ఫోల్డర్ ఓపెన్ చెయ్యి (Telugu script)
add_record(7, "te", "telugu", "corpus100-folder", "నా డౌన్‌లోడ్స్ ఫోల్డర్ ఓపెన్ చెయ్యి", {}, [CAP_OPEN_FOLDER, CAP_FIND_FILE], "single_action", ["action"], "open_folder", False, False, [], [])
# 008: te - డాక్యుమెంట్స్ డైరెక్టరీ చూపించు (Telugu script)
add_record(8, "te", "telugu", "corpus100-folder", "డాక్యుమెంట్స్ డైరెక్టరీ చూపించు", {}, [CAP_OPEN_FOLDER, CAP_SEARCH], "single_action", ["action"], "open_folder", False, False, [], [])
# 009: te-Latn - music folder open cheyyi
add_record(9, "te-Latn", "latin", "corpus100-folder", "music folder open cheyyi", {}, [CAP_OPEN_FOLDER, CAP_FIND_FILE], "single_action", ["action"], "open_folder", False, False, [], [])
# 010: te-Latn - archive directory chupinchu
add_record(10, "te-Latn", "latin", "corpus100-folder", "backup archive directory screen meeda chupinchu", {}, [CAP_OPEN_FOLDER, CAP_SEARCH], "single_action", ["action"], "open_folder", False, False, [], [])
# 011: ta - கோப்புறையை திற (Tamil script)
add_record(11, "ta", "tamil", "corpus100-folder", "பதிவிறக்கங்கள் கோப்புறையை திறக்கவும்", {}, [CAP_OPEN_FOLDER, CAP_FIND_FILE], "single_action", ["action"], "open_folder", False, False, [], [])
# 012: ta-Latn - pictures folder open pannu
add_record(12, "ta-Latn", "latin", "corpus100-folder", "pictures folder-ai open pannu", {}, [CAP_OPEN_FOLDER, CAP_FIND_FILE], "single_action", ["action"], "open_folder", False, False, [], [])

# ==============================================================================
# GROUP 2: File Operations & Finding Files (12 records)
# ==============================================================================
# 013: en - find quarterly financial report
add_record(13, "en", "latin", "corpus100-file", "Find the quarterly financial report spreadsheet", {}, [CAP_FIND_FILE, CAP_OPEN_FOLDER], "single_action", ["action"], "find_file", False, False, [], [])
# 014: en - search for resume pdf
add_record(14, "en", "latin", "corpus100-file", "Search for my resume pdf file on disk", {}, [CAP_FIND_FILE, CAP_SEARCH], "single_action", ["action"], "find_file", False, False, [], [])
# 015: hi - फाइल खोजो (Devanagari)
add_record(15, "hi", "devanagari", "corpus100-file", "टैक्स रसीद की पीडीएफ फाइल ढूंढो", {}, [CAP_FIND_FILE, CAP_OPEN_FOLDER], "single_action", ["action"], "find_file", False, False, [], [])
# 016: hi - डॉक्युमेंट तलाशो (Devanagari)
add_record(16, "hi", "devanagari", "corpus100-file", "प्रोजेक्ट प्रपोजल डॉक्यूमेंट तलाशो", {}, [CAP_FIND_FILE, CAP_OPEN_APP], "single_action", ["action"], "find_file", False, False, [], [])
# 017: hi-Latn - invoice file search karo
add_record(17, "hi-Latn", "latin", "corpus100-file", "october invoice file search karo", {}, [CAP_FIND_FILE, CAP_OPEN_FOLDER], "single_action", ["action"], "find_file", False, False, [], [])
# 018: hi-Latn - meeting notes document dhoondo
add_record(18, "hi-Latn", "latin", "corpus100-file", "yesterday meeting notes text file dhoondo", {}, [CAP_FIND_FILE, CAP_SEARCH], "single_action", ["action"], "find_file", False, False, [], [])
# 019: te - ఫైల్ వెతుకు (Telugu script)
add_record(19, "te", "telugu", "corpus100-file", "కాలేజ్ సర్టిఫికెట్ పిడిఎఫ్ ఫైల్ వెతుకు", {}, [CAP_FIND_FILE, CAP_OPEN_FOLDER], "single_action", ["action"], "find_file", False, False, [], [])
# 020: te - డాక్యుమెంట్ కనుక్కో (Telugu script)
add_record(20, "te", "telugu", "corpus100-file", "ప్రాజెక్ట్ డాక్యుమెంట్ కనుక్కో", {}, [CAP_FIND_FILE, CAP_SEARCH], "single_action", ["action"], "find_file", False, False, [], [])
# 021: te-Latn - contract file vetuku
add_record(21, "te-Latn", "latin", "corpus100-file", "vendor contract pdf file vetuku", {}, [CAP_FIND_FILE, CAP_OPEN_FOLDER], "single_action", ["action"], "find_file", False, False, [], [])
# 022: te-Latn - budget spreadsheet kanukko
add_record(22, "te-Latn", "latin", "corpus100-file", "annual budget excel sheet file kanukko", {}, [CAP_FIND_FILE, CAP_OPEN_APP], "single_action", ["action"], "find_file", False, False, [], [])
# 023: ta - கோப்பை கண்டுபிடி (Tamil script)
add_record(23, "ta", "tamil", "corpus100-file", "வருமான வரி ரசீது கோப்பை தேடு", {}, [CAP_FIND_FILE, CAP_OPEN_FOLDER], "single_action", ["action"], "find_file", False, False, [], [])
# 024: ta-Latn - project report file thedu
add_record(24, "ta-Latn", "latin", "corpus100-file", "final project report document thedu", {}, [CAP_FIND_FILE, CAP_SEARCH], "single_action", ["action"], "find_file", False, False, [], [])

# ==============================================================================
# GROUP 3: Desktop Application Control (10 records)
# ==============================================================================
# 025: en - open terminal
add_record(25, "en", "latin", "corpus100-app", "Open PowerShell terminal application", {}, [CAP_OPEN_APP, CAP_OPEN_FOLDER], "single_action", ["action"], "open_app", False, False, [], [])
# 026: en - close chrome
add_record(26, "en", "latin", "corpus100-app", "Close Google Chrome application", {}, [CAP_CLOSE_APP, CAP_SEARCH], "single_action", ["action"], "close_app", False, False, [], [])
# 027: hi - ऐप खोलो (Devanagari)
add_record(27, "hi", "devanagari", "corpus100-app", "कैलकुलेटर ऐप खोलो", {}, [CAP_OPEN_APP, CAP_SEARCH], "single_action", ["action"], "open_app", False, False, [], [])
# 028: hi - ऐप बंद करो (Devanagari)
add_record(28, "hi", "devanagari", "corpus100-app", "नोटपैड ऐप बंद करो", {}, [CAP_CLOSE_APP, CAP_OPEN_FOLDER], "single_action", ["action"], "close_app", False, False, [], [])
# 029: hi-Latn - spotify app open karo
add_record(29, "hi-Latn", "latin", "corpus100-app", "Spotify application open karo", {}, [CAP_OPEN_APP, CAP_FIND_FILE], "single_action", ["action"], "open_app", False, False, [], [])
# 030: hi-Latn - browser band karo
add_record(30, "hi-Latn", "latin", "corpus100-app", "Chrome browser application band karo", {}, [CAP_CLOSE_APP, CAP_SEARCH], "single_action", ["action"], "close_app", False, False, [], [])
# 031: te - యాప్ ఓపెన్ చెయ్యి (Telugu script)
add_record(31, "te", "telugu", "corpus100-app", "కాలిక్యులేటర్ అప్లికేషన్ ఓపెన్ చెయ్యి", {}, [CAP_OPEN_APP, CAP_FIND_FILE], "single_action", ["action"], "open_app", False, False, [], [])
# 032: te-Latn - terminal close cheyyi
add_record(32, "te-Latn", "latin", "corpus100-app", "Terminal application close cheyyi", {}, [CAP_CLOSE_APP, CAP_OPEN_FOLDER], "single_action", ["action"], "close_app", False, False, [], [])
# 033: ta-Latn - vs code open pannu
add_record(33, "ta-Latn", "latin", "corpus100-app", "VS Code application open pannu", {}, [CAP_OPEN_APP, CAP_FIND_FILE], "single_action", ["action"], "open_app", False, False, [], [])
# 034: ta-Latn - media player close pannu
add_record(34, "ta-Latn", "latin", "corpus100-app", "VLC media player application close pannu", {}, [CAP_CLOSE_APP, CAP_OPEN_FOLDER], "single_action", ["action"], "close_app", False, False, [], [])

# ==============================================================================
# GROUP 4: Web Research & Information Search (12 records)
# ==============================================================================
# 035: en - web search quantum computing
add_record(35, "en", "latin", "corpus100-search", "Google search for recent quantum computing breakthroughs", {}, [CAP_SEARCH, CAP_FIND_FILE], "single_action", ["action"], "google_search", False, False, [], [])
# 036: en - find rust async tutorial
add_record(36, "en", "latin", "corpus100-search", "Search the web for Rust Tokio async tutorials", {}, [CAP_SEARCH, CAP_OPEN_FOLDER], "single_action", ["action"], "google_search", False, False, [], [])
# 037: hi - गूगल सर्च करो (Devanagari)
add_record(37, "hi", "devanagari", "corpus100-search", "भारत में आगामी चंद्र मिशन के बारे में गूगल सर्च करो", {}, [CAP_SEARCH, CAP_FIND_FILE], "single_action", ["action"], "google_search", False, False, [], [])
# 038: hi - वेब पर खोजो (Devanagari)
add_record(38, "hi", "devanagari", "corpus100-search", "मशीन लर्निंग के नए एल्गोरिदम वेब पर खोजो", {}, [CAP_SEARCH, CAP_OPEN_FOLDER], "single_action", ["action"], "google_search", False, False, [], [])
# 039: hi-Latn - search stock market news
add_record(39, "hi-Latn", "latin", "corpus100-search", "Nifty latest market trends google search karo", {}, [CAP_SEARCH, CAP_OPEN_APP], "single_action", ["action"], "google_search", False, False, [], [])
# 040: hi-Latn - python interview questions search
add_record(40, "hi-Latn", "latin", "corpus100-search", "python advanced interview questions web par search karo", {}, [CAP_SEARCH, CAP_FIND_FILE], "single_action", ["action"], "google_search", False, False, [], [])
# 041: te - గూగుల్ లో వెతుకు (Telugu script)
add_record(41, "te", "telugu", "corpus100-search", "హైదరాబాద్ వాతావరణ అప్‌డేట్స్ గూగుల్ లో వెతుకు", {}, [CAP_SEARCH, CAP_FIND_FILE], "single_action", ["action"], "google_search", False, False, [], [])
# 042: te - సమాచారం వెతుకు (Telugu script)
add_record(42, "te", "telugu", "corpus100-search", "ఆర్టిఫిషియల్ ఇంటెలిజెన్స్ తాజా వార్తలు వెతుకు", {}, [CAP_SEARCH, CAP_OPEN_FOLDER], "single_action", ["action"], "google_search", False, False, [], [])
# 043: te-Latn - cricket score search cheyyi
add_record(43, "te-Latn", "latin", "corpus100-search", "India vs Australia live match score google search cheyyi", {}, [CAP_SEARCH, CAP_OPEN_APP], "single_action", ["action"], "google_search", False, False, [], [])
# 044: te-Latn - golang tutorial search cheyyi
add_record(44, "te-Latn", "latin", "corpus100-search", "Go concurrency goroutines tutorial search cheyyi", {}, [CAP_SEARCH, CAP_FIND_FILE], "single_action", ["action"], "google_search", False, False, [], [])
# 045: ta - இணையத்தில் தேடு (Tamil script)
add_record(45, "ta", "tamil", "corpus100-search", "தமிழ்நாடு சமீபத்திய செய்தி இணையத்தில் தேடு", {}, [CAP_SEARCH, CAP_OPEN_FOLDER], "single_action", ["action"], "google_search", False, False, [], [])
# 046: ta-Latn - machine learning roadmaps search pannu
add_record(46, "ta-Latn", "latin", "corpus100-search", "AI engineer learning roadmap web-il search pannu", {}, [CAP_SEARCH, CAP_FIND_FILE], "single_action", ["action"], "google_search", False, False, [], [])

# ==============================================================================
# GROUP 5: Webpage Reading & URL Feasibility (8 records)
# ==============================================================================
# 047: en - read page with URL
add_record(47, "en", "latin", "corpus100-webpage", "Read the article at https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)", {}, [CAP_READ_PAGE, CAP_SEARCH], "single_action", ["action"], "read_page", False, False, [], [])
# 048: en - read page missing URL -> clarify
add_record(48, "en", "latin", "corpus100-webpage", "Read this web page for me", {}, [CAP_READ_PAGE, CAP_SEARCH], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [])
# 049: hi-Latn - read URL webpage
add_record(49, "hi-Latn", "latin", "corpus100-webpage", "https://docs.python.org/3/whatsnew/3.11.html page read karo", {}, [CAP_READ_PAGE, CAP_SEARCH], "single_action", ["action"], "read_page", False, False, [], [])
# 050: hi-Latn - URL missing page read karo -> clarify
add_record(50, "hi-Latn", "latin", "corpus100-webpage", "web page read karke batao", {}, [CAP_READ_PAGE], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [])
# 051: te-Latn - read URL webpage
add_record(51, "te-Latn", "latin", "corpus100-webpage", "ee https://news.ycombinator.com link page chaduvu", {}, [CAP_READ_PAGE, CAP_SEARCH], "single_action", ["action"], "read_page", False, False, [], [])
# 052: te-Latn - link missing page chaduvu -> clarify
add_record(52, "te-Latn", "latin", "corpus100-webpage", "a web page chadivi summary ivvu", {}, [CAP_READ_PAGE], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [])
# 053: ta-Latn - read URL webpage
add_record(53, "ta-Latn", "latin", "corpus100-webpage", "https://github.com/features link-la irukkura page read pannu", {}, [CAP_READ_PAGE, CAP_SEARCH], "single_action", ["action"], "read_page", False, False, [], [])
# 054: ta-Latn - URL missing webpage read pannu -> clarify
add_record(54, "ta-Latn", "latin", "corpus100-webpage", "web page-ai padithu sollu", {}, [CAP_READ_PAGE], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [])

# ==============================================================================
# GROUP 6: Ambiguous Pronouns & Missing Slot Actions (12 records)
# ==============================================================================
# 055: en - open it (competing open_file vs open_folder)
add_record(55, "en", "latin", "corpus100-clarify", "Open it up right now", {}, [CAP_OPEN_FOLDER, CAP_FIND_FILE], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [{"kind": "item", "reference": "it", "resolved_value": None}])
# 056: en - delete that
add_record(56, "en", "latin", "corpus100-clarify", "Delete that completely", {}, [CAP_DELETE_FILE, CAP_DELETE_FOLDER], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [{"kind": "item", "reference": "it", "resolved_value": None}])
# 057: hi - इसे हटाओ (Devanagari)
add_record(57, "hi", "devanagari", "corpus100-clarify", "इसे तुरंत हटाओ", {}, [CAP_DELETE_FILE, CAP_DELETE_FOLDER], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [{"kind": "item", "reference": "it", "resolved_value": None}])
# 058: hi - वह खोलो (Devanagari)
add_record(58, "hi", "devanagari", "corpus100-clarify", "वह तुरंत खोलो", {}, [CAP_OPEN_FOLDER, CAP_FIND_FILE], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [{"kind": "item", "reference": "it", "resolved_value": None}])
# 059: hi-Latn - use open karo
add_record(59, "hi-Latn", "latin", "corpus100-clarify", "use turant open karo", {}, [CAP_OPEN_FOLDER, CAP_FIND_FILE], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [{"kind": "item", "reference": "it", "resolved_value": None}])
# 060: hi-Latn - isse delete kar do
add_record(60, "hi-Latn", "latin", "corpus100-clarify", "isse delete kar do", {}, [CAP_DELETE_FILE, CAP_DELETE_FOLDER], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [{"kind": "item", "reference": "it", "resolved_value": None}])
# 061: te - దీన్ని డిలీట్ చెయ్యి (Telugu script)
add_record(61, "te", "telugu", "corpus100-clarify", "దీన్ని వెంటనే డిలీట్ చెయ్యి", {}, [CAP_DELETE_FILE, CAP_DELETE_FOLDER], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [{"kind": "item", "reference": "it", "resolved_value": None}])
# 062: te - అది ఓపెన్ చెయ్యి (Telugu script)
add_record(62, "te", "telugu", "corpus100-clarify", "అది ఓపెన్ చెయ్యి", {}, [CAP_OPEN_FOLDER, CAP_FIND_FILE], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [{"kind": "item", "reference": "it", "resolved_value": None}])
# 063: te-Latn - danni delete cheyyi
add_record(63, "te-Latn", "latin", "corpus100-clarify", "danni ventane delete cheyyi", {}, [CAP_DELETE_FILE, CAP_DELETE_FOLDER], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [{"kind": "item", "reference": "it", "resolved_value": None}])
# 064: te-Latn - idi malli open cheyyi
add_record(64, "te-Latn", "latin", "corpus100-clarify", "idi malli open cheyyi", {}, [CAP_OPEN_FOLDER, CAP_FIND_FILE], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [{"kind": "item", "reference": "it", "resolved_value": None}])
# 065: ta - அதை அழி (Tamil script)
add_record(65, "ta", "tamil", "corpus100-clarify", "அதை உடனே அழித்துவிடு", {}, [CAP_DELETE_FILE, CAP_DELETE_FOLDER], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [{"kind": "item", "reference": "it", "resolved_value": None}])
# 066: ta-Latn - athai open pannu
add_record(66, "ta-Latn", "latin", "corpus100-clarify", "athai marubadiyum open pannu", {}, [CAP_OPEN_FOLDER, CAP_FIND_FILE], "clarify", ["action"], None, True, False, [{"name": "target", "description": "The requested target is not identifiable"}], [{"kind": "item", "reference": "it", "resolved_value": None}])

# ==============================================================================
# GROUP 7: Memory & Conversation Reference Recall (12 records)
# ==============================================================================
# 067: en - recall project Phoenix alias
add_record(67, "en", "latin", "corpus100-memory", "What did we decide about our Phoenix project?", {"projects": [{"name": "NextGen Gateway", "aliases": ["Phoenix project"]}]}, [], "memory_recall", ["memory"], None, False, False, [], [{"kind": "project", "reference": "our Phoenix project", "resolved_value": "NextGen Gateway"}])
# 068: en - recall database topic decision
add_record(68, "en", "latin", "corpus100-memory", "What was our decision on database sharding?", {"recent_topics": [{"topic": "database sharding", "decision": "adopt Vitess proxy cluster"}]}, [], "memory_recall", ["memory"], None, False, False, [], [{"kind": "prior_conversation", "reference": "database sharding", "resolved_value": "adopt Vitess proxy cluster"}])
# 069: hi - प्रोजेक्ट निर्णय याद दिलाओ (Devanagari)
add_record(69, "hi", "devanagari", "corpus100-memory", "हमारे गरुड़ प्रोजेक्ट के बारे में क्या तय हुआ था?", {"projects": [{"name": "High Altitude UAV", "aliases": ["गरुड़ प्रोजेक्ट"]}]}, [], "memory_recall", ["memory"], None, False, False, [], [{"kind": "project", "reference": "गरुड़ प्रोजेक्ट", "resolved_value": "High Altitude UAV"}])
# 070: hi - विषय का फैसला (Devanagari)
add_record(70, "hi", "devanagari", "corpus100-memory", "क्लाउड प्रोवाइडर पर हमारा क्या निर्णय था?", {"recent_topics": [{"topic": "क्लाउड प्रोवाइडर", "decision": "AWS चुना गया था"}]}, [], "memory_recall", ["memory"], None, False, False, [], [{"kind": "prior_conversation", "reference": "क्लाउड प्रोवाइडर", "resolved_value": "AWS चुना गया था"}])
# 071: hi-Latn - cache architecture decision
add_record(71, "hi-Latn", "latin", "corpus100-memory", "Redis caching strategy ke baare mein kya decide kiya tha?", {"recent_topics": [{"topic": "Redis caching strategy", "decision": "use Redis Sentinel cluster"}]}, [], "memory_recall", ["memory"], None, False, False, [], [{"kind": "prior_conversation", "reference": "Redis caching strategy", "resolved_value": "use Redis Sentinel cluster"}])
# 072: hi-Latn - project alias recall
add_record(72, "hi-Latn", "latin", "corpus100-memory", "humare Falcon initiative ke baare mein yaad dilao", {"projects": [{"name": "Edge CDN", "aliases": ["Falcon initiative"]}]}, [], "memory_recall", ["memory"], None, False, False, [], [{"kind": "project", "reference": "Falcon initiative", "resolved_value": "Edge CDN"}])
# 073: te - ప్రాజెక్ట్ గుర్తుచేయి (Telugu script)
add_record(73, "te", "telugu", "corpus100-memory", "మన త్రిశూల్ ప్రాజెక్ట్ నిర్ణయం గుర్తుచేయి", {"projects": [{"name": "Security Mesh", "aliases": ["త్రిశూల్ ప్రాజెక్ట్"]}]}, [], "memory_recall", ["memory"], None, False, False, [], [{"kind": "project", "reference": "త్రిశూల్ ప్రాజెక్ట్", "resolved_value": "Security Mesh"}])
# 074: te - గత నిర్ణయం (Telugu script)
add_record(74, "te", "telugu", "corpus100-memory", "ఆథెంటికేషన్ ఆర్కిటెక్చర్ నిర్ణయం ఏంటి?", {"recent_topics": [{"topic": "ఆథెంటికేషన్ ఆర్కిటెక్చర్", "decision": "OAuth2 అమలు చేయాలి"}]}, [], "memory_recall", ["memory"], None, False, False, [], [{"kind": "prior_conversation", "reference": "ఆథెంటికేషన్ ఆర్కిటెక్చర్", "resolved_value": "OAuth2 అమలు చేయాలి"}])
# 075: te-Latn - deployment strategy decision
add_record(75, "te-Latn", "latin", "corpus100-memory", "mana deployment pipeline gurinchi em decide chesam?", {"recent_topics": [{"topic": "deployment pipeline", "decision": "use GitHub Actions CI"}]}, [], "memory_recall", ["memory"], None, False, False, [], [{"kind": "prior_conversation", "reference": "deployment pipeline", "resolved_value": "use GitHub Actions CI"}])
# 076: te-Latn - project alias recall
add_record(76, "te-Latn", "latin", "corpus100-memory", "mana Garuda project status gurinchi gurthu cheyyi", {"projects": [{"name": "Telemetry Engine", "aliases": ["Garuda project"]}]}, [], "memory_recall", ["memory"], None, False, False, [], [{"kind": "project", "reference": "Garuda project", "resolved_value": "Telemetry Engine"}])
# 077: ta-Latn - topic decision recall
add_record(77, "ta-Latn", "latin", "corpus100-memory", "backend framework pathi enna mudivu pannom?", {"recent_topics": [{"topic": "backend framework", "decision": "FastAPI use panna mudivu"}]}, [], "memory_recall", ["memory"], None, False, False, [], [{"kind": "prior_conversation", "reference": "backend framework", "resolved_value": "FastAPI use panna mudivu"}])
# 078: mixed-en-hi - code-switch memory recall
add_record(78, "mixed-en-hi", "latin", "corpus100-memory", "Kal API gateway release date ke baare mein kya bola tha?", {"recent_topics": [{"topic": "API gateway release date", "decision": "release next Tuesday"}]}, [], "memory_recall", ["memory"], None, False, False, [], [{"kind": "prior_conversation", "reference": "API gateway release date", "resolved_value": "release next Tuesday"}])

# ==============================================================================
# GROUP 8: Direct Knowledge & Conversational Q&A (10 records)
# ==============================================================================
# 079: en - capital of Canada
add_record(79, "en", "latin", "corpus100-direct", "What is the capital of Canada?", {}, [CAP_OPEN_FOLDER], "direct_response", ["conversational"], None, False, False, [], [])
# 080: en - speed of sound
add_record(80, "en", "latin", "corpus100-direct", "Explain why the sky looks blue during daytime", {}, [], "direct_response", ["conversational"], None, False, False, [], [])
# 081: hi - भारत की सबसे लंबी नदी (Devanagari)
add_record(81, "hi", "devanagari", "corpus100-direct", "भारत की सबसे लंबी नदी कौन सी है?", {}, [], "direct_response", ["conversational"], None, False, False, [], [])
# 082: hi - सौरमंडल का सबसे बड़ा ग्रह (Devanagari)
add_record(82, "hi", "devanagari", "corpus100-direct", "सौरमंडल का सबसे बड़ा ग्रह कौन सा है?", {}, [], "direct_response", ["conversational"], None, False, False, [], [])
# 083: hi-Latn - gravity explain karo
add_record(83, "hi-Latn", "latin", "corpus100-direct", "gravity kaise kaam karti hai short mein batao", {}, [], "direct_response", ["conversational"], None, False, False, [], [])
# 084: te - భూమి చుట్టుకొలత (Telugu script)
add_record(84, "te", "telugu", "corpus100-direct", "భూమి పరిమాణం ఎంత?", {}, [], "direct_response", ["conversational"], None, False, False, [], [])
# 085: te-Latn - computer processor enti
add_record(85, "te-Latn", "latin", "corpus100-direct", "CPU ante enti brief ga cheppu", {}, [], "direct_response", ["conversational"], None, False, False, [], [])
# 086: ta - சூரிய குடும்பம் (Tamil script)
add_record(86, "ta", "tamil", "corpus100-direct", "சூரிய குடும்பத்தில் எத்தனை கோள்கள் உள்ளன?", {}, [], "direct_response", ["conversational"], None, False, False, [], [])
# 087: ta-Latn - photosynthesis explain pannu
add_record(87, "ta-Latn", "latin", "corpus100-direct", "photosynthesis enral enna brief-ah sollu", {}, [], "direct_response", ["conversational"], None, False, False, [], [])
# 088: mixed-en-te - code switch conversational
add_record(88, "mixed-en-te", "latin", "corpus100-direct", "Hi JARVIS, how is your performance today?", {}, [], "direct_response", ["conversational"], None, False, False, [], [])

# ==============================================================================
# GROUP 9: Multi-step Guided Workflows (6 records)
# ==============================================================================
# 089: en - multi-step Kubernetes setup guide
add_record(89, "en", "latin", "corpus100-multistep", "Guide me step by step through setting up a Kubernetes cluster on AWS", {}, [CAP_SEARCH], "multi_step_task", ["action", "research"], "google_search", False, False, [], [])
# 090: en - step by step SSL certificate installation
add_record(90, "en", "latin", "corpus100-multistep", "Show me the step by step instructions to configure SSL certificates on Nginx", {}, [CAP_SEARCH, CAP_OPEN_FOLDER], "multi_step_task", ["action", "research"], "google_search", False, False, [], [])
# 091: hi-Latn - steps guide karo for Linux installation
add_record(91, "hi-Latn", "latin", "corpus100-multistep", "Ubuntu Linux dual boot install karne ke steps guide karo", {}, [CAP_SEARCH], "multi_step_task", ["action", "research"], "google_search", False, False, [], [])
# 092: te-Latn - step by step guide cheyyi for GitHub actions
add_record(92, "te-Latn", "latin", "corpus100-multistep", "GitHub actions CI CD setup step by step guide cheyyi", {}, [CAP_SEARCH], "multi_step_task", ["action", "research"], "google_search", False, False, [], [])
# 093: ta-Latn - step by step instructions for Docker setup
add_record(93, "ta-Latn", "latin", "corpus100-multistep", "Docker container deploy panna step by step guide pannu", {}, [CAP_SEARCH], "multi_step_task", ["action", "research"], "google_search", False, False, [], [])
# 094: mixed-en-hi - steps guide karo
add_record(94, "mixed-en-hi", "latin", "corpus100-multistep", "Postgres database backup restore karne ke steps guide karo", {}, [CAP_SEARCH], "multi_step_task", ["action", "research"], "google_search", False, False, [], [])

# ==============================================================================
# GROUP 10: Unsupported Action Escalations (6 records)
# ==============================================================================
# 095: en - order food delivery
add_record(95, "en", "latin", "corpus100-escalate", "Order two chicken burgers and fries from Swiggy to my home", {}, [CAP_SEARCH, CAP_OPEN_FOLDER], "escalate", ["action"], None, False, True, [], [])
# 096: en - transfer money
add_record(96, "en", "latin", "corpus100-escalate", "Transfer five hundred dollars from my savings account to John", {}, [CAP_OPEN_APP], "escalate", ["action"], None, False, True, [], [])
# 097: hi - फ्लाइट टिकट बुक करो (Devanagari)
add_record(97, "hi", "devanagari", "corpus100-escalate", "दिल्ली से मुंबई की फ्लाइट टिकट बुक कर दो", {}, [CAP_SEARCH], "escalate", ["action"], None, False, True, [], [])
# 098: hi-Latn - light band karo
add_record(98, "hi-Latn", "latin", "corpus100-escalate", "bedroom ki AC temperature 22 degree par set karo", {}, [CAP_OPEN_APP], "escalate", ["action"], None, False, True, [], [])
# 099: te-Latn - car book cheyyi
add_record(99, "te-Latn", "latin", "corpus100-escalate", "Uber app lo airport ki cab book cheyyi", {}, [CAP_OPEN_APP], "escalate", ["action"], None, False, True, [], [])
# 100: mixed-en-ta - food order pannu
add_record(100, "mixed-en-ta", "latin", "corpus100-escalate", "Zomato app-la dinner order pannu", {}, [CAP_OPEN_APP], "escalate", ["action"], None, False, True, [], [])

assert len(RECORDS) == 100

# Write to file
corpus_path = OUT / 'unseen_evaluation_corpus_100.jsonl'
with corpus_path.open('w', encoding='utf-8') as f:
    for r in RECORDS:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"Successfully generated {len(RECORDS)} records in {corpus_path}")

# Validate uniqueness of utterances
utts = [r['utterance'].strip().lower() for r in RECORDS]
assert len(utts) == len(set(utts)), f"Duplicate utterances found: {len(utts) - len(set(utts))}"
print("Verified 100% unique utterances.")

