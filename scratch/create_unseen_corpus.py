"""Authoring a diverse, newly authored unseen evaluation corpus (24 records).
Features diverse domains, multilingual scripts, ambiguous queries, and memory references.
Labels established independently prior to model scoring. Label status: PROVISIONAL_INDEPENDENT_AUTHORED.
"""
import json
from pathlib import Path

OUT = Path('D:/JARVIS/reports/classifier_v4_experiment_20261010_175140')

RECORDS = [
    # 1. Folder operation (English) - Open downloads directory
    {
        "id": "unseen-01-en-folder",
        "language": "en",
        "script": "latin",
        "utterance": "Open my downloads folder right now",
        "context": {},
        "capabilities": [
            {"name": "open_folder", "description": "Open a specified directory or folder.", "required_arguments": ["path"]},
            {"name": "find_file", "description": "Find files matching a query.", "required_arguments": ["query"]},
            {"name": "google_search", "description": "Perform web search.", "required_arguments": ["query"]}
        ],
        "expected": {
            "intent": "use:open_folder",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "open_folder",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 2. File operation (Hindi Latin) - Find presentation file
    {
        "id": "unseen-02-hi-file",
        "language": "hi-Latn",
        "script": "latin",
        "utterance": "budget presentation file dhoondo",
        "context": {},
        "capabilities": [
            {"name": "find_file", "description": "Find files matching a query.", "required_arguments": ["query"]},
            {"name": "open_folder", "description": "Open a specified directory or folder.", "required_arguments": ["path"]},
            {"name": "close_app", "description": "Close a running application.", "required_arguments": ["name"]}
        ],
        "expected": {
            "intent": "use:find_file",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "find_file",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 3. Web Search (Telugu Script) - Latest news search
    {
        "id": "unseen-03-te-search",
        "language": "te",
        "script": "telugu",
        "utterance": "తాజా వాతావరణ వార్తలు గూగుల్ లో వెతుకు",
        "context": {},
        "capabilities": [
            {"name": "google_search", "description": "Perform a Google search for query.", "required_arguments": ["query"]},
            {"name": "find_file", "description": "Find local files.", "required_arguments": ["query"]},
            {"name": "open_folder", "description": "Open a folder.", "required_arguments": ["path"]}
        ],
        "expected": {
            "intent": "use:google_search",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "google_search",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 4. Web Search (Tamil Latin) - Python tutorial search
    {
        "id": "unseen-04-ta-search",
        "language": "ta-Latn",
        "script": "latin",
        "utterance": "python async programming tutorials search pannu",
        "context": {},
        "capabilities": [
            {"name": "google_search", "description": "Perform web search for tutorial or query.", "required_arguments": ["query"]},
            {"name": "read_page", "description": "Read URL webpage.", "required_arguments": ["url"]}
        ],
        "expected": {
            "intent": "use:google_search",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "google_search",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 5. Feasibility check: Read page with valid URL (English)
    {
        "id": "unseen-05-en-readpage",
        "language": "en",
        "script": "latin",
        "utterance": "Read the documentation at https://fastapi.tiangolo.com/tutorial",
        "context": {},
        "capabilities": [
            {"name": "read_page", "description": "Read a webpage from a URL.", "required_arguments": ["url"]},
            {"name": "google_search", "description": "Search the web.", "required_arguments": ["query"]}
        ],
        "expected": {
            "intent": "use:read_page",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "read_page",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 6. Feasibility check: Read page MISSING URL (English) -> Clarify missing URL argument
    {
        "id": "unseen-06-en-readpage-missing",
        "language": "en",
        "script": "latin",
        "utterance": "Read the webpage for me please",
        "context": {},
        "capabilities": [
            {"name": "read_page", "description": "Read a webpage from a URL.", "required_arguments": ["url"]}
        ],
        "expected": {
            "intent": "clarify_request",
            "execution_path": "clarify",
            "request_modes": ["action"],
            "selected_capability": None,
            "needs_clarification": True,
            "needs_escalation": False,
            "missing_information": [{"name": "target", "description": "The requested target is not identifiable"}],
            "context_references": []
        }
    },
    # 7. Ambiguous action without target (Hindi Devanagari) - "इसे हटा दो" (Delete it)
    {
        "id": "unseen-07-hi-ambiguous",
        "language": "hi",
        "script": "devanagari",
        "utterance": "इसे हटा दो",
        "context": {},
        "capabilities": [
            {"name": "delete_file", "description": "Delete a file.", "required_arguments": ["path"]},
            {"name": "delete_folder", "description": "Delete a folder.", "required_arguments": ["path"]}
        ],
        "expected": {
            "intent": "clarify_request",
            "execution_path": "clarify",
            "request_modes": ["action"],
            "selected_capability": None,
            "needs_clarification": True,
            "needs_escalation": False,
            "missing_information": [{"name": "target", "description": "The requested target is not identifiable"}],
            "context_references": [{"kind": "item", "reference": "it", "resolved_value": None}]
        }
    },
    # 8. Ambiguous action without target (Telugu Latin) - "danni open cheyyi" (Open it)
    {
        "id": "unseen-08-te-ambiguous",
        "language": "te-Latn",
        "script": "latin",
        "utterance": "danni ventane open cheyyi",
        "context": {},
        "capabilities": [
            {"name": "open_file", "description": "Open a file.", "required_arguments": ["path"]},
            {"name": "open_folder", "description": "Open a directory.", "required_arguments": ["path"]}
        ],
        "expected": {
            "intent": "clarify_request",
            "execution_path": "clarify",
            "request_modes": ["action"],
            "selected_capability": None,
            "needs_clarification": True,
            "needs_escalation": False,
            "missing_information": [{"name": "target", "description": "The requested target is not identifiable"}],
            "context_references": [{"kind": "item", "reference": "it", "resolved_value": None}]
        }
    },
    # 9. Ambiguous action without target (Tamil Script) - "அதை திற" (Open it)
    {
        "id": "unseen-09-ta-ambiguous",
        "language": "ta",
        "script": "tamil",
        "utterance": "அதை உடனே திற",
        "context": {},
        "capabilities": [
            {"name": "open_file", "description": "Open a file.", "required_arguments": ["path"]},
            {"name": "open_folder", "description": "Open a directory.", "required_arguments": ["path"]}
        ],
        "expected": {
            "intent": "clarify_request",
            "execution_path": "clarify",
            "request_modes": ["action"],
            "selected_capability": None,
            "needs_clarification": True,
            "needs_escalation": False,
            "missing_information": [{"name": "target", "description": "The requested target is not identifiable"}],
            "context_references": [{"kind": "item", "reference": "it", "resolved_value": None}]
        }
    },
    # 10. Memory Recall: Unseen project alias (English) - "Project Chimera"
    {
        "id": "unseen-10-en-memory-alias",
        "language": "en",
        "script": "latin",
        "utterance": "Do you remember our Chimera project?",
        "context": {
            "projects": [{"name": "Stealth Drone", "aliases": ["Chimera project"]}]
        },
        "capabilities": [
            {"name": "google_search", "description": "Web search.", "required_arguments": ["query"]}
        ],
        "expected": {
            "intent": "recall",
            "execution_path": "memory_recall",
            "request_modes": ["memory"],
            "selected_capability": None,
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": [{"kind": "project", "reference": "our Chimera project", "resolved_value": "Stealth Drone"}]
        }
    },
    # 11. Memory Recall: Topic decision (Hindi-English Code-Switch) - Database choice
    {
        "id": "unseen-11-hi-codeswitch-memory",
        "language": "mixed-en-hi",
        "script": "latin",
        "utterance": "Postgres migration ke baare mein humne kya finalize kiya tha?",
        "context": {
            "recent_topics": [{"topic": "Postgres migration", "decision": "schedule migration on Saturday midnight"}]
        },
        "capabilities": [],
        "expected": {
            "intent": "recall",
            "execution_path": "memory_recall",
            "request_modes": ["memory"],
            "selected_capability": None,
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": [{"kind": "prior_conversation", "reference": "Postgres migration", "resolved_value": "schedule migration on Saturday midnight"}]
        }
    },
    # 12. Memory Recall: Topic decision (Telugu-English Code-Switch) - Auth service
    {
        "id": "unseen-12-te-codeswitch-memory",
        "language": "mixed-en-te",
        "script": "latin",
        "utterance": "Auth service gurinchi mana decision enti?",
        "context": {
            "recent_topics": [{"topic": "Auth service", "decision": "use JWT with short expiry"}]
        },
        "capabilities": [],
        "expected": {
            "intent": "recall",
            "execution_path": "memory_recall",
            "request_modes": ["memory"],
            "selected_capability": None,
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": [{"kind": "prior_conversation", "reference": "Auth service", "resolved_value": "use JWT with short expiry"}]
        }
    },
    # 13. Direct Response: Factual question (English)
    {
        "id": "unseen-13-en-direct",
        "language": "en",
        "script": "latin",
        "utterance": "What is the capital city of Australia?",
        "context": {},
        "capabilities": [
            {"name": "open_folder", "description": "Open a folder.", "required_arguments": ["path"]}
        ],
        "expected": {
            "intent": "respond",
            "execution_path": "direct_response",
            "request_modes": ["conversational"],
            "selected_capability": None,
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 14. Direct Response: Factual question (Telugu Script)
    {
        "id": "unseen-14-te-direct",
        "language": "te",
        "script": "telugu",
        "utterance": "సూర్యుడు ఏ దిక్కున ఉదయిస్తాడు?",
        "context": {},
        "capabilities": [],
        "expected": {
            "intent": "respond",
            "execution_path": "direct_response",
            "request_modes": ["conversational"],
            "selected_capability": None,
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 15. Direct Response: Factual question (Hindi Devanagari)
    {
        "id": "unseen-15-hi-direct",
        "language": "hi",
        "script": "devanagari",
        "utterance": "प्रकाश की गति कितनी होती है?",
        "context": {},
        "capabilities": [],
        "expected": {
            "intent": "respond",
            "execution_path": "direct_response",
            "request_modes": ["conversational"],
            "selected_capability": None,
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 16. Multi-step task: Setup guide (English)
    {
        "id": "unseen-16-en-multistep",
        "language": "en",
        "script": "latin",
        "utterance": "Guide me step by step through deploying a docker container",
        "context": {},
        "capabilities": [
            {"name": "google_search", "description": "Search web for guides.", "required_arguments": ["query"]}
        ],
        "expected": {
            "intent": "use:google_search",
            "execution_path": "multi_step_task",
            "request_modes": ["action", "research"],
            "selected_capability": "google_search",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 17. Multi-step task: Guide in Hindi Latin
    {
        "id": "unseen-17-hi-multistep",
        "language": "hi-Latn",
        "script": "latin",
        "utterance": "react app create karne ke steps guide karo",
        "context": {},
        "capabilities": [
            {"name": "google_search", "description": "Search web for guides.", "required_arguments": ["query"]}
        ],
        "expected": {
            "intent": "use:google_search",
            "execution_path": "multi_step_task",
            "request_modes": ["action", "research"],
            "selected_capability": "google_search",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 18. Escalate: Unsupported domain (Physical world action in English)
    {
        "id": "unseen-18-en-escalate",
        "language": "en",
        "script": "latin",
        "utterance": "Order a pepperoni pizza from Dominoes to my address",
        "context": {},
        "capabilities": [
            {"name": "open_folder", "description": "Open folder.", "required_arguments": ["path"]},
            {"name": "google_search", "description": "Web search.", "required_arguments": ["query"]}
        ],
        "expected": {
            "intent": "unsupported_request",
            "execution_path": "escalate",
            "request_modes": ["action"],
            "selected_capability": None,
            "needs_clarification": False,
            "needs_escalation": True,
            "missing_information": [],
            "context_references": []
        }
    },
    # 19. Close App (Telugu Latin) - WhatsApp close
    {
        "id": "unseen-19-te-closeapp",
        "language": "te-Latn",
        "script": "latin",
        "utterance": "WhatsApp application close cheyyi",
        "context": {},
        "capabilities": [
            {"name": "close_app", "description": "Close a running desktop application.", "required_arguments": ["name"]},
            {"name": "open_folder", "description": "Open a folder.", "required_arguments": ["path"]}
        ],
        "expected": {
            "intent": "use:close_app",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "close_app",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 20. Open App (Tamil Latin) - VS Code open
    {
        "id": "unseen-20-ta-openapp",
        "language": "ta-Latn",
        "script": "latin",
        "utterance": "VS Code application open pannu",
        "context": {},
        "capabilities": [
            {"name": "open_app", "description": "Open an installed desktop application.", "required_arguments": ["name"]},
            {"name": "find_file", "description": "Find files.", "required_arguments": ["query"]}
        ],
        "expected": {
            "intent": "use:open_app",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "open_app",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 21. Open Folder (Tamil Latin) - Screenshots folder
    {
        "id": "unseen-21-ta-folder",
        "language": "ta-Latn",
        "script": "latin",
        "utterance": "screenshots directory open pannu",
        "context": {},
        "capabilities": [
            {"name": "open_folder", "description": "Open directory.", "required_arguments": ["path"]},
            {"name": "google_search", "description": "Web search.", "required_arguments": ["query"]}
        ],
        "expected": {
            "intent": "use:open_folder",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "open_folder",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 22. Find File (Telugu Latin) - Invoice PDF
    {
        "id": "unseen-22-te-file",
        "language": "te-Latn",
        "script": "latin",
        "utterance": "october invoice pdf document vetuku",
        "context": {},
        "capabilities": [
            {"name": "find_file", "description": "Find local files or documents.", "required_arguments": ["query"]},
            {"name": "open_folder", "description": "Open directory.", "required_arguments": ["path"]}
        ],
        "expected": {
            "intent": "use:find_file",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "find_file",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 23. Direct Response: Conversational greeting (Hindi Latin)
    {
        "id": "unseen-23-hi-greeting",
        "language": "hi-Latn",
        "script": "latin",
        "utterance": "Namaste JARVIS, aap kaise ho?",
        "context": {},
        "capabilities": [],
        "expected": {
            "intent": "respond",
            "execution_path": "direct_response",
            "request_modes": ["conversational"],
            "selected_capability": None,
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": []
        }
    },
    # 24. Escalate: Unsupported hardware control (English)
    {
        "id": "unseen-24-en-hardware",
        "language": "en",
        "script": "latin",
        "utterance": "Turn on the living room ceiling fan",
        "context": {},
        "capabilities": [
            {"name": "open_folder", "description": "Open directory.", "required_arguments": ["path"]}
        ],
        "expected": {
            "intent": "unsupported_request",
            "execution_path": "escalate",
            "request_modes": ["action"],
            "selected_capability": None,
            "needs_clarification": False,
            "needs_escalation": True,
            "missing_information": [],
            "context_references": []
        }
    }
]

with (OUT / 'unseen_evaluation_corpus_v4.jsonl').open('w', encoding='utf-8') as f:
    for r in RECORDS:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"Successfully created unseen evaluation corpus with {len(RECORDS)} records in {OUT}")

