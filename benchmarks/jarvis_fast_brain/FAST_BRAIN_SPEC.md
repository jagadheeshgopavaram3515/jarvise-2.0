# JARVIS Fast Brain Specification

## Reuse the existing contract

`assistant/orchestration/decision.py` already provides `DecisionRequest`, `ClassifierDecision`, `RequestMode`, `Capability`, `ContextReference`, `MissingInformation`, confidence, clarification, escalation, and a replaceable `DecisionModelAdapter`. It is sufficient for the design. No incompatible schema is needed.

## Fast Brain output

The canonical output is the existing `ClassifierDecision`:

```json
{
  "intent": "search_current_ai_news",
  "execution_path": "QWEN",
  "request_modes": ["action", "research"],
  "selected_capability": "google_search",
  "context_references": [],
  "missing_information": [],
  "confidence": 0.91,
  "needs_clarification": false,
  "needs_escalation": false,
  "evidence": [{"kind":"cue","reference":"latest/current information"}]
}
```

`execution_path` is a proposed semantic path. Existing compatibility callers can map `QWEN` to the future local path while legacy `LOCAL/GEMINI/TASK` remain supported during migration. This mapping is a future adapter decision, not a production change in this phase.

## What Qwen receives

Pass only:

```json
{
  "request": "Do that again but search for NVIDIA.",
  "decision": {
    "intent": "repeat_previous_browser_search",
    "request_modes": ["action"],
    "selected_capability": "google_search",
    "context_references": [{"reference":"that","kind":"previous_action","resolved_value":"google_search for latest AI news"}],
    "missing_information": [],
    "needs_clarification": false,
    "needs_escalation": false
  },
  "relevant_context": "one bounded prior action/result",
  "capability": {"name":"google_search","description":"Open a Google search...","parameters":{"...":"..."}},
  "observations": []
}
```

Internal-only fields are adapter ID, raw scores/logits, timing, model version, and diagnostic evidence not needed for reasoning. Include capability descriptions only for the selected family/candidates; do not send every schema by default. Include bounded context references and resolved values, not the entire conversation or vector store. The orchestrator must validate Qwen actions independently.

## Clarification and escalation

Clarification is preferred when `needs_clarification=true` and missing information is actionable, such as `Open it` with no resolvable target. Escalation is a proposal only; the Orchestrator requires evidence and policy before invoking Gemini. A tool failure alone is not sufficient.
