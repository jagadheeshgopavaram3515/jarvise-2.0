# Future Qwen/LoRA Training Data Schema

Training data should be trajectory records, not only intent labels. Keep raw user text and STT variants, but split by task family and language so paraphrases do not leak across train/test.

```json
{
  "id":"browser_search_001",
  "language":"en",
  "user_input":"Open Chrome and search for the latest AI news.",
  "stt_variants":["open chrome and search latest ai news"],
  "context":{"recent_turns":[],"memory":[],"project_facts":[]},
  "available_capabilities":[{"name":"open_app"},{"name":"google_search"}],
  "fast_brain_decision": {"intent":"search_current_ai_news","execution_path":"QWEN","request_modes":["action","research"],"selected_capability":"google_search"},
  "expected_qwen_plan":["open Chrome if needed","perform Google search"],
  "expected_tool_call":{"name":"google_search","arguments":{"query":"latest AI news"}},
  "observation":{"success":true,"message":"...","data":"bounded result"},
  "expected_next_action":{"kind":"finish"},
  "final_response":"I searched for the latest AI news, sir.",
  "labels":{"requires_clarification":false,"requires_escalation":false}
}
```

Use the actual Registry names only. Examples: `google_search` for “latest AI news”; resolve “that” to the prior `google_search` and change the query to NVIDIA; use `grep_code` then `read_file` for the classifier source; use `google_search`/`read_page` for current Qwen information; ask clarification for “Open it” when no target is in context. Tool arguments must be exact JSON and observations must be real or explicitly synthetic benchmark fixtures.

## Learning categories

Cover JARVIS voice/conversation, intent and capability selection, valid arguments, multi-step location→weather and project inspection, context resolution, memory references, research, browser/desktop/file tasks, error recovery, clarification, escalation, multilingual/code-switched input, and concise final responses. LoRA should teach stable behavior and formatting, not permissions or facts that belong to Registry/runtime.

## Multilingual design

Create independent natural examples in English, Telugu, Hindi, Romanized Telugu, Romanized Hindi, and English/Telugu and English/Hindi code-switching. Include realistic STT spelling, omitted articles, transliterated words, and mixed imperative phrases. Preserve the same underlying task labels but do not mechanically translate every row. Hold out speakers/phrasing and test language generalization separately.
