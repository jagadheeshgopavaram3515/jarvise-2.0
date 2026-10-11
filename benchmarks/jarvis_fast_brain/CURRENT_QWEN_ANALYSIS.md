# Current Qwen Analysis

## Verified source map

| Area | Current implementation |
|---|---|
| Initialization | `config.OLLAMA_MODEL` defaults to `qwen2.5-coder:3b`; `OllamaClient`, `QwenClassifier`, and `LocalAgent` each hold an Ollama URL/model. `main.py` warms the local model. |
| Classifier | `router.py` deterministic regex fast path, then `/api/generate`, `stream:false`, `num_predict:5`, temperature 0, stop tokens, classifier timeout 5 s. Output is one of `TASK/LOCAL/GEMINI`; invalid/error defaults to Gemini. |
| Conversation | `OllamaClient._build_prompt` adds JARVIS system prompt, current date/time, selected memory, recent history, optional real-time briefing, and the user text. `stream()` uses Ollama streaming and chunks sentences/word groups. |
| Planning | `LocalAgent.create_plan` sends a JSON-mode Qwen request with goal and up to three history turns. It asks for a 1–3 step plan. |
| Tool selection | `LocalAgent.select_action` sends goal, plan, completed steps, last observation, errors, recent history, and a manually selected six-tool list. It validates the returned name against Registry. Separately, `tools/router.py` may use Gemini for selection across the broader Registry. |
| Execution | `LocalAgent.execute_tool` calls `ToolExecutor`; executor validates registry membership, runs in a worker pool, enforces a 25 s default timeout, normalizes results, and audits calls. |
| Observation | `format_observation` bounds/normalizes results for Qwen. The graph loops through select → execute → observe, increments `retry_count`, and stops at default `max_iterations=8`. |
| Recovery | Qwen sees errors and observations on the next selection call. Plan-generation failure uses a fallback plan; malformed action output becomes an error. No explicit evidence-based Gemini escalation is in the local agent loop. |
| Memory | Conversation Ollama prompts use `store.load_history`, `format_history`, and `format_memory_context`; recall intent changes memory retrieval. Agent prompts only receive caller-supplied history, not the full memory engine context. |
| Project context | Agent system prompt hard-codes repository layout. Project facts arrive only through `grep_code`, `read_file`, `list_directory`, and test observations. |
| Multilingual | STT accepts English/Telugu/Hindi/Tamil scripts and code switching; legacy router sends Indic/romanized indicators to Gemini. Qwen prompt itself has no dedicated language contract. |
| Concurrency | `ToolExecutor` has a four-worker pool, but no Qwen serialization/queue is evident. Requests can therefore overlap if callers overlap. This needs measurement before relying on it. |
| Streaming | Conversation `OllamaClient` streams; classifier and all `LocalAgent` JSON calls are non-streaming. |

## Configuration facts

`OLLAMA_TIMEOUT_S=45`, `OLLAMA_KEEP_ALIVE=10m`, model `qwen2.5-coder:3b`, classifier timeout 5 s. The classifier has `num_predict=5`; agent JSON calls set temperature 0.1 but no explicit output cap; final agent synthesis sets temperature 0.2 and `num_predict=250`. Conversation sets temperature 0.3 and no explicit output cap. Context length is not set in the Ollama payload, so Ollama/model defaults apply; this is unknown, not a measured context limit.

## Current flow

```text
input text
  -> QwenClassifier fast regex OR Qwen one-word request
  -> LOCAL: OllamaClient prompt -> streamed TTS-compatible chunks
  -> TASK: LangGraph understand -> Qwen plan -> Qwen action JSON
          -> ToolExecutor/Registry -> bounded observation -> Qwen action JSON ...
          -> Qwen final synthesis
  -> GEMINI: GeminiClient streaming path
```

## Current limitations

The legacy route loses intent details, capability, uncertainty, missing information, and context references. The graph does not consume the richer `ClassifierDecision` contract. Tool capability exposure is inconsistent: the agent manually advertises six tools, while the separate Gemini selector can see the Registry. Current router errors route to Gemini even when the failure is local infrastructure rather than semantic need. Real-time retrieval is embedded in provider prompting, and Qwen latency is spent in classifier generation, plan generation, repeated action calls, final synthesis, and optional context retrieval.

## Environment inspection

Python is 3.10.11. LangGraph 1.2.11 is installed. No `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`, or `accelerate` package was found in the inspected `pip list`. A usable Ollama `/api/tags` inventory was not confirmed during inspection. These are environment observations, not claims that the model or server can never be available.
