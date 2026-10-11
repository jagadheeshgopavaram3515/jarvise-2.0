# Phase C — Base Qwen Benchmark

## Current facts

The configured endpoint is `http://localhost:11434`; Ollama was reachable. `qwen2.5-coder:3b` is installed as a 3.1B Q4_K_M GGUF. `/api/tags` reports a 32,768-token model context. JARVIS config is model `qwen2.5-coder:3b`, timeout 45 seconds, keep-alive `10m`; classifier timeout is 5 seconds. No model was resident before the raw audit; after inference `/api/ps` showed Qwen resident with 2,155,169,709 bytes VRAM and an active runtime context length of 4096. The tags metadata and active loaded context therefore differ and both are recorded.

## Phase B smoke replay

Dataset: `replay/datasets/smoke.jsonl`  
SHA-256: `82237b8ebf6392da84226e409727400c29aefae0fc42ad9efc6376ec86ff317a`  
Prompt version: `base-v1`  
Run artifact: `results_phasec_smoke_base.json`

| Case | Status | Latency | Parsed result | Evaluator result |
|---|---|---:|---|---|
| `conversation_001` | MODEL_SUCCESS | 22,753.4 ms | JSON, but action `greet` and invented `greeting_tool` | Not scored as task success; schema is benchmark-accepted JSON only |
| `clarify_001` | MODEL_SUCCESS | 4,038.7 ms | Selected `open_file` with `example.txt` | `CLARIFICATION_ERROR` |

The smoke dataset has only two synthetic cases and is not sufficient for general capability or reliability conclusions. It does, however, demonstrate two important current behaviors: Qwen can produce parseable JSON without following the requested action schema semantically, and it may invent a target for an underspecified request.

## Direct raw-Qwen latency observations

These were direct Ollama calls, not voice latency and not full JARVIS latency.

| Prompt | Latency |
|---|---:|
| Explain requirements to open Chrome | 5,773.5 ms |
| Open Chrome | 4,183.8 ms |
| Search latest AI news | 2,556.0 ms |
| Required tools for prior request | 4,183.3 ms |
| JSON action for Chrome | 2,635.6 ms |
| Warm exact `READY`, run 1 | 2,598.5 ms |
| Warm exact `READY`, run 2 | 2,448.3 ms |
| Warm exact `READY`, run 3 | 2,390.4 ms |

The three repeated calls are a sample of three, so no stable p95 is claimed. Warm sample mean is approximately 2,479 ms. Ollama reported nonzero load durations even on repeated calls; residency and exact runtime scheduling should be measured with a larger controlled run.

## Memory/runtime observations

The active `/api/ps` record reported Qwen `size_vram=2,155,169,709` bytes and `context_length=4096`. This is model/runtime VRAM, not total system RAM. No reliable combined idle-RAM/GUI/STT/browser peak was collected in this Phase C run. The previous user observation of approximately 2.2 GB RAM remains a separate observation and must not be treated as total system memory.

## Measurement limitations

The Phase B adapter reuses `OllamaClient` configuration and prompt construction but performs benchmark-local non-streaming transport so errors remain visible. It currently measures adapter wall time and JSON parsing time; it does not yet expose every Ollama duration field in its result schema. The next benchmark iteration should add load, prompt-eval, and generation durations directly from Ollama metadata and separate cold/warm runs.

## Current evidence-based conclusion

Base Qwen is available and can generate structured text, but the available smoke evidence does not establish reliable intent, capability, argument, clarification, or multi-step performance. A meaningful Phase C dataset must be expanded with reviewed, fixture-backed cases before routing requirements or LoRA priorities are finalized.
