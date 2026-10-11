# Phase B Replay Evaluation

## Scope

Implemented only an offline replay harness under `benchmarks/jarvis_fast_brain/replay/`. No production files were modified by this phase. Existing unrelated working-tree changes were preserved.

## Architecture

Versioned JSONL cases feed the same bounded input shape into a system adapter. `BaseQwenAdapter` uses the existing `OllamaClient` for model configuration and prompt construction, while using benchmark-local transport so raw Ollama failures remain visible. `FastBrainNotImplemented` is an explicit placeholder. Fixture execution is lookup-only.

## Leakage prevention

`model_input()` and the base adapter construct input without reading `case.expected`. Expected labels are added to result records after adapter execution. Tests cover this boundary.

## Metrics and failures

Results retain per-case raw output, parsed output, status, timing, prompt size, exception, expected record, and failure category. Current categories include `OLLAMA_TIMEOUT`, `OLLAMA_ERROR`, `PARSER_ERROR`, `CAPABILITY_MISMATCH`, `QWEN_ARGUMENT_ERROR`, and `CLARIFICATION_ERROR`. Aggregate output intentionally avoids a single weighted score.

## Current benchmark status

No benchmark numbers are claimed here. Base Qwen results require Ollama availability and are generated only by an explicit runner invocation. Fast Brain → Qwen is `NOT_IMPLEMENTED` until a real adapter exists.

## Tests

The next test addition should cover schema validation, expected-label isolation, fixture determinism, failure preservation, and result serialization. The harness itself has not been run against a live Ollama model in this phase.

## Exact next phase

Add focused offline tests and expand the smoke dataset into reviewed train/development and sealed evaluation JSONL, using real Registry capability descriptions and deterministic fixture results. Then run the base replay with documented warm/cold and memory conditions. Do not integrate the Fast Brain or train LoRA yet.
