# Qwen Capability Boundary & Provider Routing Benchmark

This is an isolated Phase D measurement package. It does not modify production routing, STT, TTS, ToolRegistry permissions, Gemini behavior, or the orchestration graph. It does not train or integrate Fast Brain, BERT, Laya, or LoRA.

The benchmark distinguishes:

1. Raw Qwen through Ollama.
2. Qwen using the current LocalAgent contract and fixture tools.
3. Qwen given a complete Registry capability description in a controlled experiment only.

Existing Phase B replay fixtures and Phase C reports are reused. The current dataset is exploratory and synthetic; it is not a 250–400-case human-reviewed benchmark, so broad reliability conclusions are explicitly marked `INSUFFICIENT EVIDENCE`.

## Current measured environment

`qwen2.5-coder:3b`, Q4_K_M GGUF, 3.1B parameters, Ollama reachable at `http://localhost:11434`. Model metadata reports 32,768 context tokens; active `/api/ps` reported runtime context length 4096 and approximately 2.16 GB VRAM residency. JARVIS uses 45 s Ollama timeout and `10m` keep-alive.

## Existing evidence

- Phase C raw warm sample: 2.390–2.599 s for a tiny exact response, N=3.
- Phase B smoke: 2 synthetic cases; both JSON parse successes, one clarification failure.
- Context probe: 1K–32K requested prompts all returned the target token, but active runtime metadata and prompt token counts indicate the larger requests were not evaluated at the model’s advertised 32K capacity. This is not a semantic context-quality benchmark.

See the reports in this directory for the boundary analysis and limitations.
