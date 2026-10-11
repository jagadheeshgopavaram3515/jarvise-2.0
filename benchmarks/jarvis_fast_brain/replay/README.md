# Phase B Replay Harness

This is an independent, offline benchmark boundary. It does not start JARVIS, invoke real tools, use the browser, access the network through fixtures, or modify production code.

## Run

```powershell
python -m benchmarks.jarvis_fast_brain.replay.runner --dataset benchmarks/jarvis_fast_brain/replay/datasets/smoke.jsonl --system base_qwen
python -m benchmarks.jarvis_fast_brain.replay.runner --dataset benchmarks/jarvis_fast_brain/replay/datasets/smoke.jsonl --system fast_brain_qwen
```

The first command requires a reachable Ollama server and records `OLLAMA_ERROR`/`OLLAMA_TIMEOUT` instead of converting failures into Gemini success. The second reports `NOT_IMPLEMENTED`; it never fabricates Fast Brain results.

Expected labels are evaluator-only. The adapter receives request, bounded context, capabilities, and observations; expected fields are attached only after model output. Fixtures are lookup-only and cannot execute Registry handlers.
