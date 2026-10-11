# Runbook

Run from `D:\JARVIS`.

```powershell
python -m pytest -q benchmarks/jarvis_fast_brain/replay/tests
python -m benchmarks.jarvis_fast_brain.replay.runner --dataset benchmarks/jarvis_fast_brain/replay/datasets/smoke.jsonl --system base_qwen
```

Raw Ollama probes must use the configured `qwen2.5-coder:3b`, record the request, model metadata, response, Ollama duration fields, and `/api/ps` residency. Do not call real Registry handlers. Use the existing lookup-only fixture executor for agent trajectories.

For context stress, record requested prompt size separately from `prompt_eval_count` and active runtime context. A successful response at a requested 32K prompt is not evidence that JARVIS evaluated 32K tokens.
