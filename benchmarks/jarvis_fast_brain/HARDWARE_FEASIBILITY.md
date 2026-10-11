# Hardware Feasibility

## Confirmed inputs

Target machine: Intel i5-9300H, 8 GB RAM, GTX 1650 4 GB. User measurement: Qwen 2.5 Coder 3B uses about 2.2 GB RAM while loaded. Python 3.10.11 and LangGraph 1.2.11 are present; the inspected package list did not contain the normal LoRA/QLoRA training packages. Ollama model inventory was not confirmed by the inspection query.

## Design consequences

The 2.2 GB observation is not the whole process footprint: Ollama/runtime overhead, KV cache, Python, STT, browser, GUI, embeddings, and OS memory also count. 8 GB makes simultaneous resident Qwen + training model + STT/browser risky. Keep one reasoning model resident, serialize Qwen requests, bound prompts/observations, and avoid assuming a second classifier model can remain loaded.

The 4 GB GPU may be useful for inference or a highly constrained quantized experiment, but VRAM sufficiency for QLoRA depends on quantization implementation, sequence length, batch size, gradient checkpointing, optimizer, and offload. It is not confirmed feasible here. CPU LoRA training on an i5-9300H is theoretically possible for tiny experiments but likely slow and memory-sensitive; no timing estimate is treated as a guarantee.

## Required measurements before any training

Record idle RAM/VRAM; Ollama-loaded Qwen; STT warm; TTS; browser; GUI; embeddings; and combined peak during one request. Then measure prompt length, generation latency, concurrency behavior, and model unload/reload time. A training smoke test must record peak RAM/VRAM and terminate safely before production use.

## Feasibility verdict

Inference and design work are plausible under memory discipline. A practical local LoRA/QLoRA training workflow is unverified and should be treated as an experiment requiring a separate environment or stronger hardware until the stack and peaks are measured. Do not load base and adapter models simultaneously on this machine by assumption.
