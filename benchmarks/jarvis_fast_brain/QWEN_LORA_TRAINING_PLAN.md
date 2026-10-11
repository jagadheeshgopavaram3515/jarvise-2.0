# Qwen LoRA/QLoRA Training Plan (No Training Yet)

## Objective

Teach Qwen JARVIS-specific behavior: compact plans, Registry-grounded tool calls, bounded observations, clarification, recovery, escalation signals, multilingual understanding, and concise spoken answers. Do not train permissions, executable policy, or hidden tool access into weights.

## Proposed experiment

Freeze a base Qwen 2.5 Coder 3B reference and a versioned trajectory dataset. Train a small adapter only after the dataset and evaluation harness exist. Compare base and adapter with identical prompts, capabilities, fixtures, decoding settings, and hardware. Keep adapters separate first; merge/export only if serving compatibility is demonstrated.

## Required stack and checks

A practical run would normally need PyTorch, Transformers, PEFT, Accelerate, Datasets, and possibly TRL/bitsandbytes. The inspected environment did not show those packages. Compatibility must be checked against the exact Qwen checkpoint format, CUDA/PyTorch build, and license before installation. No stack was installed during this phase.

Ollama serves its own model packaging; an adapter trained elsewhere is not automatically an Ollama model. A future experiment must prove one of: Ollama supports the adapter format directly for this model, or the adapter is merged/exported to a compatible GGUF/Modelfile path. This is currently unconfirmed.

## Training sequence

1. Validate dataset schema and fixtures offline.
2. Establish base Qwen benchmark and memory/latency baseline.
3. Run a tiny overfit/smoke experiment, if hardware permits.
4. Train LoRA first; attempt QLoRA only with a verified CUDA toolchain.
5. Export adapter/merged artifact in a disposable benchmark directory.
6. Serve it separately from production and replay the same evaluation set.

## Do not train yet because

The current environment lacks the verified training stack, the Ollama serving route is unconfirmed, and production responsibilities are still split between legacy Qwen and Gemini tool selection. Training before fixing the measurement boundary could encode the wrong architecture.
