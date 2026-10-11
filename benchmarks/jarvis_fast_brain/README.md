# JARVIS Fast Brain + Qwen Design Phase

This folder is a design-only proposal. It does not change production routing, `graph.py`, STT, TTS, ToolRegistry, Gemini, or the current Qwen classifier. It also does not train or download a LoRA model.

## Proposed architecture

`STT -> Fast Brain -> Qwen (reason/plan) -> Orchestrator -> ToolRegistry/Executor -> observation -> Qwen -> TTS`

The Fast Brain makes a small structured decision. It never executes a tool. Qwen handles local interpretation, planning, tool choice, recovery, and the final answer. The Orchestrator owns state and workflow. ToolRegistry remains the security and execution boundary. Gemini is an explicit escalation provider, not a generic failure catch-all.

## What the repository does today

The current implementation has three Qwen uses: `QwenClassifier` in `orchestration/router.py` returns `TASK`, `LOCAL`, or `GEMINI`; `OllamaClient` generates conversational replies and streams sentence-sized chunks; and `LocalAgent` makes JSON planning/tool-selection/synthesis calls. A separate `assistant/tools/router.py` can ask Gemini to select and phrase a tool call. The current routing graph still only carries the legacy route string.

Read the companion documents in this order:

1. `CURRENT_QWEN_ANALYSIS.md` — source-grounded current state.
2. `ARCHITECTURE.md` — target boundaries and migration principles.
3. `FAST_BRAIN_SPEC.md` — decision contract and Qwen interface.
4. `TRAINING_DATA_SCHEMA.md` — future supervised/trajectory data.
5. `QWEN_LORA_TRAINING_PLAN.md` and `HARDWARE_FEASIBILITY.md` — experiments and constraints.
6. `EVALUATION_PLAN.md` — base-versus-adapter benchmark.

## Status

Confirmed facts are separated from proposals and estimates. The exact next phase is an offline evaluation harness that replays a frozen task set against the existing interfaces without changing production behavior.
