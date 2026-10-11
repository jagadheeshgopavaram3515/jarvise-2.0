# Fast Brain Encoder Adapter v1

Status: offline foundation only.

## Purpose

This adapter represents the selected multilingual encoder architecture and exposes raw classification tensors for later evaluation. It is not connected to the Orchestrator and does not produce a `FastBrainDecision`.

## Implementation

Added:

- `assistant/orchestration/fast_brain/encoder_adapter.py`
- `tests/test_fast_brain_encoder_adapter.py`

No production routing files were modified.

## Architecture

```text
local tokenizer
  → encoder
  → masked mean pooling
  → 256 → 128 projection
  → intent/capability/clarification/missing-information heads
  → raw logits
```

The default maximum sequence length is 160 tokens. The encoder is frozen by default. The adapter performs no calibration, OOD thresholding, execution-path derivation, escalation policy, or training.

## Local-only model loading

The adapter accepts a caller-provided local checkpoint directory. It uses `local_files_only=True` and never calls a download API. Missing artifacts produce `UNAVAILABLE`; malformed local artifacts produce `INVALID`; inference failures produce `ERROR`. No alternate model is silently selected.

The default model identifier is `distilbert-base-multilingual-cased`, but no checkpoint is bundled or downloaded by this step.

## Raw output contract

`EncoderOutput` contains status, raw tensor fields for intent, capability, clarification, and missing-information logits, token count, effective sequence length, truncation state, tokenization latency, encoder latency, total latency, and an error string when unavailable or failed.

Class mappings are sourced directly from the frozen Fast Brain enums: 9 intent classes, 12 capability classes including `unknown`, and 10 missing-information labels.

## Safety and limitations

The adapter does not execute tools, call Qwen/Gemini, browse, access memory, route requests, generate answers, or invent final decisions. Tokenization success for English, Telugu, Hindi, Tamil, romanized, or code-switched text is not semantic accuracy. No approved checkpoint was available for this foundation run, so no semantic or latency claims are made.

## Verification

Focused tests cover unavailable-state behavior, deterministic mappings, max length, masked mean pooling, zero-mask handling, and the absence of fabricated logits/decisions. Existing Fast Brain and Phase 3C tests remain separate and unchanged.

No model was downloaded. No model was trained. No network access was requested by the adapter. No production routing was changed.

## Exact next step

Prepare the human-reviewed Fast Brain training dataset and training/evaluation pipeline. Do not begin that step automatically from this adapter.
