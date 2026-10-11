# JARVIS FastClassifier Phase 4 Comparison

## 1. Objective and status

This phase compares the existing Phase 3C BERT classifier, Laya, and the legacy
Qwen classifier offline on the same frozen JARVIS-specific transcripts and gold
decision candidates. Production routing, LangGraph, tools, permissions, and STT
were not changed. No capability was executed.

**No winner is declared.** All 434 labels remain pending independent human
review, including native-speaker review. The results are diagnostic rather than
statistically conclusive.

## 2. Dataset

V4 contains 434 examples: 216 train, 78 development, and 140 sealed test.
Train/dev retain v3 rows and groups; the test is entirely new for Phase 4 and
does not reuse the 18 v3 sealed-test IDs.

The test has 20 examples in each slice: English, Telugu native script, Hindi
Devanagari, romanized Telugu, romanized Hindi, English/Telugu code-switching,
and English/Hindi code-switching. It contains 20 semantic scenario groups;
all seven language variants of a scenario remain together in test. Therefore
the 140 rows represent 20 correlated scenarios, not 140 independent scenarios.

Path counts are: single action 56, clarification 21, direct response 14,
research 14, escalation 14, multi-step task 14, and memory recall 7. Advertised
capabilities use names present in the actual default ToolRegistry. Reminder and
email are unsupported-operation cases because the registry exposes neither.

Human-reviewed: **0**. Pending review: **434**. New test rows were manually
authored synthetic candidates independently of model predictions; language
quality has not been verified by native speakers.

## 3. Models and conditions

| Approach | Configuration | Decision interface | Device |
|---|---|---|---|
| BERT | `compact-mbert-frozen-v3`; cached `nlptown/bert-base-multilingual-uncased-sentiment`; frozen encoder and logistic heads | Full structured contract, one batched encoder forward | CPU |
| Laya | `laya==0.3.20`; `convaiinnovations/laya/multilingual`; official mmBERT-base 322M checkpoint | Six typed questions in one `predict` forward, deterministically projected to the contract | CPU |
| Qwen | Existing `QwenClassifier`; `qwen2.5-coder:3b`; configured 7 s Ollama timeout | Legacy `LOCAL/GEMINI/TASK` tuple only | Deterministic CPU path plus attempted Ollama |

Laya was installed in an isolated benchmark virtual environment. Its official
package and API were inspected before implementing the adapter. The first
Unicode run exposed Windows pipe encoding corruption; forcing UTF-8 fixed input
transport without translating or changing any transcript.

All models received the original transcript. BERT and Laya also received the
same context and capability descriptors because their interfaces support them.
The production Qwen interface accepts text only, which is a material interface
limitation and prevents full apples-to-apples structured scoring.

## 4. Core results

| Metric | BERT | Laya | Legacy Qwen |
|---|---:|---:|---:|
| Execution path | 88/140 (62.9%) | 10/140 (7.1%) | 10/80 comparable completed (12.5%) |
| Capability, applicable | 24/84 (28.6%) | 2/84 (2.4%) | unavailable |
| Intent | 68/140 (48.6%) | 27/140 (19.3%) | unavailable |
| Request-mode exact set | 112/140 (80.0%) | 14/140 (10.0%) | unavailable |
| Clarification macro F1 | 0.769 | 0.400 | unavailable |
| Escalation macro F1 | 0.673 | 0.331 | unavailable |
| Strict six-field exact match | 57/140 (40.7%) | 0/140 (0.0%) | unavailable |
| Schema/output validity | 140/140 (100%) | 140/140 (100%) | tuple validity only |
| Fallback-trigger rate | 0/140 | 0/140 | not applicable to legacy tuple |
| Benchmark errors/timeouts | 0/0 | 0/0 | 0/33 |

Strict overall accuracy requires an exact match on execution path, capability,
intent, request modes, clarification, and escalation. It is not computed for
Qwen because its interface exposes none of those fields except a provider route.

Qwen's denominator excludes 27 clarification/escalation examples that its
three-route output cannot represent. More importantly, all 33 attempted Ollama
calls timed out; the 107 completed rows came from deterministic regex fast
paths. Consequently this run does **not** measure live Qwen model accuracy and
cannot support a model-quality ranking against BERT or Laya.

## 5. Language results

Execution-path accuracy by 20-row test slice:

| Slice | BERT | Laya | Qwen completed/comparable only |
|---|---:|---:|---:|
| English | 65% | 10% | 50% (n=2) |
| Telugu native | 60% | 5% | 6.7% (n=15) |
| Hindi native | 55% | 10% | 6.7% (n=15) |
| Romanized Telugu | 40% | 5% | 10% (n=10) |
| Romanized Hindi | 60% | 5% | 25% (n=8) |
| English/Telugu | 80% | 5% | 13.3% (n=15) |
| English/Hindi | 80% | 10% | 13.3% (n=15) |

These are descriptive slices. The rows are correlated translations/variants,
and none has native-speaker review. Qwen slice denominators vary because model
calls timed out and clarification/escalation paths are unrepresentable.

## 6. Latency and memory

| Measurement | BERT | Laya | Qwen |
|---|---:|---:|---:|
| First decision | 919.9 ms | 2,238.0 ms | mixed fast path/model timeout; not comparable |
| Warm p50 | 167.1 ms | 3,039.3 ms | fast path 1.10 ms; no successful Ollama sample |
| Warm p95 | 504.2 ms | 5,065.3 ms | fast path 1.98 ms; no successful Ollama sample |
| Measured load/start boundary | 61,021.8 ms outer / 1,387.0 ms encoder | 15,535.1 ms worker / 12,362.8 ms model | not isolated |
| RSS increase | 334,315,520 bytes | 1,502,556,160 bytes | not measured reliably |

The BERT outer startup includes cold Python/import overhead and is not directly
comparable with the encoder's internal load timer. Laya timings use a persistent
worker and one model forward per example. Both PyTorch environments reported
CUDA unavailable, so the visible GTX 1650 was not used. Continuous CPU/GPU
utilization was not sampled; no utilization percentage is claimed.

## 7. Error analysis

### BERT

- Capability matching remains the largest weakness: only 24/84 applicable
  actions selected the correct advertised capability.
- Romanized Telugu was the weakest path slice at 40%.
- Clarification recall was high (18/21) but precision was 0.50 because 18
  non-clarification examples were also flagged.
- Escalation recognized 6/14 positives and generated 9 false positives.
- Compared with Phase 3C's different 18-row test, path accuracy was 61.1% versus
  62.9% here, while capability was 44.4% versus 28.6%. These are **not controlled
  improvements/regressions** because the datasets differ. Warm p95 was 255.5 ms
  there and 504.2 ms here.

### Laya

- The zero-shot typed-question formulation strongly overpredicted clarification
  or escalation and almost never chose the correct action capability.
- No row matched all six core fields. This does not establish that every Laya
  configuration or fine-tuned Laya model performs poorly; only this official
  base multilingual checkpoint and documented one-forward formulation were tested.
- Schema validity was 100% after the transport fix, showing the deterministic
  projection worked even though semantic quality did not.

### Qwen

- Native Indic and many romanized/code-switched requests were forced to GEMINI
  by deterministic language rules before Qwen inference.
- Every actual Ollama request reached the classifier timeout and the legacy
  implementation fell back to GEMINI; the outer benchmark recorded 33 timeouts.
- Context, capability, clarification, escalation, intent, and mode cannot be
  evaluated through `classify(text)` and were not fabricated.

Representative per-row predictions and failed metric names are retained in each
machine-readable result. Confusion matrices are included under
`execution_path_confusion`.

## 8. Comparability and prior artifacts

All Phase 4 models used the same 140 IDs and original transcripts. Structured
input parity is impossible for legacy Qwen because its production interface is
text-only. The old Qwen 3/15-style baseline and Phase 3B/3C results used other
datasets and denominators, so they are not presented as controlled deltas.
Existing baseline files were preserved.

The v4 test was not used to retrain or tune BERT. No v4 BERT artifact was
created; the exact Phase 3C artifact was evaluated first as required. Laya was
not fine-tuned. Qwen production code was not changed.

## 9. Acceptance and recommendation

Neither structured candidate meets the prior gates. BERT misses execution path,
capability, intent, clarification/escalation F1, and p95 latency targets. Laya
misses all semantic and latency targets. Both pass schema validity and fallback
targets. Qwen cannot be judged as a model because no model-backed call completed.

**NOT READY FOR PRODUCTION INTEGRATION.** The evidence does not support replacing
the production classifier or changing routing.

The next experiment should happen only after independent review/adjudication of
the 20 v4 semantic groups, especially Telugu/Hindi naturalness and capability
labels. Then freeze a reviewed test revision, diagnose Ollama availability, and
rerun Qwen under a controlled idle process. For model work, train or calibrate
one candidate using train/dev only and evaluate once on a newly sealed reviewed
set; do not tune against these exposed v4 test predictions.

## 10. Reproduction and tests

```powershell
python benchmarks/fast_classifier/prepare_v4.py
python -m benchmarks.fast_classifier.run_phase4_comparison bert
python -m benchmarks.fast_classifier.run_phase4_comparison laya
python -m benchmarks.fast_classifier.run_phase4_comparison qwen
python -m benchmarks.fast_classifier.run_phase4_comparison comparison
```

The isolated Laya environment requires `laya==0.3.20`, its official multilingual
checkpoint, and `psutil`.

Final verification:

- `py_compile` passed for all modified Python files.
- `pytest` passed 90/90 tests in 23.66 seconds across the Phase 4 integrity
  tests and existing compact-classifier, benchmark, decision, orchestration,
  and agent suites.
- The real BERT, Laya, and legacy-Qwen benchmark commands all completed and
  produced the four requested machine-readable result files.

