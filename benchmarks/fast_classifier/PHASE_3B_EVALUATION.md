# JARVIS FastClassifier Phase 3B Evaluation

Status: offline candidate implemented and measured; **not ready for runtime
integration**.

## Dataset and adjudication

- Immutable v1 artifacts were preserved byte-for-byte.
- Dataset v2 contains 228 rows: 28 retained v1 scenarios and 200 newly generated
  multilingual synthetic candidates.
- All 28 retained IDs have `derived_from_id` traceability and were migrated to
  semantic execution paths.
- Train/dev/test counts: 160/50/18.
- Every paraphrase/translation scenario has one `group_id` and exactly one
  split. No synthetic row is in the sealed test split.
- The 200 additions are template-generated and pending human review.
- Because semantic paths and compositional intents changed, all 28 v2
  adjudications are also pending human review. The authoritative reviewed v1
  rows remain untouched, but v2 has **zero human-approved labels** at present.

Consequently, all v2 quality numbers below are provisional engineering results,
not authoritative multilingual model claims.

## Candidate implemented

`CompactMultilingualAdapter` uses the locally cached
`nlptown/bert-base-multilingual-uncased-sentiment` encoder as a frozen
multilingual feature extractor. The sentiment head is unused. Logistic heads
were actually trained on the 160 synthetic training rows for execution path,
request modes, clarification, escalation, missing-information type, reference
state, and dynamic capability compatibility.

One decision performs one batched encoder forward pass containing the request
and advertised capability descriptions. It never calls Qwen and can select only
from capabilities in `DecisionRequest`. Output is deterministically serialized
and validated with `ClassifierDecision`. No tools are executed.

This is a deviation from the smaller Phase 3A candidate: the cached encoder is a
12-layer multilingual BERT base checkpoint. It was the only usable multilingual
encoder available locally without downloading a model or adding a dependency.

## Environment and method

- OS: Windows 10 build 26200
- CPU: Intel64 Family 6 Model 158, 4 physical / 8 logical cores
- RAM: 8,464,355,328 bytes
- PyTorch: 2.9.0 CPU-only; CUDA unavailable
- Candidate and legacy: separate fresh processes, same 18 test rows, three runs
  with order offsets 0/6/12
- Candidate labels: semantic v2 labels pending human review
- Legacy labels: deterministic projection of representable semantic paths back
  to LOCAL/GEMINI/TASK; three clarification/escalation rows excluded
- Qwen was warmed once after cold loading exceeded its classifier timeout. That
  warm-up took approximately 17.1 seconds wall time and is not included in
  inference latency.

## Candidate quality: sealed test

Denominator is 18 unless specified.

| Metric | Result | Target | Gate |
|---|---:|---:|---|
| Execution-path accuracy | 8/18 = 44.4% | >=85% | FAIL |
| Applicable capability selection | 3/9 = 33.3% | >=90% | FAIL |
| Intent accuracy | 8/18 = 44.4% | >=80% | FAIL |
| Request-mode exact set | 10/18 = 55.6% | >=85% | FAIL |
| Clarification precision | 25.0% | Report | - |
| Clarification recall | 100.0% | Report | - |
| Clarification macro F1 | 0.652 | >=0.85 | FAIL |
| Escalation precision | 0.0% | Report | - |
| Escalation recall | 0.0% | Report | - |
| Escalation macro F1 | 0.419 | >=0.85 | FAIL |
| Context-reference exact match | 12/18 = 66.7% | No gate | - |
| Missing-information exact match | 13/18 = 72.2% | No gate | - |
| Schema validity | 18/18 = 100% | 100% | PASS |
| Fallback-trigger rate | 0/18 = 0% | <=1% | PASS |

The legacy projection scored 3/15 representable paths (20.0%) on the same test
rows. The candidate improved this provisional route score, but remains far below
the acceptance threshold.

### Language and script route accuracy

| Group | N | Accuracy |
|---|---:|---:|
| English | 9 | 33.3% |
| Native script | 4 | 50.0% |
| Romanized | 3 | 33.3% |
| Code-switched | 2 | 100.0% |

No group has the required 20 examples, so the per-language >=75% gate is **not
evaluable**. Individual language counts are one or two outside English; no
multilingual generalization claim is supported.

Context-resolved route accuracy was 0/2. Code-switched route accuracy was 2/2,
but that denominator is too small to interpret. All five supported-capability
test rows missed the execution path, and only three of nine applicable
capability labels were selected correctly.

Representative failures include conversation requests escalated, calculator
actions clarified or treated as direct responses, a resolved continuation
misclassified as memory recall, both unsupported-action examples missed by the
escalation head, and noisy STT escalated instead of selecting `open_app`.

## Latency and memory

Candidate over 54 sealed-test predictions (three runs):

| Measurement | Mean | p50 | p95 |
|---|---:|---:|---:|
| End-to-end adapter inference | 89.179 ms | 77.038 ms | 166.151 ms |
| Encoder model call only | 85.429 ms | 73.934 ms | 162.572 ms |
| Preprocessing | 0.809 ms | 0.878 ms | 1.207 ms |
| Postprocessing | 2.921 ms | 2.674 ms | 4.083 ms |

- First actual inference after startup, observed on development: 306.219 ms.
- Adapter construction/startup: 13,242.140 ms.
- Encoder's internal weight-load interval: 789.617 ms; the larger adapter
  startup includes Python/module and artifact initialization.
- Candidate RSS delta after startup: 333,246,464 bytes (~318 MiB).
- CPU/GPU utilization percentages were not sampled; only device availability,
  process RSS, and wall-clock timing were measured.

Matched warmed legacy local-model calls over 42 model-backed predictions:
mean 2494.354 ms, p50 2464.625 ms, p95 2582.559 ms. The candidate warm p95 is
15.5x faster and below 250 ms, so the warm latency gate passes. The startup cost
is substantial and must be addressed separately.

## Acceptance decision

Passed:

- Structural validity.
- Fallback-trigger rate.
- Zero tool execution in evaluation/tests.
- One encoder inference with no Qwen pre-classification.
- Warm p95 absolute and relative latency targets.

Failed or unavailable:

- Execution path, capability selection, intent, request modes, clarification,
  and escalation quality.
- Per-language quality gate is unavailable due insufficient denominators.
- No v2 labels have completed human review.
- Startup latency is operationally poor, although it was not a numeric Phase 3A
  gate.

The evidence does **not** support proceeding to runtime integration. The next
phase should be dataset review and model iteration, not graph/router wiring.

## Required next work

1. Human-review the 28 adjudications and synthetic language quality; reject or
   correct poor translations before any authoritative test.
2. Add genuinely reviewed examples until each evaluated language/script group
   has a meaningful denominator, while preserving scenario-group isolation.
3. Replace the sentiment-tuned base checkpoint with an authorized compact
   multilingual semantic encoder, or fine-tune an appropriate cached/base model
   after approval.
4. Improve capability contrastive training with hard negatives and add balanced
   unsupported-action/escalation scenarios to development data.
5. Re-run the same sealed protocol only after freezing reviewed labels; do not
   integrate until all quality and latency gates pass together.
