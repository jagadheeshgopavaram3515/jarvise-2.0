# JARVIS FastClassifier Phase 3C Evaluation

## 1. Objective

Review v2 labels, improve focused train/dev coverage, improve and retrain the
offline frozen-mBERT candidate, and measure it without changing production.
No model was downloaded, no tool was executed, and no runtime route changed.

## 2. Starting Phase 3B state

The saved Phase 3B result used 160 synthetic training rows and an 18-row sealed
test. It achieved 8/18 execution paths, 3/9 applicable capabilities, 8/18
intents, 10/18 exact mode sets, clarification macro F1 0.652, escalation macro
F1 0.419, and warm p95 166.151 ms. Labels were pending human review.

## 3. Files inspected and implementation map

Inspection covered the decision contract, compact adapter, v2 preparation and
training, evaluator and runner, all 228 v2 records, manifests/adjudication/model
card, both Phase 3B reports, classifier tests, and the immediate ToolRegistry.

- Input: utterance plus bounded JSON context; advertised capabilities are
  descriptor-only and contain no handlers.
- Encoder: frozen mean-pooled mBERT; sentiment head unused.
- Heads: balanced logistic path, four mode, clarification, escalation,
  missing-field, reference-kind/resolution, and capability-pair classifiers.
- Intent: deterministically composed from path and selected capability.
- Artifact: compressed joblib heads plus JSON metadata and exact training IDs.
- Evaluation: exact field metrics, binary macro F1, schema/fallback rates,
  grouped path accuracy, and `perf_counter_ns` latency.

## 4. Dataset review findings

All 228 records were reviewed. The primary structural problem was scenario
collapse: each synthetic class was largely represented by one topic family.
Dev omitted direct-response and escalation. Test omitted memory and multistep
and was too small for language claims. `find_file` descriptors disagreed with
the real registry (`name` versus `filename`). Music playback was labeled as
executable through a search-only capability. Details and unresolved annotation
questions are in `v2/PHASE_3C_ADJUDICATION.md`.

## 5. Label corrections

- `en-music-001`: single `youtube_search` action to unsupported escalation.
- `cap-unsupported-email-001`: removed irrelevant missing recipient address
  from an unavailable-capability escalation.
- 140 records: corrected the repeated `find_file` argument descriptor to
  `filename`; this is metadata, not 140 semantic relabels.

Every affected ID and before/after value is in the v3 manifest. Corrections
were determined before the Phase 3C sealed run and were not based on that run.

## 6. New and modified examples

V3 derives all 228 v2 records and adds 84 manually authored synthetic
candidates: 56 train and 28 dev. Additions cover all seven paths, additional
capabilities, distinct unsupported actions, native scripts, romanization,
code-switching, and context. They are assistant-reviewed and explicitly remain
pending independent human/native-speaker review. None is called human-reviewed.

## 7. Split and test integrity

| Split | Examples | Use |
|---|---:|---|
| Train | 216 | Head fitting only |
| Dev | 78 | Experiment selection only |
| Sealed test | 18 | One final selected-model evaluation |

Scenario groups never cross splits. Test IDs and execution-path labels are the
same v2 sealed records; `cap-unsupported-email-001` has the pre-declared
non-route missing-information correction. No test ID appears in model metadata.
The test remains small and was already used in Phase 3B, so it supports a
controlled iteration comparison, not an unbiased population estimate.

## 8. Model changes

The encoder remains frozen. Phase 3C tested joint request/capability encoding,
then rejected it due to quality/latency. The selected model keeps independent
embeddings and pair features, uses the improved v3 training data, balanced
heads, and `C=4.0`. It performs one batched BERT forward per decision and never
calls Qwen. Production continues using its existing classifier.

## 9. Training configuration

- Model: cached `nlptown/bert-base-multilingual-uncased-sentiment`
- Seed: 42; sklearn logistic regression; balanced class weights; `C=4.0`
- Max length: 160; mode/capability thresholds: 0.5
- Training examples: 216; encoder frozen; CPU only
- Selected training time: 22,247.8 ms; encoder load during training: 686.4 ms
- Training RSS: 235,048,960 to 890,978,304 bytes. This includes training
  embeddings and is not inference memory.

## 10. Experiments

The current v2 model first scored 19/50 paths and 2/25 capabilities on v2 dev.
Joint-pair v3 experiments used `C=0.25`, `1.0`, and `4.0`. Their dev path
accuracies were 33.3%, 26.9%, and 44.9%; capability accuracy was 16.7% for all.
Warm p95 ranged from 494 to 1,319 ms. A capability-threshold-zero diagnostic did
not improve ranking. The independent `C=4.0` experiment reached 60.3% path
accuracy before the final targeted data addition. No encoder fine-tuning was
attempted because the smaller representation/data change was not yet exhausted.

## 11. Best-model selection

Independent `C=4.0` was selected using dev only because it materially improved
path/mode/clarification quality while preserving the shorter Phase 3B input.
After adding targeted training-only coverage, the frozen configuration was
retrained once and was not changed after the sealed evaluation.

Final dev confirmation: path 61/78 (78.2%), capability 12/36 (33.3%), intent
49/78 (62.8%), modes 65/78 (83.3%), clarification macro F1 0.900, escalation
macro F1 0.819. The gap to test shows substantial scenario overfitting.

## 12. Final sealed benchmark

| Metric | Phase 3C result |
|---|---:|
| Execution path | 11/18 (61.1%) |
| Applicable capability | 4/9 (44.4%) |
| Intent | 11/18 (61.1%) |
| Request-mode exact set | 14/18 (77.8%) |
| Clarification macro F1 | 0.719 (1 positive) |
| Escalation macro F1 | 0.652 (2 positives) |
| Schema validity | 18/18 (100%) |
| Fallback rate | 0/18 (0%) |

## 13. Controlled Phase 3B comparison

The test examples are the same, so count differences are comparable, although
two labels were adjudicated in v3 and the sample remains very small.

| Metric | Phase 3B | Phase 3C | Difference |
|---|---:|---:|---:|
| Execution path | 8/18 | 11/18 | +3, +16.7 points |
| Capability | 3/9 | 4/9 | +1, +11.1 points |
| Intent | 8/18 | 11/18 | +3, +16.7 points |
| Modes | 10/18 | 14/18 | +4, +22.2 points |
| Clarification macro F1 | 0.652 | 0.719 | +0.067 |
| Escalation macro F1 | 0.419 | 0.652 | +0.232 |
| Warm p95 | 166.151 ms | 255.465 ms | +89.315 ms regression |

These are observed seed-set changes, not statistical evidence of generalization.

## 14. Legacy Qwen comparison

The preserved matched Qwen report scored 3/15 (20.0%) on legacy-representable
projected routes, with model-backed p95 2,582.6 ms. It cannot emit or be scored
for structured fields. The Phase 3C semantic 11/18 result has a different output
space and denominator, so it is not presented as direct accuracy superiority.
No new Qwen run was needed because the sealed examples were retained.

## 15. Multilingual results

Descriptive path results were English 6/9, native script 2/4, romanized 1/3,
and code-switched 2/2. Individual language cells contain only one or two
examples outside English. No multilingual performance claim is supported.

## 16. Latency

Across 54 rotated test predictions, end-to-end mean was 115.741 ms, p50 87.179
ms, and p95 255.465 ms. Adapter startup was 11,351.8 ms; internal encoder load
was 830.9 ms. Inference-only component measurements are retained in the JSON
report. The p95 narrowly misses the 250 ms target and regressed from Phase 3B.

## 17. Memory

Inference-process RSS rose from 138,727,424 to 472,596,480 bytes, a 333,869,056
byte (318.4 MiB) increase. CPU utilization was not sampled; GPU use was absent.

## 18. Tests

`python -m pytest tests/test_compact_classifier.py tests/test_classifier_benchmark.py tests/test_fast_classifier.py tests/test_orchestration.py tests/test_agent.py -q`
completed with **85 passed in 26.42s**. Tests cover v1/v2 preservation, v3 traceability and group integrity,
train/test exclusion, one inference, no Qwen, capability restriction, malformed
outputs, fallback behavior, metric calculations, and tool/network isolation.

## 19. Failures and limitations

The model failed native/romanized calculator actions, noisy STT, a resolved
follow-up, one Telugu conversation, and one smart-home escalation. Capability
selection remains weak. Dev/test divergence is large. The test has only one
clarification and two escalation positives. Labels lack independent human and
native-speaker review. Intent remains composed rather than directly learned.

## 20. Integration recommendation

**NOT READY FOR PRODUCTION INTEGRATION.** Path, capability, intent, request-mode,
clarification, escalation, multilingual-denominator, and latency gates fail.
Only schema validity and fallback-rate gates pass. Production routing remains
unchanged.

## 21. Exact next step

Before testing Laya, obtain independent human/native-speaker adjudication of v3,
especially the sealed records and romanized/STT cases. Then create a new grouped
holdout with at least 20 examples per language and multiple examples per path,
freeze it, and rerun this BERT artifact once. That reviewed holdout should become
the apples-to-apples Laya/Qwen/BERT evaluation set.

