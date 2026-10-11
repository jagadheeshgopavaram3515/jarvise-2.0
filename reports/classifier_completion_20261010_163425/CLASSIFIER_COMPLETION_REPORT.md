# CLASSIFIER COMPLETION REPORT

Executive verdict: **PARTIALLY COMPLETE**. The isolated implementation, baseline reproduction, eight controlled fits, real encoder evaluation, confidence/abstention analysis and relevant regression suite are complete. General resolution/serialization defects are fixed in experimental v2, but **not all eleven examples match exactly**: unverified rendering conventions, missing vocabulary and model errors remain. Production readiness and 90% independently verified accuracy are not established.

Strict full-record provisional agreement improved **8/40 (20%) → 10/40 (25.0%)** on the unchanged fixed validation. No ground-truth labels, scenarios or denominators were altered. Output path agreement improved **31/40 → 33/40**; capability remains **20/40**. Scores are agreement with 39 synthetic_unverified and one pending_review validation targets, **not verified accuracy**. Verified-gold denominator is zero.

## Evidence and exact baseline reproduction

Inspected canonical compact_adapter.py, decision.py, train_candidate.py, actual loaders/builders, immutable model heads and prior reports. Initial git status/diff-stat were recorded. No reference project was accessed. The dataset-build and earlier audit reports remain unchanged; the experiment starts from the prior 232-record authorized provisional TRAIN and fixed 40-record VALIDATION.

Both frozen original baseline and prior best reproduced **8/40**, with every predicted field reproduced under the actual frozenset request_modes semantics. The initial bytewise mapping comparison failed only because mode list serialization order differed across processes. This was investigated before training: canonical sorted-mode comparison shows no semantic or numeric field discrepancy. Metric definition did not change. Reproduction predictions, raw head reports, per-field metrics, confusion matrices, cached-runtime latency and failures are saved separately.

Local tokenizer/encoder snapshot and all feature-cache/checkpoint hashes match the prior verified assets. The encoder stayed frozen; only new LR coefficients were fitted. Actual local encoder execution on all 40 validation records yielded **10/40**, median **132.4 ms**, p95 **159.0 ms**, plus **643.4 ms** encoder startup. Cached-feature runtime is separately labeled in metrics and must not substitute for encoder latency. Actual versus cached eight-field predictions agree for 40/40 records; maximum confidence numeric delta is 6.36e-07. Confidence bit-level drift is reported, not mistaken for target prediction drift.

## Root causes and implemented fixes

- `CompactMultilingualAdapter._predict_references` ignores supervised reference text and calls `_first_context_value`. It can return a topic or alias instead of the relevant decision. `experimental_adapter.py:typed_resolve` uses typed recent_topics/projects/explicit-reference records, prefers explicitly named entities, returns a unique decision/name and leaves ambiguous or malformed values unresolved. It never reads a target or arbitrary first scalar. A unit test exposed substring matching of one-letter names; whole-token matching fixed that genuine general bug.
- The original adapter always emits the full utterance as reference text and a fixed missing-information description; the original trainer fits only kind/name. Experimental v2 adds **training-only LR rendering heads** for reference_span and missing_description. Internal __utterance__/__none__ sentinels encode source identity/no-reference cases; they are not new output enums. Missing names remain older-contract nonempty strings. Unresolved span/name head conflicts are explicitly safe-fallbacked rather than silently invented. The expected source text/description is never injected into input context or changed to match the model.
- `project_reference` recognizes a uniquely explicitly mentioned project alias and extracts its real source span; typed_resolve returns the named project. This supports the legitimate untrained project domain through inference context, not fabricated training labels or a special hologram exception. Regression uses an unseen space-elevator alias, not a copied validation rule.
- `train_candidate.py` originally keyed descriptors by name. New isolated training preserves every per-example complete descriptor. Eight experiments compare full 3072-dimensional pair features versus 1536-dimensional abs-difference/product interaction features, C/weighting and capability thresholds. Scores remain poor for capabilities; rank improvements were not claimed where absent.
- The older parser validates shape but does not enforce all cross-field meanings. Experimental adapter validates final ClassifierDecision and explicitly returns existing escalation-shaped data with abstained/reason metadata for unsupported head outputs. No production caller/router was connected or changed. Experimental confidence is the minimum relevant path/reference/span/description/capability model support, **not calibrated correctness probability**.

The rendering conventions themselves are inconsistent: some source references are whole utterances, some literal phrases, some abstract English "prior discussion" or "it" for non-English requests. Both are schema-valid, but their intended semantic contract is unverified. No A (incorrect gold/source semantic label) was confirmed just because predictions disagreed. D/E denotes underspecified annotation/rendering convention, not invalid enum/schema. Remaining B errors are disclosed.

## Eleven cases: individual classification and outcomes

| Original ID | Diagnosis | Reference exact after | Missing exact after | Full exact after |
|---|---|---|---|---|
| dev_memory_project-te-Latn | C: deterministic field supervision/resolution defect; D/E: exact rendering policy underspecified; B: remaining reference/span prediction error | False | True | False |
| dev_memory_project-hi-Latn | C: deterministic field supervision/resolution defect; D/E: exact rendering policy underspecified; B: remaining reference/span prediction error | False | True | False |
| en-memory-001 | C: deterministic field supervision/resolution defect; D/E: exact rendering policy underspecified; B: remaining reference/span prediction error; E: exact desired phrase absent from train vocabulary | False | True | False |
| dev_clarify_unresolved-hi-Latn | C: deterministic field supervision/resolution defect; D/E: exact rendering policy underspecified; B: remaining reference/span prediction error; B: missing-name/description prediction error | False | False | False |
| dev_clarify_unresolved-ta-Latn | C: deterministic field supervision/resolution defect; D/E: exact rendering policy underspecified | True | True | True |
| dev_memory_project-ta-Latn | C: deterministic field supervision/resolution defect; D/E: exact rendering policy underspecified; B: remaining reference/span prediction error | False | True | False |
| ctx-hologram-001 | C: deterministic field supervision/resolution defect; D/E: exact rendering policy underspecified | True | True | False |
| dev_memory_project-en | C: deterministic field supervision/resolution defect; D/E: exact rendering policy underspecified; B: remaining reference/span prediction error | False | True | False |
| dev_clarify_unresolved-en | C: deterministic field supervision/resolution defect; D/E: exact rendering policy underspecified; B: remaining reference/span prediction error; B: missing-name/description prediction error | False | False | False |
| hi-codeswitch-memory-001 | C: deterministic field supervision/resolution defect; D/E: exact rendering policy underspecified; B: remaining reference/span prediction error; E: exact desired phrase absent from train vocabulary | False | True | False |
| dev_clarify_unresolved-te-Latn | C: deterministic field supervision/resolution defect; D/E: exact rendering policy underspecified | True | True | True |

Complete per-record source annotations, baseline outputs, new outputs and reasons are in eleven_case_evidence.json; no expected label changed. Typed context-resolution tests cover **all eleven** using their actual supplied input/context. These tests verify resolver semantics independently of span-head score. Two unseen literal span phrases cannot be produced by the trained rendering vocabulary; no validation phrase was added to classes. Memory-project abstract-span predictions remain weak because training uses a different context/template distribution. Structured fields repaired in a case do not imply its other path/mode fields are correct.

## Dataset, label and feature gates

TRAIN **232 synthetic_unverified**, VALIDATION **40** (seven groups), unchanged from prior run. Eighteen pending training records remain excluded. Every record passes actual compact schema. Protected original-test IDs, duplicate/normalized-equivalent request texts and connected scenario groups do not cross the fitting/selection boundary. Validation has no baseline training-ID overlap. Broader semantic independence remains NOT_VERIFIED; prior semantic neighbor triage is not discarded or called proof of no leakage. No sealed label/outcome was read for construction or tuning; no old benchmark becomes an independent test.

Class distributions, schema/integrity counts, missing classes, multilingual tags and exclusions are machine-readable in data_quality_report.json/class_distributions.csv. Reference kind project remains absent from trained kind classes; the explicit contextual alias rule is separately tested. Missing-information conversion retains documented name/description fields; prior_request is valid in the older open-string contract and is not silently forced into a Fast Brain enum. The prior corrected family mapping covers all Registry actions including open_folder/recent_files/grep_code/run_project_tests; tests verify these omissions. Registry argument repairs remain in derived candidate copies, with original descriptors preserved.

Features contain only original user text, supplied context and advertised capabilities. Labels, reviewer history, IDs, splits and newly learned rendering targets are not in feature serialization. New artifacts require ExperimentalCompactAdapter: an interaction matcher expects 1536 features and is **not compatible with direct use by the unchanged 3072-feature production compact adapter**. The caller contract remains ClassifierDecision, but the artifact/adapter version is explicit. No production patch was applied.

## Before/after metrics: full output versus raw heads

| Target | Frozen baseline | Previous candidate | Experimental v2 |
|---|---:|---:|---:|
| intent | 19/40 | 19/40 | 19/40 |
| execution_path | 31/40 | 31/40 | 33/40 |
| request_modes | 35/40 | 34/40 | 36/40 |
| selected_capability | 20/40 | 20/40 | 20/40 |
| context_references | 27/40 | 26/40 | 32/40 |
| missing_information | 33/40 | 33/40 | 38/40 |
| needs_clarification | 37/40 | 37/40 | 40/40 |
| needs_escalation | 38/40 | 38/40 | 36/40 |

| Raw head | Agreement | Macro F1 |
|---|---:|---:|
| execution_path | 34/40 | 0.610 |
| needs_clarification | 38/40 | 0.886 |
| needs_escalation | 38/40 | 0.487 |
| missing_information | 37/40 | 0.546 |
| reference_kind | 36/40 | 0.365 |
| reference_resolved | 40/40 | 1.000 |
| mode:conversational | 38/40 | 0.886 |
| mode:memory | 40/40 | 1.000 |
| mode:research | 38/40 | 0.886 |
| mode:action | 39/40 | 0.969 |
| capability_matcher | 36/77 | 0.386 |
| reference_span | 2/11 | 0.074 |
| missing_description | 2/4 | 0.333 |

Raw heads and postprocessed output are distinct. The main full-record metric retains **all eight supervised target fields**, including exact reference span/value and missing-information description, with mode sets canonicalized. Confidence/evidence have no supervised source target and are excluded consistently across baseline and candidates. All 40 records remain scored, including fallback. Per-class precision/recall/F1/support and confusion matrices include unseen validation classes; no class was removed to manufacture success.

Eight unique coefficient artifacts were preserved. Best selection was strict exact first, then output path/capability as disclosed tie-break. C=16, balanced L2/lbfgs, seed=42, interaction pairing and matching threshold=.25 selected `experiment_06`. Exact counts across the bounded matrix are [10, 10, 4, 10, 10, 10, 6, 7]/40. Four configurations tie at the best exact count; changing pair representation does not demonstrate capability improvement. Remaining capability ranking and annotation/contract problems dominate; more repeated tuning on these 40 unverified correlated samples would not establish 90% accuracy.

## Multilingual and abstention evidence

| Language | Exact provisional agreement | Scenario groups |
|---|---:|---:|
| en | 2/12 | 7 |
| hi | 1/1 | 1 |
| hi-Latn | 0/7 | 3 |
| mixed-en-hi | 1/2 | 2 |
| mixed-en-ta | 1/1 | 1 |
| mixed-en-te | 1/1 | 1 |
| ta | 1/1 | 1 |
| ta-Latn | 1/7 | 3 |
| te | 1/1 | 1 |
| te-Latn | 1/7 | 3 |

Native Hindi/Tamil/Telugu each have one validation example; English twelve, Latin transliteration seven each, code-switching one/two. These are insufficient independently reviewed samples. Per-script metrics and class support are saved; no script or language is declared production-supported from favorable small counts. Original multilingual utterances are preserved and no translations were fabricated.

| Confidence threshold | Accepted/40 | Exact among accepted | Full-set exact including fallback/40 |
|---|---:|---:|---:|
| 0 | 38/40 | 10/38 | 10/40 |
| 0.25 | 38/40 | 10/38 | 10/40 |
| 0.5 | 20/40 | 10/20 | 11/40 |
| 0.65 | 6/40 | 6/6 | 7/40 |
| 0.8 | 4/40 | 4/4 | 5/40 |
| 0.9 | 0/40 | 0/0 | 1/40 |

Accepted-set correctness is provisional only, and high rates at low coverage are **not** the 90% full-set target. A fallback can itself agree with a source unsupported-request target, so full-set exact is shown separately. Thresholds were diagnostics, not a production-calibrated policy. Main comparison remains no optional threshold, **10/40**. Low-support outputs use existing escalation-shaped contract data, expose abstention_reason, fabricate no capability/missing field/context value, and never execute fallback tools/providers.

## Tests, blockers and commands

Relevant suite included 12 existing classifier/benchmark/Fast Brain/annotation/shadow modules plus the new experimental tests. Final result: **153 passed, 0 failed, 33 skipped/blocked**. Protected original-dataset/historical-outcome reads are blocked and explicitly skipped; no claim is made that those tests passed. Existing production logging initialization is replaced by a no-file logger only inside the test harness; production files are unchanged. All test temp artifacts are inside this run. Pytest plugin auto-loading/cache writes and network calls are disabled. Individual results/reasons appear in test_results.json and pytest_results.xml.

Tests cover all eleven typed resolutions, unseen alias extraction, ambiguous/empty/malformed context, structured missing-information, all omitted mappings, per-pair structure, unknown positive classes, rendering vocabulary TRAIN-only, multilingual output validity, exact baseline preservation and reproduction, fixed validation, actual encoder and monotone abstention coverage. Initial regression failures (single-character substring resolution and Windows default UTF-8 decoding in a test) were corrected in this isolated implementation; final rerun passes. The broader suite remains limited by sealed-data protections, not silently omitted.

Exact executed commands:

```powershell
python -B reports/classifier_completion_20261010_163425/run_completion.py
python -B reports/classifier_completion_20261010_163425/finish_run.py
python -B reports/classifier_completion_20261010_163425/run_tests.py
python -B reports/classifier_completion_20261010_163425/finalize_completion.py
```

The first runner reproduced baselines and completed all eight fits, then a report-only record-key typo interrupted final analysis. finish_run resumes saved fits and **does not retrain/overwrite experiment artifacts**; it re-evaluates the tested final resolver/uncertainty policy into separate final_policy_metrics files, preserving initial metrics. Exact model artifacts were never rewritten after their fitting. General semantic logic is in experimental_adapter.py; no eleven-ID lookup exists in inference.

## Exact artifacts and production impact

Baseline: `D:\JARVIS\benchmarks\fast_classifier\models\v3\compact_heads.joblib`
SHA-256: `7f4dd399b7238c7bf0c166454c3ff5aa09d6e3bf1afb82666b526d43cf452f05`

Previous candidate: `D:\JARVIS\reports\autonomous_bert_training_20261010_161410\best_candidate\compact_heads.joblib`
SHA-256: `517b4008423b65ed3361946cfdbf8c24351289b7b1c392528ce1a853c52b025a`

Selected experimental model: `D:\JARVIS\reports\classifier_completion_20261010_163425\experiment_06\compact_heads.joblib`
SHA-256: `4a75e077e02573b36a911e468e513aff839ab37496883e64ac935122a28a458f`

Adapter: `D:\JARVIS\reports\classifier_completion_20261010_163425\experimental_adapter.py` (hash in completion_manifest.json).

TRAIN SHA-256: `67d57286b528b62abb04355ecb0933c3906f5aa59805d8ff68df424b1afd4c6d`
VALIDATION SHA-256: `6f8cf0d9a8b1b7aab65d2fc2918bf26d327af00b4c76bf28f94aaad936a36692`

All changed/created files are **new files under this experimental directory**; exact inventory/hash list is in FILES_CREATED.md/completion_manifest.json. Existing application code, production routing, data/annotations/review history, baseline/previous candidate and encoder cache are unchanged. All watched input hashes match; pre-existing git status entries are preserved: True. No download, deployment or reference-project access occurred.

## Recommendation and acceptance boundary

**Do not deploy this candidate.** Technical experiments and general fixes are implemented, trained and tested; full semantic completion remains partial. Strict provisional agreement is only 25.0%, capability selection 50.0%, several exact renderings are unsupported/inconsistent, and verified accuracy is unmeasured. Passing tests or gaining two synthetic-label matches does not establish readiness.

The **single highest-value next step** is to freeze an exact reviewed ClassifierDecision annotation/rendering policy and obtain independent, qualified human labels for diverse capability/reference scenarios under that policy. This resolves the present ambiguity about literal spans versus abstract references/descriptions and supplies real capability-domain evidence. Do not solve it by rewriting targets to mirror current predictions. The isolated implementation/model remains available for a later approved experiment, with production unchanged.
