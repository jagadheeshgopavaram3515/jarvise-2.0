# Autonomous frozen multilingual BERT + Logistic Regression experiment

**90% provisional target NOT REACHED.** Baseline and best experimental candidate each achieved **8/40 = 20.0% exact full-record provisional-label agreement**. Training actually ran on **232 synthetic_unverified records**; validation was fixed at **40 records in seven groups**. Four controlled experiments completed, then stopped after three consecutive experiments without improving the primary metric, as explicitly required. No verified accuracy was measured: independently verified gold denominator is **0**.

## Baseline, inputs and authorization

The candidate directory `reports/multilingual_dataset_build_20261010_155129` and all its recorded output hashes matched before use. The baseline joblib and JSON metadata exist and were fingerprinted. Source inputs, candidate files, prior user-modified tracked files and local encoder artifacts remained unchanged after the run. The frozen baseline was never overwritten or installed anywhere. No production adapter/routing file or historical annotation was edited. This run contains only new models, analysis and separately versioned provisional datasets.

The user's experiment-specific authorization allows structurally valid synthetic_unverified fitting, distinct from verified gold readiness. Of 250 candidate TRAIN rows, **232** passed and were used; **18 pending-reconciliation rows** were excluded, with reasons in exclusions.json. Every one of the **40** candidate VALIDATION rows stayed in the denominator: 39 synthetic_unverified and one pending_review. No difficult validation example was moved into training or silently excluded. Original utterances/targets/context/capabilities/status/provenance are preserved; no labels were changed because of predictions. Ambiguous/rejected/protected rows did not enter training. There is no new sealed-test result.

Validation has zero original baseline training-ID membership, zero train/validation ID overlap, zero normalized text overlap, zero declared connected-group overlap and zero protected original-test IDs. Scenario/translation/template groups from the prior build are preserved. These mechanical checks are PASS; completeness of semantic grouping remains NOT_VERIFIED. Additional frozen-vector neighbor triage is saved in semantic_overlap_triage.csv, not treated as confirmed duplicate or gold label evidence. Results must not be described as independently verified generalization or real-world accuracy.

## Local encoder and feature parity

Local encoder and tokenizer were available at `C:\Users\Jagadheesh\.cache\huggingface\hub\models--nlptown--bert-base-multilingual-uncased-sentiment\snapshots\8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d`. The checkpoint/model/tokenizer files were hashed before/after, including the 669 MB model.safetensors. No downloads or external APIs were used. Hugging Face offline flags plus a network/write audit guard were enabled. Encoder is AutoModel for nlptown/bert-base-multilingual-uncased-sentiment; sentiment classifier ignored; eval/inference_mode, every parameter requires_grad=False; no encoder optimizer/backward/fine-tuning.

Actual offline extraction produced 288 unique serialized texts with finite 768-dimensional request/capability embeddings and finite 3072-dimensional matching features. Attention-mask mean pooling, L2 normalization, max_length=160 and independent descriptor encoding match the existing adapter. No serialized text exceeded 160 tokens. Full per-example descriptors are preserved: cache keys use complete serialized text, not just name. Gold targets, IDs, reviewer evidence and statuses never enter the model feature text. Corrected pair-feature construction exactly equals inference construction. One actual encoder-forward adapter prediction matched cached-feature replay; all baseline/experimental decisions otherwise replay those frozen vectors through the **unmodified real adapter's postprocessing**. Frozen feature-cache/input hashes make reuse explicit and versioned.

## Search and stopping

Predeclared search_plan.json contains at most 20 configurations for C, weighting, solver/penalty and seed/convergence options. The actual completed sequence used balanced L2/lbfgs, seed 42, max_iter=3000 and C=[4,1,16,0.25]. Exact agreements were **[8,7,8,6]/40**. Experiment 01 established the best experimental result; experiments 02–04 did not improve it, triggering plateau stopping. Unrun solver/weight/seed configurations are disclosed in the plan; they were not claimed tested. No validation-selected relabel, metric relaxation, threshold alteration or dataset swapping was applied.

Each experiment fitted all 11 Logistic Regression heads on TRAIN features only and saved a unique compressed joblib, configuration, classes, convergence diagnostics, predictions and metrics. At least two fitted classes per head were checked. Classes are compared through model.classes_, including the positive index in the actual capability/boolean inference code. The matcher is binary 0/1, not a per-tool classifier. Missing names/reference kinds retain their original older-contract class spaces. Training-only class inventory and zero-support validation classes appear in class_support.csv; reference kind **project** is absent from fitted training targets and remains counted as a validation error opportunity.

The best new artifact **ties the baseline on the primary metric**; it is not an established improvement and is not a deployment recommendation. Its mode/reference field agreements are slightly below baseline. Raw head/path improvements are not substituted for full-record agreement.

## Actual metrics and denominators

Primary exactness covers all eight supervised ClassifierDecision target fields: intent, execution_path, request_modes, selected_capability, context_references, missing_information, needs_clarification and needs_escalation. Reference spans/values and missing descriptions are exact; modes are compared as sets. Confidence/evidence are not supplied supervised targets and are not included. Missing/failed predictions count incorrect; no critical target/class was dropped. Baseline and experimental schema prediction failures are 0 and 0.

| Output field | Baseline | Best experimental |
|---|---:|---:|
| intent | 19/40 | 19/40 |
| execution_path | 31/40 | 31/40 |
| request_modes | 35/40 | 34/40 |
| selected_capability | 20/40 | 20/40 |
| context_references | 27/40 | 26/40 |
| missing_information | 33/40 | 33/40 |
| needs_clarification | 37/40 | 37/40 |
| needs_escalation | 38/40 | 38/40 |

| Head (raw predictions) | Agreement | Provisional rate | Macro F1 |
|---|---:|---:|---:|
| execution_path | 33/40 | 82.5% | 0.597 |
| needs_clarification | 38/40 | 95.0% | 0.886 |
| needs_escalation | 38/40 | 95.0% | 0.487 |
| missing_information | 35/40 | 87.5% | 0.496 |
| reference_kind | 35/40 | 87.5% | 0.353 |
| reference_resolved | 40/40 | 100.0% | 1.000 |
| request_modes:conversational | 38/40 | 95.0% | 0.886 |
| request_modes:memory | 40/40 | 100.0% | 1.000 |
| request_modes:research | 36/40 | 90.0% | 0.804 |
| request_modes:action | 39/40 | 97.5% | 0.969 |
| capability_matcher | 36/77 | 46.8% | 0.386 |

Per-class precision, recall, F1, support and confusion matrices (including zero support) are in per_class_metrics.json and confusion_matrices/. Raw path head is **33/40**, whereas actual output path is **31/40** after clarification/escalation/capability postprocessing. Capability matcher has **36/77** binary pair agreement; selected capability output is **20/40**. Neither binary agreement nor a narrow successful field constitutes full-decision accuracy. Exact primary remains 8/40.

## Multilingual results

| Language | Strict exact agreement | Validation scenarios | Evidence |
|---|---:|---:|---|
| en | 2/12 | 7 | insufficient independently reviewed evidence |
| hi | 1/1 | 1 | insufficient independently reviewed evidence |
| hi-Latn | 0/7 | 3 | insufficient independently reviewed evidence |
| mixed-en-hi | 1/2 | 2 | insufficient independently reviewed evidence |
| mixed-en-ta | 1/1 | 1 | insufficient independently reviewed evidence |
| mixed-en-te | 1/1 | 1 | insufficient independently reviewed evidence |
| ta | 1/1 | 1 | insufficient independently reviewed evidence |
| ta-Latn | 0/7 | 3 | insufficient independently reviewed evidence |
| te | 1/1 | 1 | insufficient independently reviewed evidence |
| te-Latn | 0/7 | 3 | insufficient independently reviewed evidence |

All ten language tags present in training are disclosed. Each script/language group has its actual denominator, field agreements, per-head class supports and training support. Per-script results are in per_script_metrics.json. Native Telugu/Hindi/Tamil each have only **one validation example**; 1/1 cannot establish language support. Latin transliteration groups have seven each, code-switching groups one/two, English twelve. Every group lacks independently reviewed gold; even favorable percentages remain insufficient evidence. Translations/template variants are not independent semantic examples.

## Why the primary metric remains low

**11/40 records have objectively incompatible deterministic outputs under the unchanged adapter**, establishing a conservative maximum strict exactness of **72.5%**, even if learned heads were perfect. Annotated reference spans differ from the adapter's complete-utterance output; explicitly annotated resolved values differ from arbitrary first-context-scalar traversal; missing-information descriptions differ from the fixed output description. This is an implementation/target alignment problem, not proof those labels are wrong. Targets were not rewritten to inflate agreement. Structural upper-bound analysis is diagnostic only, never a model score or fitting input.

Other remaining errors include poor capability compatibility/ranking across scenarios, original limited positive domain coverage, rare references (project missing from train), synthetic template dependence, path/clarification/escalation interactions, and unverified multilingual/semantic labels. Error slices are saved by language, script, capability, intent, missing information, context and path. All 32 nonexact validation records are in proposed_label_review.csv; predicted disagreement is explicitly not proof of annotation error. Corrected descriptor handling did not improve the primary metric in this fixed experiment. Strong semantic overlap or annotation inconsistency requires human confirmation; no automatic correction was inferred from similarity.

The existing production adapter was not modified. Proposed runtime/input/reference policy work would need independent semantics and source evidence, not an oracle that injects gold annotations into features. The prior dataset build's isolation and objective conversion/mapping repairs were retained. Further same-intervention fitting stopped at the mandated plateau; safe independent error/coverage/hash analysis continued.

## Artifacts, tests and reproducibility

Selected experimental model: `D:\JARVIS\reports\autonomous_bert_training_20261010_161410\best_candidate\compact_heads.joblib`

SHA-256: `517b4008423b65ed3361946cfdbf8c24351289b7b1c392528ce1a853c52b025a`

Baseline model SHA-256: `7f4dd399b7238c7bf0c166454c3ff5aa09d6e3bf1afb82666b526d43cf452f05`

Training dataset SHA-256: `67d57286b528b62abb04355ecb0933c3906f5aa59805d8ff68df424b1afd4c6d`

Validation dataset SHA-256: `6f8cf0d9a8b1b7aab65d2fc2918bf26d327af00b4c76bf28f94aaad936a36692`

Software versions, complete encoder/input/output fingerprints, seeds, CPU thread settings, exact training/validation IDs and elapsed time are recorded in training_manifest.json and reproducibility_manifest.json. Every candidate and the selected copy are newly versioned under this run. All recorded source/model/cache hashes were rechecked; originals unchanged. Git status before/after preserves pre-existing user changes. The manifests exclude their own recursive hash by design.

Focused offline regression/integrity suite: **46 tests passed**, zero failures/errors (13 actual experiment-output checks and 33 previously constructed pure builder/gold/descriptor/reference checks). Test fixtures never became real labels. No production routing was connected and no original artifact replaced.

## Final answers and next boundary

1. Local encoder available: **yes**, actual offline features extracted with fingerprinted snapshot.
2. Training ran: **yes**, four complete experimental LR fits with frozen BERT.
3. Eligible fitting/validation: **232/40**, all provisional targets, validation fixed in seven groups; 18 pending training exclusions.
4. Baseline/best strict provisional agreement: **8/40 (20%)/8/40 (20%)**; output path **31/40 (77.5%)** for both.
5. 90% target reached: **no**; stopped on three consecutive primary plateaus after four experiments.
6. Per-head/class/language results: tables and JSON/confusion artifacts above; no verified accuracy claim.
7. Remaining defects/uncertainty: deterministic output alignment, capability/domain coverage, absent project reference class, incomplete semantic grouping and unverified annotations/native-language meanings.
8. Artifact paths/hashes: exact selected model and dataset hashes above; complete inventory in reproducibility_manifest.json and FILES_CREATED.md.
9. Existing source/production changes: **none**; only files under this fresh run directory were created.

Do not deploy or replace the baseline with this tied candidate. Address runtime reference/span/description semantics and obtain stronger independently reviewed scenario/domain coverage before expecting a valid 90% full-decision experiment. No data, reviewer decisions, targets or sealed labels were changed to manufacture a pass.
