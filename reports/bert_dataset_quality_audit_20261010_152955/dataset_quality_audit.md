# JARVIS frozen BERT dataset forensic audit

Final readiness: **BLOCKED_DATA_QUALITY**. The current heads have verifiable dataset provenance, but the original labels are synthetic and lack independent full-contract verification. The dataset does not substantiate a meaningful >=90% generalization claim. No model was trained, no inference/network/provider call was made, and no existing project file was intentionally written. Only this report directory was created.

| Category | Status | Basis |
|---|---|---|
| Exact dataset/head provenance | PASS | v3 SHA-256 and all 216 training IDs match loaded joblib and JSON metadata |
| Encoder revision/historical source snapshot | NOT_VERIFIED | Model ID recorded; resolved encoder revision, weight hash and source-code-at-training hash absent |
| Label/schema quality | FAIL | No independently verified original training labels; later taxonomy differs; missing trained classes |
| Duplicate and split integrity | FAIL | Confirmed v4 train/test text duplication; historical re-split preserves that duplicate across train/development |
| Class balance and independent coverage | FAIL | Only 22 train scenario groups; sparse capabilities; missing-information and reference classes absent |
| Multilingual quality | WARNING | Native and Latin examples exist; no adequate independent native-speaker verification |
| Context/input fidelity | FAIL | Consolidation discards actual input context/capability descriptors; reference resolver chooses unrelated first scalar |
| Features/preprocessing | FAIL | Name-keyed descriptor overwrite creates training/inference differences |
| Human review traceability | WARNING | 31 ledger entries conflict with stale zero-review summary; 18 accept/correct entries use newer contract |
| Sealed-test label distribution | NOT_VERIFIED | Labels and outcomes excluded from analysis; only authorized IDs/text/group/split/hash metadata used |
| Read-only source integrity | PASS | All hashed inputs unchanged after analysis; no existing validation runners with write side effects executed |

## Verified source-to-head flow

v1 seed has 28 rows. `prepare_v2.py` produces 228 rows (160 train, 50 dev, 18 test) through scenario synthesis and prior-label adjudication. `prepare_v3.py` derives those rows, changes metadata/limited historical annotations, and adds 84 manually authored synthetic examples: 56 train and 28 dev. V3 has **312 rows: 216 TRAIN, 78 DEV, 18 sealed TEST**. DEV is the validation/model-selection split; no distinct validation split exists.

`train_candidate.py:train()` invokes `load_dataset(dataset_path, split="train")`. The loader parses every line before filtering, including test records, but fitting only consumes the selected train rows. Thus the training loader has unnecessarily broad sealed-label visibility; ID/hash evidence does not show test targets were fitted. No status gate excludes pending/synthetic examples. All 216 training rows are explicitly synthetic candidates pending human review.

Actual artifact: `benchmarks/fast_classifier/models/v3/compact_heads.joblib`; adapter `compact-mbert-frozen-v3`; source `benchmarks/fast_classifier/v3/dataset.jsonl`; SHA-256 `94856b101b71c08655a6407b3e313938cef742beacef2b60ea4a1f20741b5a24`. The dataset hash, IDs, split membership, JSON metadata and loaded artifact agree. All 216 IDs are listed in `source_provenance.json`. Runtime fitting history and encoder revision cannot be independently reconstructed from IDs alone.

| Head | Target source | Feature shape |
|---|---|---|
| execution_path | expected.execution_path | 768 request features |
| 4 request_modes | membership in expected.request_modes | 768 request features each |
| clarification / escalation | respective expected boolean | 768 request features |
| missing_information | first expected item name, else __none__ | 768 request features |
| reference_kind | first expected reference kind, else __none__ | 768 request features |
| reference_resolved | first reference resolved_value is not null | 768 request features |
| capability_matcher | descriptor name == expected.selected_capability | 3072 pair features |

Features use `nlptown/bert-base-multilingual-uncased-sentiment` via AutoModel, ignoring the sentiment classification head. Tokenizer is local-files-only; max_length=160, padding/truncation enabled; attention-mask mean pooling followed by L2 normalization. Forward calls use eval/inference_mode. Only LogisticRegression `.fit()` optimizes parameters: balanced class weights, C=4, seed=42, lbfgs, max_iter=2000. No encoder fine-tuning, transliteration or dataset text normalization appears in this training path.

Actual encoding is **independent**, despite the current trainer's default of joint_pair. Request input is `User request: <utterance>\nContext: <sorted JSON truncated to 2000 characters>`. Capabilities are encoded separately. Pair vector concatenates request, capability, absolute difference and elementwise product. No gold fields or IDs are serialized. No saved embedding cache exists in this trainer, so stale persistent-cache checking is not applicable; training arrays are ephemeral. Capability descriptors form a name-keyed in-memory cache, creating the discrepancy below. Encoder token truncation effects were not measured because no encoder/tokenizer was instantiated.

Intent, reference span/value, missing-information description, evidence and confidence are partly deterministic outputs, not independently learned heads. The older ClassifierDecision only enums request_modes; intent/path/reference kind/missing name are nonempty strings, not closed enums. The newer FastBrainDecision family/path enums must not be substituted into this contract. In particular `qwen_agent`, `respond` and `fast_path` in later reviews are not the compact model's seven semantic path classes. No exact intent-composition mismatch was found in nonsealed v3 records; the initial concern was resolved by checking actual targets.

Production source has no import/call of CompactMultilingualAdapter outside its own module; existing instantiations are offline evaluation runners. Its actual input contract is DecisionRequest(user_text, context mapping, Capability tuples). This is source wiring evidence, not proof about a running/custom-injected service. The IDE's Qwen/LoRA evaluation plan describes a separate end-to-end task; no Qwen/LoRA evaluation was performed here.

## The 359-record benchmark and human reviews

`prepare_v4.py` retains 216 train and 78 dev rows and supplies 140 test rows, for 434 total. `build_fast_brain_dataset.py` imports v4 into **359 historical_synthetic candidates**, quarantining 75 records. It then re-splits by declared group into **239 train, 57 development, 63 sealed_candidate**. This queue is a mixture of original train/dev/test lineage, not the exact data fitted by compact v3. The older v3 test IDs are not the newly authored v4 test set. Review queue and adjudication artifacts are workflow artifacts, not additional independent examples.

Review ledger: **15 accept, 3 correct, 13 skip**, 31 distinct records, one recorded reviewer identity. **328 records have no review decision**; 13 skips remain unresolved. All 359 source records still say pending_review/historical_synthetic, while the ledger separately holds 18 accept/correct decisions. That is preserved history, not grounds for overwriting either file. The saved review_analysis.json reports zero reviewed records and is stale. There are no recorded reject/ambiguous decisions in this ledger; zero is not evidence that ambiguity is absent. The 75 builder rejections are import/schema quarantine, not human rejections.

Of the recorded later accept/correct decisions, 12 lie in the nonsealed analysis scope; 11 refer to original training IDs. The other six accept/correct decisions are protected by candidate/original-test lineage and their labels were not inspected. **Zero original full ClassifierDecision training labels are established as independently verified.** Eighteen later recorded human decisions are not eighteen independently adjudicated original labels. Two-reviewer/native-speaker correctness, naturalness and translation preservation remain NOT_VERIFIED.

## Concrete high-severity findings

1. **Clarification records vanish during consolidation.** `normalize()` casts each structured missing-information object with `str(x)` instead of extracting its name; `FastBrainRecord.validate()` rejects these strings against the actual MissingInformation enum. All 75 quarantine entries report unknown missing-information labels; accepted queue metadata shows no clarification-positive examples. Example nonsealed source IDs: `synth_ambiguous_open-en-01`, `train_clarify_missing_target-en`, `train_clarify_search-en`. The 359 queue is therefore biased by import failure, not evidence that clarification is unnecessary. No sealed label was used to establish this mechanism.
2. **Capability-family builder omissions.** Real Registry actions such as open_folder, recent_files, grep_code and run_project_tests are not handled explicitly by `capability()`. Categories can incidentally rescue some examples; 25 nonsealed records with these actions actually map to none. `train_open_downloads-en` is an inspectable example. These are bugs in the later queue builder, not proof of bad original compact labels.
3. **Training/inference descriptor discrepancy.** Five capability names have multiple descriptor strings in training. The independent trainer stores the last descriptor for each name, whereas inference embeds each request's actual descriptor. **120 of 216 training requests** contain a differing descriptor; **340 training capability pairs** differ. Counts by name: `{'open_app': 80, 'find_file': 100, 'google_search': 80, 'open_folder': 40, 'read_page': 40}`. The CSV flags all occurrences of variant names as potential overwrites; this measured subset establishes the actual mismatch.
4. **Unlearnable label classes.** Training missing_information classes are only __none__/target. DEV includes prior_request (20) and query (1), absent from the fitted head. Training reference_kind is only __none__/item/prior_conversation; DEV also includes prior_request (20), folder, project, search_result and topic (one each). Those predictions cannot be emitted by the fitted heads. The older contract accepts these strings; this is coverage failure, not invalid enum usage.
5. **Deterministic reference mismatch.** 31 nonsealed v3 records have annotated resolved_value differing from `_first_context_value(context)`. Training `synth_recall_trip-en-01` annotates the decision but traversal chooses the topic name. There are 27 fixed reference-span mismatches and 54 fixed description mismatches. Correct gold labels can be incompatible with implementation; these are not model-disagreement-based relabel claims.
6. **Registry descriptor mismatches in the later builder.** 0 nonsealed v3 descriptor occurrences disagree with current registered required arguments. Static inspection of v4 CAPS finds 4 builder mismatches: `[{'capability': 'weather', 'builder_required': ['city'], 'registry_required': []}, {'capability': 'search_files', 'builder_required': ['query'], 'registry_required': ['keyword']}, {'capability': 'list_directory', 'builder_required': ['path'], 'registry_required': []}, {'capability': 'run_project_tests', 'builder_required': ['project_path'], 'registry_required': []}]`. Registry `search_files` requires keyword, not query; list_directory/run_project_tests have no required parameters, unlike the v4 builder. This conclusion uses source configuration, not sealed labels. Registry-at-training provenance is not recorded. See integrity_checks.json.
7. **Context lost and metadata oversimplified.** All consolidated records lack the original context and capability arrays; context_resolvable is merely bool(original context), which does not prove resolution. Original IDs allow reconstruction from v4, but the standalone queue does not reproduce inference inputs. 10 nonsealed later reviews specify active_task while original context is empty; this is human-review triage, not a rewritten decision. 43 nonsealed queue examples say mixed script although their text is Latin-only. clean/noise metadata is defaulted by the builder, not measured STT robustness.

## Duplicates, related examples and split contamination

No duplicate IDs were found in any source dataset. V3 has one exact/normalized duplicate pair, entirely in test; label comparison is NOT_VERIFIED. No exact/normalized TRAIN–DEV duplicate or declared-group crossing was found in v3. Training has no exact utterance duplicates. This does not prove semantic independence: source scenario templates, translations and small topic families are correlated.

V4 has one **confirmed exact TRAIN–TEST pair**: `train_open_downloads-en` (train) and `v4_open_downloads-en` (test), both "Open my Downloads folder." One contaminated test text is **1/140 = 0.71%** of test; 2/434 = 0.46% of dataset rows belong to that pair. Test target agreement was not checked. The later queue carries this pair into `hist-2ab4a4ab05600583` (train) and `hist-35138bdc0c6c8e35` (development): 2/359 = 0.56% of queue rows and 1/57 = 1.75% of development. Different declared group IDs conceal the duplication from group-only validation.

Same-script character triage finds **7 uncertain cross-split pairs in v4** (5 TRAIN–TEST, 2 DEV–TEST), **5 in the historical queue**, and zero in v3 at similarity >=0.85. Examples include Python-docs URL variants, close-Notepad transliterations, Downloads translations and repeated unresolved-open forms. They are candidates for semantic review, not confirmed label errors. Cross-language translation leakage and minor-substitution template families are NOT_VERIFIED by this character method. Declared groups are reported separately; different same-class actions are not automatically duplicates.

The current 359-record benchmark overlaps **188/359 = 52.37%** original training IDs, versus 171 nonmembers. Its overlap by current queue split is `{'train': 152, 'development': 8, 'sealed_candidate': 28}`. V3/V4 DEV and original sealed TEST IDs have zero training-ID membership; that did not prevent the exact v4 text contamination. Historical ID hashing changes record IDs, so matching uses original_id. No absence-of-ID claim is treated as leakage-free.

## Balance, coverage and language evidence

| Execution path | TRAIN | DEV (validation) |
|---|---:|---:|
| clarify | 28 (13.0%) | 26 (33.3%) |
| direct_response | 24 (11.1%) | 4 (5.1%) |
| escalate | 36 (16.7%) | 5 (6.4%) |
| memory_recall | 24 (11.1%) | 7 (9.0%) |
| multi_step_task | 24 (11.1%) | 5 (6.4%) |
| research | 24 (11.1%) | 5 (6.4%) |
| single_action | 56 (25.9%) | 26 (33.3%) |

TRAIN has only **22 declared scenario groups**: eight original 20-row synthetic scenario families plus fourteen four-language targeted scenarios. DEV has 17 declared groups. The 216 rows therefore substantially overstate independent semantic breadth; even independence between differently named groups is not guaranteed. Action classes close_app/open_folder/open_url have only four positive TRAIN examples each, derived from one scenario; no training positive for many real Registry actions (including weather/location, project inspection/testing, read_file, recent_files and search_files). Capability matcher has binary matched/unmatched classes rather than a capability enum, but absence of positive actions still constrains its learned compatibility.

All request mode/binary/missing/reference/capability distributions and proportions are in label_distribution.csv. Zero-count classes are explicitly added from nonsealed labels and the actual Registry. __none__ is an explicit head sentinel, distinct from source selected_capability null and empty lists. No fake labels were added to sources. Balanced class weighting is reasonable for skewed binary heads but cannot create missing classes or independent examples; path imbalance is moderate, while DEV is dominated by clarify and single_action. Sampling would need group/language-aware human-reviewed data, not duplication.

| Recorded language | Observed script | Training rows |
|---|---|---:|
| en | latin | 54 |
| hi | devanagari | 11 |
| hi-Latn | latin | 31 |
| mixed-en-hi | latin | 12 |
| mixed-en-ta | latin | 12 |
| mixed-en-te | latin | 12 |
| ta | tamil | 11 |
| ta-Latn | latin | 31 |
| te | telugu | 11 |
| te-Latn | latin | 31 |

English, Telugu, Hindi, Tamil, Latin transliteration and language mixing are present. Their training rows are authored scenario variants, not independent proof of language support. Native-script coverage is especially concentrated in a few paths; no language/script has adequate independently verified original training-label coverage. DEV native Telugu/Hindi/Tamil counts are small; language-specific accuracy cannot support a robust 90% claim. Spelling, grammar, meaning-preserving translations and transliteration variants require native-speaker adjudication. No source was translated or normalized in place; original utterances remain in the review CSV, with detected script stored separately.

## Integrity, evidence limits and 90% readiness

Every loaded head is LogisticRegression with finite coefficients; actual class arrays and feature dimensions are recorded in integrity_checks.json. Probability lookups use model.classes_ and correctly handle binary positive class indices. No direct target fields are included in encoder input. Constructed context/capability sets can nevertheless act as synthetic scenario cues, and real inference may supply different sets. No live adapter integration or production-context fidelity was inferred from offline examples.

Historical DEV confirmation reports 61/78 execution paths (78.2%), 12/36 applicable capabilities (33.3%), 49/78 intents (62.8%) and 65/78 exact mode sets (83.3%). These are pre-existing diagnostics against unverified labels, not new accuracy results. Sealed outcomes were not used. Accuracy must specify the field or full-decision exactness and be measured on independent reviewed groups; a class-imbalanced aggregate alone is not the requested generalization evidence.

Schema parsing passed for all analyzed nonsealed examples and nonsealed reviewed labels. A schema pass is not semantic correctness: the older parser accepts any nonempty path/intent string and does not fully enforce path/uncertainty/capability invariants. The original trainer lacks a human-review gate and parses sealed rows before filtering. Encoder/tokenizer revision, actual token-loss distribution and source-code version at fitting remain NOT_VERIFIED. Early source preview incidentally displayed one sealed record and historical prose containing test outcomes; neither was used for label analysis or tuning. Automated reports suppress sealed labels, including historical copies; this audit cannot claim zero incidental sealed visibility.

Input hashes were rechecked without changes. Existing working-tree changes were preserved. No pytest, dataset builders, training/evaluation main functions, model downloads or reference-project access were performed. Automated parsing and head inspection are analysis only. Semantic-review CSV rows are flags with original evidence, not corrected labels; multiple rows per record and multiple dataset versions must not be added as independent bad-record counts.

## Explicit answers and priorities

1. Exact fitted data: 216 train rows in hash-matched v3 dataset, listed verbatim by ID in source_provenance.json; 160 synthesized v2 train rows plus 56 targeted v3 rows.
2. Independently verified original training labels: **0 established**. Later ledger records 18 accept/correct decisions under another contract, 12 nonsealed eligible and 11 later training-ID reviews; independent full-contract/native correctness remains NOT_VERIFIED.
3. Synthetic/pending/unreviewed: **216/216 original training rows** synthetic pending review; **359/359 consolidated rows** historical_synthetic/pending metadata. Of queue review decisions, 18 accept/correct, 13 skip, 328 absent, zero recorded ambiguous/rejected; 75 import rejections. Semantic ambiguity and actual contradictory-gold counts remain NOT_VERIFIED. The visible differing "Continue that." labels have different context and are not established contradictions.
4. Training overlap: **216** source training IDs and **188/359** later queue original IDs; overlap by split above. Every available source record receives an overlap CSV row.
5. Leakage: one confirmed v4 TRAIN–TEST duplicate pair, propagated into queue TRAIN–development; seven/five uncertain near-text cross-split pairs respectively. No v3 exact TRAIN–DEV overlap. Translation/template leakage beyond these checks is NOT_VERIFIED.
6. Unreliable/sparse labels: missing prior_request/query and five reference kinds absent from fitted heads; four-example capabilities; consolidation missing clarification entirely; none capability-family mappings; context/reference-derived labels and unreviewed synthetic gold.
7. Adequate verified language/script coverage: **none established**. Presence and counts are documented; naturalness and independent intent/action preservation are unverified.
8. Bugs: structured missing-information conversion, omitted capability mappings, name-keyed descriptor overwrite, fixed reference/description reconstruction and missing class coverage; current Registry argument discrepancies and weak schema/review gating. No evidence of direct gold-field feature leakage or encoder fine-tuning.
9. Meaningful 90% target: **not yet supported** by trusted independent coverage/evaluation. BLOCKED_DATA_QUALITY, not BLOCKED_PROVENANCE, because exact source hash/IDs are established. This does not prove 90% is impossible after approved data/contract work.
10. Five highest-priority fixes, requiring separate approval:
   1. Commission blind full-contract human/native-speaker review of original train/dev labels; reconcile newer review decisions explicitly without rewriting history.
   2. Correct and audit the consolidation missing-information and capability-family mappings; recover quarantined candidates through review.
   3. Align training/inference capability descriptors, Registry argument schemas and reference/missing-field output semantics.
   4. Rebuild splits from audited paraphrase/translation/template families; remove confirmed cross-split contamination and establish a fresh locked reviewed evaluation set.
   5. Add independent reviewed scenarios for missing/rare classes and native/Latin/code-switched coverage; freeze encoder/source fingerprints and report per-head, per-class and language metrics with uncertainty.

No fixes were applied. Await explicit approval before changing datasets, labels, code, models or reviewer history.
