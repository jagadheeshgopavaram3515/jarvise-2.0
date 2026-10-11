# Compact multilingual dataset build v1

Verdict: **BLOCKED_PENDING_HUMAN_REVIEW**. A structurally valid, versioned candidate dataset and corrected isolated build/feature implementations are complete. There are **zero verified gold records**, **zero eligible supervised fitting records**, and **zero new final evaluation records**. Do not train on the candidate files or treat their provisional splits as independent accuracy evidence.

All outputs are in this new timestamped directory. Existing datasets, Golden Review bundle, human ledger, audit report, frozen heads, production source and pre-existing user changes are unchanged. No retraining, inference, encoder loading, network/provider calls or production integration occurred. 84 input files were fingerprinted and rechecked.

## Exact counts and distinct states

| State/category | Count | Interpretation |
|---|---:|---|
| verified_gold | 0 | No approved exact ClassifierDecision references |
| verified_gold_needs_adjudication | 0 | No submitted exact-contract approvals to adjudicate; potential issues remain in review queue |
| pending_review | 19 | Historical decision/reconciliation evidence; not approved compact targets |
| ambiguous | 0 | No confirmed human ambiguous judgment; this does not establish absence of ambiguity |
| rejected/schema failures | 0 | All retained nonsealed candidates pass real parser; old builder rejections were recoverable structure errors |
| synthetic_unverified | 271 | Authored/synthetic candidates with unverified exact targets |
| canonical nonsealed candidates | 290 | Stable original-ID records; multiple revisions retained as provenance, not examples |
| candidate scenario groups | 34 | Conservative connected components; independent semantic status unverified |
| original versions collapsed | 502 | Historical source revisions are not counted as independent examples |
| provisional candidate TRAIN | 250 | 27 groups; contains unverified labels only |
| provisional candidate VALIDATION | 40 | 7 groups; no original model training IDs |
| new final TEST | 0 | Not collected, not reviewed and not locked |
| quarantine entries | 75 | All accounted for |
| recovered nonsealed candidates | 52 | Retained within the 290, never counted again as new gold |
| protected quarantine entries | 23 | Metadata-only; sealed target payloads not copied or used |
| protected original-test IDs | 162 | Across v1/v2/v3/v4; excluded from candidate fitting pools |
| native-speaker review queue | 214 | Inputs only; identities/expertise still unconfirmed |

The pool has 290 rather than all 294 v4 train/dev records because **four records have original v1 sealed-test lineage**, even though later versions moved them to development. Those historical test IDs remain protected. In all, 452 distinct historical source IDs resolve to 290 nonsealed candidates plus 162 protected IDs. All 75 quarantined records resolve to 52 recoverable nonsealed candidates plus 23 protected references. No sealed targets were exposed in build reports or copied into candidate records.

## Provenance and Golden Review reconciliation

Inventory covers v1/v2/v3/v4 compact sources, consolidation dataset/rejections/ledger, original builders and schemas, model metadata/joblib hash, audit evidence and complete Golden Review file hashes. `source_provenance.json` records source hashes, rows/IDs, split/status/provenance metadata and the working-tree modifications recorded before the build. `git_status_before.txt` preserves pre-existing user state.

The Golden bundle has 359 blinded records and **zero judgments/adjudications**. Its input hashes match the frozen steward manifest. Steward prediction, comparison and label files were hashed only, not used to choose labels. The historical ledger has 31 decisions (15 accept, 3 correct, 13 skip) for the newer FastBrainDecision taxonomy. Original identity, timestamps, label, notes and decisions are retained for nonprotected candidates. Protected decisions retain metadata and source/hash pointers without their labels.

No newer family/path review is silently transformed into complete compact gold. Only partial semantic proposals (respond→direct_response, memory→memory_recall, clarify→clarify; clarification booleans/names) are recorded with rationale. qwen_agent/fast_path cannot determine semantic compact paths; research can need one/multiple steps; escalation policy is broader than unavailable action; family is not selected tool; uncertainty is not compact confidence; requirements are not resolved references. Remaining targets require exact-schema human review. `adjudication_proposals.jsonl` contains AI-authored pending proposals, explicitly not decisions. `adjudication_ledger.jsonl` is empty and reserved for append-only approved human adjudication. No historical decision was rewritten.

`pending_review.jsonl` and `synthetic_unverified.jsonl` preserve exact utterance, original context and capabilities, original annotation, all nonsealed source revisions, original IDs/group/split, review evidence and current candidate state. Loader annotation_status remains a valid original enum; dataset_state separately expresses the six verification states. Derived files use schema_version 3.0 plus dataset_version compact-dataset-candidates-1.0.0; the old loader accepts these provenance additions. None is labeled verified just because shape validates. No translation or canonical text was invented; available source metadata is retained separately and absent forms remain null.

## Corrected implementation and remaining boundaries

1. **Missing-information conversion:** new `extract_missing` reads documented name/description fields and preserves them. The compact contract allows arbitrary nonempty missing names, so prior_request remains prior_request. FastBrain conversion is separately strict and diagnoses unknown enum values instead of stringifying objects or silently aliasing prior_request to reference. All 75 prior quarantine entries are accounted for; none becomes gold automatically.
2. **Capability omission:** explicit mappings cover all 19 registered actions, including open_folder/recent_files→file and grep_code/run_project_tests→project. Family metadata is a proposal distinct from execution path/intent/tool selection, not a new classifier class.
3. **Descriptor mismatch:** `exact_pair_features` serializes each example's own descriptor in original order and computes exactly the same request/capability/absolute-difference/product vector as inference. Same name/different description variants cannot overwrite one another. New trainer uses this branch by default with independent encoding. Focused tests establish structural parity with deterministic fake vectors; no BERT features were computed.
4. **Context/reference:** full original context/capabilities are preserved. Annotated reference values/spans and missing descriptions are not replaced to accommodate wrong current outputs. A pure tested resolver proposal requires a unique explicit inference-time resolved_references entry and returns unresolved for absent/ambiguous context; it never traverses the first arbitrary scalar or reads gold targets. The existing adapter remains unchanged. 65 candidates have deterministic output-alignment concerns documented separately; full-decision readiness also needs approved runtime/span/description policy work.
5. **Registry arguments:** current Registry is extracted statically without loading handlers. Derived copies align weather required=[]; search_files required=[keyword]; list_directory/run_project_tests required=[]. Original descriptors remain preserved. Empty Registry required lists do not imply that weather is semantically executable without a city/coordinates/location source. Such semantic prerequisites require review, not fabricated context.

Implemented files are new and isolated; the existing train_candidate.py remains unsafe for pending candidates. `train_verified_candidate.py` is a new, unexecuted entry point: it checks exact-contract gold status, two independent qualified reviewers, input/target hashes and class fitability before encoder loading; it preserves per-example descriptors and refuses existing model destinations. `proposed_training_patch.diff` and `proposed_reference_patch.diff` are unapplied integration proposals. No application/production file was changed. See approval_boundaries.md.

## Coverage and multilingual preservation

| Language tag | Observed script | Candidates | Share of 290 | Gold |
|---|---|---:|---:|---:|
| en | latin | 76 | 26.2% | 0 |
| hi | devanagari | 13 | 4.5% | 0 |
| hi-Latn | latin | 44 | 15.2% | 0 |
| mixed-en-hi | latin | 15 | 5.2% | 0 |
| mixed-en-ta | latin | 14 | 4.8% | 0 |
| mixed-en-te | latin | 14 | 4.8% | 0 |
| ta | tamil | 13 | 4.5% | 0 |
| ta-Latn | latin | 44 | 15.2% | 0 |
| te | telugu | 13 | 4.5% | 0 |
| te-Latn | latin | 44 | 15.2% | 0 |

All ten requested language/script categories are preserved, including native Telugu/Hindi/Tamil, Latin forms and English–Telugu/Hindi/Tamil mixing. Original language tags are retained; observed Unicode scripts are separate (Latin-only code switching is not called mixed script). Labels, spelling, naturalness, semantic translations and transliteration equivalents remain unverified. The English queue also needs independent review; multilingual/native/Latin expertise is explicitly requested in the blinded packet.

| Execution path | Candidate TRAIN | Candidate VALIDATION | Verified TRAIN |
|---|---:|---:|---:|
| clarify | 48 | 4 | 0 |
| direct_response | 24 | 4 | 0 |
| escalate | 40 | 1 | 0 |
| memory_recall | 24 | 7 | 0 |
| multi_step_task | 24 | 4 | 0 |
| research | 28 | 0 | 0 |
| single_action | 62 | 20 | 0 |

`class_coverage_by_split.csv` covers path, four modes, clarification/escalation, missing names, reference kind/resolved and binary capability matching. Gold train/validation/test rows explicitly have support zero. Candidate distributions are tagged UNVERIFIED_CANDIDATE and include denominators and scenario groups. `language_script_coverage.csv` gives per-split language/script and head/class denominators. `data_collection_plan.csv` requests missing target/file/application/url/query/location/time/scope/permission/reference cases and uncommon reference kinds plus actual Registry domain coverage. Zero-count rows are inventory, not fabricated examples or labels. Binary matcher classes are 0/1; capability names are domain coverage, not invented per-tool heads. Five groups is only a triage indicator for sparse scenarios, not a scientifically established 90% support threshold.

## Group splits and leakage checks

Source scenario IDs group paraphrases, translations, transliterations and template variants. NFC/casefold/whitespace duplicates are unioned, preserving text. Same-script >=0.85 character similarity across candidate groups causes conservative transitive union (1 such edges). This is leakage-prevention triage, not confirmed semantic identity; wider paraphrases/translations still need humans. Candidate group connected components are frozen in the provisional manifest with stable IDs/hash. No test labels or outcomes determine grouping, mapping, selection or fitting.

TRAIN/VALIDATION partition whole groups using seed-42 hash ordering. Every group containing an original model training ID is pinned to candidate TRAIN; approximately 20% of all groups is held as candidate VALIDATION from remaining groups. There are **zero shared IDs, normalized texts or groups**, **zero original-training IDs in candidate VALIDATION**, and **zero protected test IDs in either candidate file**. These checks establish mechanical isolation; independent semantic-label validity is NOT_VERIFIED. Candidate validation is a review/collection partition, not a new gold evaluation.

Confirmed Downloads duplicate: original `train_open_downloads-en` stays only as an unverified candidate; `v4_open_downloads-en` stays in protected source metadata and is excluded from all candidate/supervised pools. Originals are preserved. Related Downloads records remain in their source/group components. The old 359 benchmark is never promoted to independent test data. New final test stays empty: collect new independent reviewed scenario groups, confirm no training/template/translation relation, and lock their input/annotation hashes before any tuning.

## Quality gates and readiness

| Gate | Status |
|---|---|
| All 290 candidate targets pass actual loader/schema | PASS |
| Full original context and source descriptors preserved | PASS |
| Current Registry argument alignment in derived descriptors | PASS |
| Per-example train/inference pair-feature structure | PASS (offline regression) |
| Provisional candidate split ID/text/group isolation | PASS |
| Protected original-test separation | PASS |
| Original source/user-change/model hashes unchanged | PASS |
| Exact compact-schema human gold approvals | FAIL: zero |
| Representative independently verified classes/languages | NOT_VERIFIED: zero gold |
| Broad semantic/translation group completeness | NOT_VERIFIED; human review needed |
| Existing adapter reference/span/description emission alignment | WARNING: runtime proposals not applied |
| Independent reviewed locked final test | FAIL: absent |
| Training/evaluation performed | NO |

Focused offline regression suite: **33 tests passed**, zero failures/errors. Covers structured missing objects/unknown FastBrain names, every Registry action family, four Registry argument discrepancies, same-name descriptor feature parity, full-context preservation, explicit/ambiguous reference behavior, no label serialization, exact gold hash/qualification gate, candidate loader acceptance, empty/unverified fit blocking, Downloads protection and transitive groups. Synthetic unit approvals are fixtures only and are never included in datasets or human-review counts. Proposed source patches compile but are not applied or end-to-end validated.

## Next approved work

No reviewers are confirmed. Per the user's direction, this safe candidate build is finished without waiting for names or requesting them again. Use only reviewer/ for blind initial review; withhold source targets/ledger/proposals until initial judgments are frozen. Obtain two qualified independent exact-contract labels and separate adjudication for meaningful disagreement/multilingual semantics. Collect fresh diverse scenarios for absent classes and undercovered scripts without duplicate oversampling. Approve runtime reference/span/description policy and integration of the isolated gold/descriptor gate. Then rebuild approved group splits and independently collect/lock the final test. Stop before any model fitting until a separate training experiment is approved.

No 90% improvement is promised or measured. This is a structurally valid candidate dataset, **not a verified trainable gold dataset**.
