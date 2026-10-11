# Fast Classification gold benchmark: blinded review prepared

## 1. Executive summary

**Accuracy: NOT MEASURED.** The user confirmed no independent human or language reviewer is currently available. All 359 selected records have blank proposed gold labels; independently adjudicated=0, population baseline predictions=0, Qwen/model/network calls=0. A review bundle and gold-gated offline runner are prepared, not a measured classifier benchmark. Unit-test predictions are synthetic test evidence only.

The target is the deterministic-v1 seven-field Fast Brain component, not the production Qwen router, mBERT/Laya candidate, raw encoder heads, or historical annotation builder. Existing labels and the 31 decisions remain supporting evidence only. All 359 are selected for a future census of the queued population, not claimed representative of real users. 277 require qualified multilingual/script/transliteration review; 82 require independent English review.

Start with [reviewer instructions](fast_classification_gold_review_20261010_133541/reviewer/reviewer-instructions.md), [blinded records](fast_classification_gold_review_20261010_133541/reviewer/records.jsonl), [taxonomy](fast_classification_gold_review_20261010_133541/reviewer/taxonomy.md), and [annotation schema](fast_classification_gold_review_20261010_133541/reviewer/annotation.schema.json). Share only the reviewer directory before initial judgments. The steward directory contains historical labels and must stay separate.

## 2. Exact evaluation target

| Property | Specification |
| --- | --- |
| Implementation | assistant/orchestration/fast_brain/deterministic.py:12–48 |
| Entry point | DeterministicFastBrain.decide(FastBrainRequest) |
| Backend | deterministic-v1 |
| Input | Exact original transcript; actual source context mapping; advertised capability names |
| Output | Frozen seven-field FastBrainDecision, validated by actual validator |
| Checkpoint/settings | No checkpoint, generation settings, network or model |
| Future execution | Direct decide then validate; catch/report errors before safe_decide masks them |
| Version | SHA256 source hashes in evaluation-target.json and manifest.json |
| Current status | Reference gate not satisfied: prediction run deferred |

Source [deterministic.py](../assistant/orchestration/fast_brain/deterministic.py):22–48 has only stop/time/date/open Chrome/YouTube patterns; other utterances return information/respond/none. It does not use supplied context/capabilities. [adapter.py](../assistant/orchestration/fast_brain/adapter.py):11–31 defines inputs and fallback; [shadow.py](../assistant/orchestration/fast_brain/shadow.py):22–53 observes without changing the authoritative route, but safe_decide hides errors behind a valid fallback. Future evaluation captures direct errors so they remain in denominators.

Systems kept separate: [router.py](../assistant/orchestration/router.py):60–75,99–215 implements legacy TASK/LOCAL/GEMINI with local-model fallback; [graph.py](../assistant/orchestration/graph.py):264–335 connects that classifier to production. These provider categories cannot be scored as seven-field semantics. [compact_adapter.py](../assistant/orchestration/compact_adapter.py):105 onwards is compact-mbert-frozen-v3 using older ClassifierDecision; metadata names nlptown/bert-base-multilingual-uncased-sentiment, max_length=160, frozen encoder/logistic heads, CPU candidate. [encoder_adapter.py](../assistant/orchestration/fast_brain/encoder_adapter.py):62–115 loads a local encoder and creates new raw heads; it does not emit a trained seven-field decision. The model specification is design-only; no replacement candidate is integrated here. Historical builders transform expected annotations rather than infer from utterances.

## 3. Authoritative taxonomy

Enums were read from the actual contract through an isolated package loader; no application initializer was executed. [contract.py](../assistant/orchestration/fast_brain/contract.py):13–87 contains values; 137–160 deserializes all required fields.

**intent_family**: `conversation`, `information`, `deterministic_action`, `local_task`, `research`, `memory`, `multi_step_task`, `clarification`, `unsupported`.

**capability_family**: `none`, `application`, `browser`, `file`, `project`, `weather`, `location`, `reminder`, `memory`, `research`, `system`, `unknown`.

**execution_path**: `fast_path`, `respond`, `qwen_agent`, `memory`, `research`, `clarify`, `escalate`.

**uncertainty**: `high`, `medium`, `low`, `ambiguous`, `outside_envelope`.

**context_requirements**: `conversation`, `memory`, `project`, `current_information`, `observations`, `active_task`, `none`.

**clarification.required**: boolean; **clarification.missing**: `target`, `file`, `application`, `url`, `query`, `location`, `time`, `scope`, `permission`, `reference`.

**escalation.candidate**: boolean; **escalation.reason**: `current_information_required`, `unsupported_capability`, `outside_qwen_envelope`, `repeated_local_failure`, `context_overload`, `explicit_cloud_request`, `None` (None means JSON null).

All seven top-level concepts and nested required/missing/candidate/reason fields are required. [validator.py](../assistant/orchestration/fast_brain/validator.py):16–38 enforces nonempty context, exclusive none, clarification iff missing, escalation iff reason, clarify/clarification requiring missing information, and no fast_path/unknown. It does not enforce general semantic intent/path/capability compatibility, exact keys or strict boolean types. Annotation shape checks are stricter and separate; production contract is unchanged.

The reviewer taxonomy gives operational meanings for file/folder versus application, browser navigation versus research, multistep tasks, information versus conversation, memory, current-information needs, missing references and unsupported/escalation evidence. Context requirements represent dependency, not context-object existence. active_task is not automatically appropriate for an empty context. Uncertainty has no implemented calibration thresholds; the design proposes calibrated bands. A shared versioned rubric is needed before claiming strict uncertainty/full-field correctness. Do not copy baseline values to resolve that issue.

## 4. Dataset provenance and integrity

| Quantity | Count/finding |
| --- | --- |
| Source v4 records | 434 |
| Selected queued/v1 records | 359 |
| Excluded consolidation records | 75 |
| Historical accept / correct / skip | 15 / 3 / 13 |
| Historical reject / ambiguous | 0 / 0 |
| Pending without decision | 328 |
| Existing per-field independent review documentation | 0 |
| Duplicate record/decision IDs | 0 |
| Exact duplicate text | Open my Downloads folder. twice, distinct IDs |
| Semantic groups | 49 |
| Lexical near-duplicate screen | 6 pairs at SequenceMatcher>=0.9 within identical language metadata; not a semantic proof |
| Train / development / sealed_candidate | 239 / 57 / 63 |
| Known domain mapping diagnoses | 46 |
| Empty-context active_task diagnoses | 319 |
| Invalid canonical / missing serialized decision fields | 0 / 0 |

Counts were recalculated, not taken as ground truth from an old report. Historic accepts/corrections include 18 full serialized labels but have no field-rationale/independence/blinding/qualification documentation; that does not prove fields were not considered, but cannot establish independent gold. Existing gold eligible=0 under the confirmed reviewer availability. Historical IDs are anonymized only in the new steward copy; originals untouched.

The 75 excluded records include 75 source clarification candidates with missing-information annotations. Rejection is caused by normalization/schema incompatibility (for example stringifying older missing-information objects), not proof they are bad utterances. The queue therefore has no clarification labels and no clarify execution path. [Selection-gap artifact](fast_classification_gold_review_20261010_133541/steward/selection-gap.json) discloses this filter. Original rejected/adjudication artifacts remain unchanged; adjudication queue currently has 0 records.

## 5. Gold-set selection methodology

A full census of all 359 queue records is prepared. No sampling only of errors or successful baseline cases: every represented intent, capability, path, language/script and all 49 semantic groups are included, along with all 46 known mapping cases and 319 context diagnoses. Stable IDs are retained privately; blind IDs/order use ascending SHA256(gold-review-v1:record_id). Input hashes cover exact text, context, capability descriptors, language and reviewer requirements.

Historical represented intents: {'research': 43, 'local_task': 92, 'unsupported': 55, 'conversation': 42, 'memory': 38, 'information': 39, 'deterministic_action': 7, 'multi_step_task': 43}. Capabilities: {'weather': 7, 'browser': 55, 'none': 143, 'memory': 38, 'location': 7, 'file': 29, 'research': 36, 'application': 42, 'project': 2}. Execution paths: {'research': 43, 'qwen_agent': 135, 'respond': 143, 'memory': 38}. These strata come from disputed annotations and serve coverage bookkeeping only. Missing clarification/clarify/escalate/fast_path coverage and stop/time/date handler probes require a separately reviewed extension; no fabricated supplementary labels were added. Both known-error and ordinary groups are recorded only in the steward manifest, not initial review cards.

## 6. Independent review and adjudication method

Each blinded card contains exact utterance, original supplied context, advertised capabilities, language/script requirements, blank judgments and blank adjudication. Source expected labels, group names/splits, historical decisions, previous audits and predictions are withheld. No guessed translations or prefilled labels appear. Procedural directory separation is not a filesystem access control; reviewers must attest blinding honestly.

Judgments support pending, labeled (a proposed correct semantic interpretation), ambiguous and unresolved; ambiguous/unresolved keep label null. Each completed label needs all nine dimension rationales, anonymized reviewer ID, human/independent versus self-review status, qualifications, input/rubric hashes and timezone timestamp. Self-review/historical/AI labels never qualify automatically. Code-switching requires all represented languages; native scripts and romanization require corresponding expertise. Unsupported availability, context/reference and uncertainty doubts must not be forced into labels.

A second reviewer labels independently before seeing the first. Preserve both judgments. Disagreements require a separate qualified adjudicator, complete rationale and resolution notes; unresolved cases stay excluded. One verified independent review is allowed but not described as two-reviewer consensus. The gate checks declared provenance, structure and completeness; a human steward must verify real independence and linguistic competence. Freeze gold labels before prediction reveal, and retain original annotations separately. Nothing appends to the old human decision history.

## 7. Overall accuracy

**NOT MEASURED. Independently adjudicated denominator N=0; numerator=null, accuracy=null.** Missing reference labels are not counted as incorrect predictions or silently treated as correct. No population predictions were run. No confidence interval or genuine gold percentage exists.

Future strict accuracy = number of schema-valid predictions matching every required gold field/nested value / all eligible independent reference records. Invalid/missing/crashed predictions remain in this denominator as incorrect. Unresolved or unqualified references are excluded with explicit IDs/reasons; full selected coverage and exclusions are reported separately. JSON key order/spacing is ignored; lists and meaningful values are preserved.

## 8. Per-field metrics

| Dimension | Numerator | Denominator | Accuracy |
| --- | --- | --- | --- |
| intent_family | null | 0 | NOT MEASURED |
| capability_family | null | 0 | NOT MEASURED |
| execution_path | null | 0 | NOT MEASURED |
| context_requirements | null | 0 | NOT MEASURED |
| clarification.required | null | 0 | NOT MEASURED |
| clarification.missing | null | 0 | NOT MEASURED |
| escalation.candidate | null | 0 | NOT MEASURED |
| escalation.reason | null | 0 | NOT MEASURED |
| uncertainty | null | 0 | NOT MEASURED |

Each future denominator is the same eligible independent gold population, including prediction failures; numerator counts exact field equality. Schema-valid output rate, failure/abstention rate, categorical confusion matrices and precision/recall/F1 are NOT MEASURED now. The included runner supports exact/per-field/schema/failure counts and confusion matrices after eligibility; advanced subgroup/F1/group-aware intervals must be reported under the predeclared steward analysis protocol. No zero-support class is advertised as 0% measured accuracy.

## 9. Language/script breakdown

| Language/script | Selected | Independent gold | Accuracy |
| --- | --- | --- | --- |
| code_switch/mixed | 77 | 0 | NOT MEASURED |
| english/latin | 82 | 0 | NOT MEASURED |
| hindi/devanagari | 30 | 0 | NOT MEASURED |
| hindi/latin | 48 | 0 | NOT MEASURED |
| tamil/latin | 31 | 0 | NOT MEASURED |
| tamil/tamil | 13 | 0 | NOT MEASURED |
| telugu/latin | 48 | 0 | NOT MEASURED |
| telugu/telugu | 30 | 0 | NOT MEASURED |

Actual source tags distinguish mixed-en-te (31), mixed-en-hi (32), mixed-en-ta (14), and romanized versus native scripts. observed_scripts derives code-point presence separately; metadata mixed can describe code switching entirely in Latin script. Language competence is missing, not guessed by this assistant. Correlated translations/paraphrases within groups prohibit an iid real-world interpretation of sample confidence intervals. A complete queue census is a fixed-set descriptive score, not broader generalization.

## 10. Error analysis and representative examples

No classifier error rate is measured. Previously diagnosed builder examples remain supporting data-quality evidence: Archive directory and recent-file requests can lose file capability; project TODO searches lose project/file domain; empty contexts receive active_task. Their labels are withheld from first reviewers. The 46/319 counts belong to annotation diagnostics, not baseline false-prediction counts. Source-native and transliterated variants must be independently interpreted.

The known Archive request does not authorize computer_control (invalid enum). The original expected action, canonical conversion and historical corrections remain separately preserved for later comparison. “Open the second one” has source context and a resolved reference annotation; the blinded reviewer gets only real context, not the expected resolved label. Missing referents from the 75 excluded cases are a coverage gap, not automatically gold or a measured abstention failure.

## 11. Comparison against historical labels

Historical consistency: NOT MEASURED in this gold-gated run. Previous component outputs exist in old reports but are not imported as fresh predictions or independent gold. Future analysis must separately compare builder quality, target classifier versus independent gold, and Qwen-review quality (out of scope here). Agreement with synthetic/accepted annotations cannot become headline accuracy.

## 12. Limitations and unresolved records

All 359 records await independent reference labels. There are no human language reviewers; uncertainty/route/context field semantics need a documented common rubric. The original population is synthetic and filtered, with 49 related groups and missing clarification/handler strata. Existing decisions can anchor judgments and lack independent per-field provenance. Reviewer attestations are not automatically authenticated. None of these gaps is resolved by passing schema checks or tests.

## 13. Reproducibility, commands and environment

Environment: Windows, Python 3.10.11, timezone Asia/Calcutta for user-facing timestamps; no model dependencies required. Preparation: `python -B D:\JARVIS\scratch\gold_benchmark_20261010_133541\prepare.py` (new bundle only; refuses overwrite). Tests: `python -B D:\JARVIS\scratch\gold_benchmark_20261010_133541\run_tests.py` — **37 passed in 0.36 seconds (final rerun; initial pass 0.41 seconds)**. New gold-gate tests are synthetic fixtures, not real reference labels. Existing foundation/dataset tests also ran. First test-wrapper run was blocked when pyreadline tried a subprocess; disabling optional readline in the isolated test runner allowed the rerun without weakening external-action guards.

Draft 2020-12 schema and blank pending template validated by jsonschema. `git diff --check` passed. CLI for a later independently reviewed file:

```powershell
python -B -m benchmarks.fast_classifier.gold_review --reviews <new-reviewed-jsonl> --manifest D:\JARVIS\reports\fast_classification_gold_review_20261010_133541\steward\manifest.json --out <new-benchmark-output-directory>
```

This command was **not run on population labels** because no gold is available. It verifies input/rubric hashes and frozen baseline source hashes, refuses existing output, loads real source in an isolated namespace to avoid production initialization, and predicts only eligible records. Direct execution timing/errors are retained. It cannot approve labels into training data or original decision history.

## 14. Readiness recommendation

Ready for offline independent review and engineering tests; **not ready for supervised accuracy claims, training-data approval, deployment or replacement experiments based on this set**. First obtain qualified independent reviewers and agree field semantics. Preserve initial labels, resolve disagreements, freeze references, then execute the deterministic baseline. Repairing the builder or extending clarification/handler coverage needs a separately versioned dataset decision; no fix or training was performed here.

## 15. Generalization and integrity

**No measured result supports broader generalization: genuine accuracy is NOT MEASURED.** Even a later all-359 score applies to this selected synthetic queue, not arbitrary real-world speech. No Qwen, Gemini, network, model training/download/deployment, production startup, desktop/microphone or service control ran in this task. All original datasets, human decisions, historical logs/reports and production files remain unchanged. Reference projects were not accessed; nothing committed.

Integrity: 288 pre-existing files hashed, byte-identical; initial git-status entries retained. New paths only: `benchmarks/fast_classifier/gold_review.py`, `tests/test_gold_review.py`, `scratch/gold_benchmark_20261010_133541/` harness/test/evidence files, this timestamped report, and `fast_classification_gold_review_20261010_133541/` reviewer/steward artifacts. Full new-path list and file hashes are in the integrity/reproducibility manifest. Tests/scoring do not duplicate or mutate historical decisions.

Artifacts: [gold preparation](fast_classification_gold_review_20261010_133541/steward/gold-evaluation-set.jsonl), [predictions (empty)](fast_classification_gold_review_20261010_133541/steward/predictions.jsonl), [per-record comparison](fast_classification_gold_review_20261010_133541/steward/per-record-comparison.jsonl), [metrics](fast_classification_gold_review_20261010_133541/steward/aggregate-metrics.json), [error-analysis CSV](fast_classification_gold_review_20261010_133541/steward/error-analysis.csv), [review/adjudication manifest](fast_classification_gold_review_20261010_133541/steward/manifest.json), [steward instructions](fast_classification_gold_review_20261010_133541/steward/adjudication-instructions.md). Share only the reviewer directory for initial judgments.
