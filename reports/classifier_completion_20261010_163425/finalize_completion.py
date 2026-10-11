"""Finalize evidence/report only; no training, inference, or source mutation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,csv,hashlib,collections,subprocess,datetime
ROOT=Path('D:/JARVIS');OUT=Path(__file__).resolve().parent;PRIOR=ROOT/'reports/autonomous_bert_training_20261010_161410';BUILD=ROOT/'reports/multilingual_dataset_build_20261010_155129'
def js(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def dump(n,x):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def table(n,rs):
 with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
manifest=js('run_manifest.json');best=js('best_metrics.json');meta=js('best_candidate_metadata.json');actual=js('actual_encoder_evaluation.json');baseline=js('baseline_reproduction.json');previous=js('previous_best_reproduction.json');tests=js('test_results.json');curve=js('abstention_metrics.json')
cases=list(csv.DictReader((OUT/'eleven_incompatibilities.csv').open(encoding='utf-8-sig')))
train=[json.loads(l) for l in (PRIOR/'provisional_train.jsonl').read_text(encoding='utf-8').splitlines()];val=[json.loads(l) for l in (PRIOR/'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]
valby={r['id']:r for r in val};oldby={r['record_id']:r['prediction'] for r in baseline['predictions']};newby={r['record_id']:r['prediction'] for r in best['predictions']}
classifications=[]
for case in cases:
 r=valby[case['record_id']];e=r['expected'];p=newby[r['id']];unseen=case['original_id'] in ('en-memory-001','hi-codeswitch-memory-001')
 category='C: deterministic field supervision/resolution defect; D/E: exact rendering policy underspecified'
 if p['context_references']!=e['context_references']:category+='; B: remaining reference/span prediction error'
 if unseen:category+='; E: exact desired phrase absent from train vocabulary'
 if p['missing_information']!=e['missing_information']:category+='; B: missing-name/description prediction error'
 case['classification']=category
 case['unresolved']='None for structured reference/missing fields; other head errors may remain' if case['after_reference_exact']=='True' and case['after_missing_exact']=='True' else 'Unreviewed rendering convention or model/vocabulary error remains; no label rewrite or example exception'
 classifications.append({**case,'utterance':r['utterance'],'source_annotation':e,'baseline_output':oldby[r['id']],'experimental_output':p,'human_gold':False})
table('eleven_incompatibilities.csv',cases);dump('eleven_case_evidence.json',classifications)
# Compare actual encoder results at the contract level, not floating confidence bit equality.
fields=('intent','execution_path','request_modes','selected_capability','context_references','missing_information','needs_clarification','needs_escalation')
semantic_same=[];confidence_delta=[]
for p,q in zip(actual['predictions'],best['predictions']):
 cached=q['prediction'];semantic_same.append(all(sorted(p[f])==sorted(cached[f]) if f=='request_modes' else p[f]==cached[f] for f in fields));confidence_delta.append(abs(p['confidence']-cached['confidence']))
actual.update({'cache_vs_actual_required_fields_equal':all(semantic_same),'required_field_disagreement_count':sum(not x for x in semantic_same),'maximum_confidence_numeric_delta':max(confidence_delta),'comparison_note':'Full mapping bit-equality differs with floating inference batch numerics; eight required target fields compared separately.'});dump('actual_encoder_evaluation.json',actual)
dump('per_head_metrics.json',best['raw_heads']);dump('per_class_metrics.json',{k:v['report'] for k,v in best['raw_heads'].items()});dump('per_language_metrics.json',best['languages'])
for name,m in best['raw_heads'].items():dump('confusion_matrices/'+name.replace(':','_')+'.json',{'labels':m['classes'],'matrix':m['confusion'],'denominator':m['denominator'],'metric_type':'PROVISIONAL_LABEL_AGREEMENT'})
script_rows={}
for r,row in zip(val,best['predictions']):
 script='+'.join(r['language_metadata']['observed_scripts']);item=script_rows.setdefault(script,{'denominator':0,'exact':0,'groups':set(),'class_support':collections.Counter()});item['denominator']+=1;item['exact']+=row['exact'];item['groups'].add(r['group_id']);item['class_support'][r['expected']['execution_path']]+=1
dump('per_script_metrics.json',{k:{'denominator':v['denominator'],'exact':v['exact'],'scenario_groups':len(v['groups']),'class_support':dict(v['class_support']),'status':'INSUFFICIENT_GOLD_EVIDENCE'} for k,v in script_rows.items()})
failed_fields=collections.Counter()
for r,row in zip(val,best['predictions']):
 for f in fields:
  p=row['prediction'];ok=sorted(p[f])==sorted(r['expected'][f]) if f=='request_modes' else p[f]==r['expected'][f]
  if not ok:failed_fields[f]+=1
table('remaining_label_review.csv',[{'record_id':r['id'],'original_id':r['original_id'],'language':r['language'],'utterance':r['utterance'],'target_status':r['dataset_state'],'review_action':'Review semantic/rendering consistency; prediction disagreement alone is not label-error evidence','source_targets_changed':False} for r,row in zip(val,best['predictions']) if not row['exact']])
hist=list(csv.DictReader((OUT/'final_experiment_history.csv').open(encoding='utf-8-sig')))
field_table='| Target | Frozen baseline | Previous candidate | Experimental v2 |\n|---|---:|---:|---:|\n'+''.join(f'| {f} | {baseline["fields"][f]}/40 | {previous["fields"][f]}/40 | {best["fields"][f]}/40 |\n' for f in fields)
head_table='| Raw head | Agreement | Macro F1 |\n|---|---:|---:|\n'+''.join(f'| {k} | {v["agreement"]}/{v["denominator"]} | {v["report"]["macro avg"]["f1-score"]:.3f} |\n' for k,v in best['raw_heads'].items())
case_table='| Original ID | Diagnosis | Reference exact after | Missing exact after | Full exact after |\n|---|---|---|---|---|\n'+''.join(f'| {r["original_id"]} | {r["classification"]} | {r["after_reference_exact"]} | {r["after_missing_exact"]} | {r["after_full_exact"]} |\n' for r in cases)
lang_table='| Language | Exact provisional agreement | Scenario groups |\n|---|---:|---:|\n'+''.join(f'| {k} | {v["exact_agreement"]}/{v["denominator"]} | {v["scenario_groups"]} |\n' for k,v in sorted(best['languages'].items()))
curve_table='| Confidence threshold | Accepted/40 | Exact among accepted | Full-set exact including fallback/40 |\n|---|---:|---:|---:|\n'+''.join(f'| {r["threshold"]} | {r["accepted"]}/40 | {r["accepted_exact"]}/{r["accepted"]} | {r["full_set_exact"]}/40 |\n' for r in curve)
inputs_unchanged={p:sha(p)==v for p,v in manifest['inputs'].items() if OUT not in Path(p).parents or Path(p).suffix=='.joblib'}
workspace_revisions={p:{'initial_read_sha256':v,'final_sha256':sha(p),'reason':'Owned intermediate report refreshed by final tested policy; original data/model untouched'} for p,v in manifest['inputs'].items() if OUT in Path(p).parents and Path(p).suffix!='.joblib' and sha(p)!=v}
assert all(inputs_unchanged.values())
manifest['protected_input_hash_checks']=inputs_unchanged;manifest['owned_intermediate_report_revisions']=workspace_revisions;dump('run_manifest.json',manifest)
git_after=subprocess.run(['git','status','--porcelain=v1'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
(OUT/'git_status_after.txt').write_text(git_after,encoding='utf-8');before=(OUT/'git_status_before.txt').read_text(encoding='utf-8-sig').splitlines();git_preserved=set(before).issubset(set(git_after.splitlines()))
report=f'''# CLASSIFIER COMPLETION REPORT

Executive verdict: **PARTIALLY COMPLETE**. The isolated implementation, baseline reproduction, eight controlled fits, real encoder evaluation, confidence/abstention analysis and relevant regression suite are complete. General resolution/serialization defects are fixed in experimental v2, but **not all eleven examples match exactly**: unverified rendering conventions, missing vocabulary and model errors remain. Production readiness and 90% independently verified accuracy are not established.

Strict full-record provisional agreement improved **8/40 (20%) → {best['exact_agreement']}/40 ({best['exact_agreement']/40:.1%})** on the unchanged fixed validation. No ground-truth labels, scenarios or denominators were altered. Output path agreement improved **31/40 → {best['fields']['execution_path']}/40**; capability remains **{best['fields']['selected_capability']}/40**. Scores are agreement with 39 synthetic_unverified and one pending_review validation targets, **not verified accuracy**. Verified-gold denominator is zero.

## Evidence and exact baseline reproduction

Inspected canonical compact_adapter.py, decision.py, train_candidate.py, actual loaders/builders, immutable model heads and prior reports. Initial git status/diff-stat were recorded. No reference project was accessed. The dataset-build and earlier audit reports remain unchanged; the experiment starts from the prior 232-record authorized provisional TRAIN and fixed 40-record VALIDATION.

Both frozen original baseline and prior best reproduced **8/40**, with every predicted field reproduced under the actual frozenset request_modes semantics. The initial bytewise mapping comparison failed only because mode list serialization order differed across processes. This was investigated before training: canonical sorted-mode comparison shows no semantic or numeric field discrepancy. Metric definition did not change. Reproduction predictions, raw head reports, per-field metrics, confusion matrices, cached-runtime latency and failures are saved separately.

Local tokenizer/encoder snapshot and all feature-cache/checkpoint hashes match the prior verified assets. The encoder stayed frozen; only new LR coefficients were fitted. Actual local encoder execution on all 40 validation records yielded **{actual['strict_exact']}/40**, median **{actual['latency_ms']['median']:.1f} ms**, p95 **{actual['latency_ms']['p95']:.1f} ms**, plus **{actual['load_ms']:.1f} ms** encoder startup. Cached-feature runtime is separately labeled in metrics and must not substitute for encoder latency. Actual versus cached eight-field predictions agree for {sum(semantic_same)}/40 records; maximum confidence numeric delta is {max(confidence_delta):.3g}. Confidence bit-level drift is reported, not mistaken for target prediction drift.

## Root causes and implemented fixes

- `CompactMultilingualAdapter._predict_references` ignores supervised reference text and calls `_first_context_value`. It can return a topic or alias instead of the relevant decision. `experimental_adapter.py:typed_resolve` uses typed recent_topics/projects/explicit-reference records, prefers explicitly named entities, returns a unique decision/name and leaves ambiguous or malformed values unresolved. It never reads a target or arbitrary first scalar. A unit test exposed substring matching of one-letter names; whole-token matching fixed that genuine general bug.
- The original adapter always emits the full utterance as reference text and a fixed missing-information description; the original trainer fits only kind/name. Experimental v2 adds **training-only LR rendering heads** for reference_span and missing_description. Internal __utterance__/__none__ sentinels encode source identity/no-reference cases; they are not new output enums. Missing names remain older-contract nonempty strings. Unresolved span/name head conflicts are explicitly safe-fallbacked rather than silently invented. The expected source text/description is never injected into input context or changed to match the model.
- `project_reference` recognizes a uniquely explicitly mentioned project alias and extracts its real source span; typed_resolve returns the named project. This supports the legitimate untrained project domain through inference context, not fabricated training labels or a special hologram exception. Regression uses an unseen space-elevator alias, not a copied validation rule.
- `train_candidate.py` originally keyed descriptors by name. New isolated training preserves every per-example complete descriptor. Eight experiments compare full 3072-dimensional pair features versus 1536-dimensional abs-difference/product interaction features, C/weighting and capability thresholds. Scores remain poor for capabilities; rank improvements were not claimed where absent.
- The older parser validates shape but does not enforce all cross-field meanings. Experimental adapter validates final ClassifierDecision and explicitly returns existing escalation-shaped data with abstained/reason metadata for unsupported head outputs. No production caller/router was connected or changed. Experimental confidence is the minimum relevant path/reference/span/description/capability model support, **not calibrated correctness probability**.

The rendering conventions themselves are inconsistent: some source references are whole utterances, some literal phrases, some abstract English "prior discussion" or "it" for non-English requests. Both are schema-valid, but their intended semantic contract is unverified. No A (incorrect gold/source semantic label) was confirmed just because predictions disagreed. D/E denotes underspecified annotation/rendering convention, not invalid enum/schema. Remaining B errors are disclosed.

## Eleven cases: individual classification and outcomes

{case_table}
Complete per-record source annotations, baseline outputs, new outputs and reasons are in eleven_case_evidence.json; no expected label changed. Typed context-resolution tests cover **all eleven** using their actual supplied input/context. These tests verify resolver semantics independently of span-head score. Two unseen literal span phrases cannot be produced by the trained rendering vocabulary; no validation phrase was added to classes. Memory-project abstract-span predictions remain weak because training uses a different context/template distribution. Structured fields repaired in a case do not imply its other path/mode fields are correct.

## Dataset, label and feature gates

TRAIN **232 synthetic_unverified**, VALIDATION **40** (seven groups), unchanged from prior run. Eighteen pending training records remain excluded. Every record passes actual compact schema. Protected original-test IDs, duplicate/normalized-equivalent request texts and connected scenario groups do not cross the fitting/selection boundary. Validation has no baseline training-ID overlap. Broader semantic independence remains NOT_VERIFIED; prior semantic neighbor triage is not discarded or called proof of no leakage. No sealed label/outcome was read for construction or tuning; no old benchmark becomes an independent test.

Class distributions, schema/integrity counts, missing classes, multilingual tags and exclusions are machine-readable in data_quality_report.json/class_distributions.csv. Reference kind project remains absent from trained kind classes; the explicit contextual alias rule is separately tested. Missing-information conversion retains documented name/description fields; prior_request is valid in the older open-string contract and is not silently forced into a Fast Brain enum. The prior corrected family mapping covers all Registry actions including open_folder/recent_files/grep_code/run_project_tests; tests verify these omissions. Registry argument repairs remain in derived candidate copies, with original descriptors preserved.

Features contain only original user text, supplied context and advertised capabilities. Labels, reviewer history, IDs, splits and newly learned rendering targets are not in feature serialization. New artifacts require ExperimentalCompactAdapter: an interaction matcher expects 1536 features and is **not compatible with direct use by the unchanged 3072-feature production compact adapter**. The caller contract remains ClassifierDecision, but the artifact/adapter version is explicit. No production patch was applied.

## Before/after metrics: full output versus raw heads

{field_table}
{head_table}
Raw heads and postprocessed output are distinct. The main full-record metric retains **all eight supervised target fields**, including exact reference span/value and missing-information description, with mode sets canonicalized. Confidence/evidence have no supervised source target and are excluded consistently across baseline and candidates. All 40 records remain scored, including fallback. Per-class precision/recall/F1/support and confusion matrices include unseen validation classes; no class was removed to manufacture success.

Eight unique coefficient artifacts were preserved. Best selection was strict exact first, then output path/capability as disclosed tie-break. C=16, balanced L2/lbfgs, seed=42, interaction pairing and matching threshold=.25 selected `{meta['selected_experiment']}`. Exact counts across the bounded matrix are {[int(r['strict_exact']) for r in hist]}/40. Four configurations tie at the best exact count; changing pair representation does not demonstrate capability improvement. Remaining capability ranking and annotation/contract problems dominate; more repeated tuning on these 40 unverified correlated samples would not establish 90% accuracy.

## Multilingual and abstention evidence

{lang_table}
Native Hindi/Tamil/Telugu each have one validation example; English twelve, Latin transliteration seven each, code-switching one/two. These are insufficient independently reviewed samples. Per-script metrics and class support are saved; no script or language is declared production-supported from favorable small counts. Original multilingual utterances are preserved and no translations were fabricated.

{curve_table}
Accepted-set correctness is provisional only, and high rates at low coverage are **not** the 90% full-set target. A fallback can itself agree with a source unsupported-request target, so full-set exact is shown separately. Thresholds were diagnostics, not a production-calibrated policy. Main comparison remains no optional threshold, **{best['exact_agreement']}/40**. Low-support outputs use existing escalation-shaped contract data, expose abstention_reason, fabricate no capability/missing field/context value, and never execute fallback tools/providers.

## Tests, blockers and commands

Relevant suite included 12 existing classifier/benchmark/Fast Brain/annotation/shadow modules plus the new experimental tests. Final result: **{tests['passed']} passed, {tests['failed']} failed, {tests['skipped_or_blocked']} skipped/blocked**. Protected original-dataset/historical-outcome reads are blocked and explicitly skipped; no claim is made that those tests passed. Existing production logging initialization is replaced by a no-file logger only inside the test harness; production files are unchanged. All test temp artifacts are inside this run. Pytest plugin auto-loading/cache writes and network calls are disabled. Individual results/reasons appear in test_results.json and pytest_results.xml.

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

Baseline: `{ROOT/'benchmarks/fast_classifier/models/v3/compact_heads.joblib'}`
SHA-256: `{manifest['baseline_model_sha256']}`

Previous candidate: `{PRIOR/'best_candidate/compact_heads.joblib'}`
SHA-256: `{manifest['previous_candidate_sha256']}`

Selected experimental model: `{meta['model_path']}`
SHA-256: `{meta['model_sha256']}`

Adapter: `{OUT/'experimental_adapter.py'}` (hash in completion_manifest.json).

TRAIN SHA-256: `{manifest['train_sha256']}`
VALIDATION SHA-256: `{manifest['validation_sha256']}`

All changed/created files are **new files under this experimental directory**; exact inventory/hash list is in FILES_CREATED.md/completion_manifest.json. Existing application code, production routing, data/annotations/review history, baseline/previous candidate and encoder cache are unchanged. All watched input hashes match; pre-existing git status entries are preserved: {git_preserved}. No download, deployment or reference-project access occurred.

## Recommendation and acceptance boundary

**Do not deploy this candidate.** Technical experiments and general fixes are implemented, trained and tested; full semantic completion remains partial. Strict provisional agreement is only {best['exact_agreement']/40:.1%}, capability selection {best['fields']['selected_capability']/40:.1%}, several exact renderings are unsupported/inconsistent, and verified accuracy is unmeasured. Passing tests or gaining two synthetic-label matches does not establish readiness.

The **single highest-value next step** is to freeze an exact reviewed ClassifierDecision annotation/rendering policy and obtain independent, qualified human labels for diverse capability/reference scenarios under that policy. This resolves the present ambiguity about literal spans versus abstract references/descriptions and supplies real capability-domain evidence. Do not solve it by rewriting targets to mirror current predictions. The isolated implementation/model remains available for a later approved experiment, with production unchanged.
'''
(OUT/'CLASSIFIER_COMPLETION_REPORT.md').write_text(report,encoding='utf-8')
names=sorted({p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file() and 'pytest_tmp_' not in str(p) and 'test_scratch' not in p.parts}|{'FILES_CREATED.md','completion_manifest.json'})
(OUT/'FILES_CREATED.md').write_text('# New experimental files\n\nNo existing source/model/data file modified. Temporary pytest fixture files are preserved in pytest_tmp_* within this directory and are not datasets.\n\n'+'\n'.join('- `'+n+'`' for n in names)+'\n',encoding='utf-8')
dump('completion_manifest.json',{'timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat(),'verdict':'PARTIALLY COMPLETE','baseline_hash':manifest['baseline_model_sha256'],'candidate_hash':meta['model_sha256'],'inputs_unchanged':inputs_unchanged,'git_status_preserved':git_preserved,'tests':{k:tests[k] for k in ('passed','failed','skipped_or_blocked')},'outputs':{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='completion_manifest.json' and 'pytest_tmp_' not in str(p) and 'test_scratch' not in p.parts},'self_hash':'excluded to avoid recursive hashing','production_changes':False,'protected_labels_used_for_tuning':False})
print(json.dumps({'verdict':'PARTIALLY COMPLETE','best':best['exact_agreement'],'denominator':40,'tests':{k:tests[k] for k in ('passed','failed','skipped_or_blocked')},'all_inputs_unchanged':all(inputs_unchanged.values()),'actual_required_fields_match':all(semantic_same),'report':str(OUT/'CLASSIFIER_COMPLETION_REPORT.md')},indent=2))
