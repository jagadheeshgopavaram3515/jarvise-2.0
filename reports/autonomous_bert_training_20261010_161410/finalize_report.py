"""Read-only post-experiment analysis; no fitting, no encoder, no relabeling."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,csv,collections,copy,hashlib,io,unittest,datetime,subprocess,types
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];BUILD=ROOT/'reports/multilingual_dataset_build_20261010_155129'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BUILD));sys.path.insert(0,str(OUT))
pkg=types.ModuleType('assistant.orchestration');pkg.__path__=[str(ROOT/'assistant/orchestration')];sys.modules.setdefault('assistant.orchestration',pkg)
from dataset_tools import observed_scripts,compact_targets
from assistant.orchestration.compact_adapter import serialize_request,serialize_capability,_first_context_value,_compose_intent,_pair_features
from assistant.orchestration.decision import DecisionRequest,Capability
def js(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def jl(n):return [json.loads(l) for l in (OUT/n).read_text(encoding='utf-8').splitlines() if l.strip()]
def dump(n,x): (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def table(n,rows,fields):
 with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
train=jl('provisional_train.jsonl');val=jl('provisional_validation.jsonl');meta=js('best_candidate_metadata.json');best=js(meta['selected_experiment']+'/metrics.json');baseline=js('baseline_metrics.json');repro=js('reproducibility_manifest.json')
fields=('intent','execution_path','request_modes','selected_capability','context_references','missing_information','needs_clarification','needs_escalation')
# Structural ceilings diagnose output limitations; oracle targets never enter features or fitting.
structural=[]
for r in val:
 e=r['expected'];why=[]
 if _compose_intent(e['execution_path'],e['selected_capability'])!=e['intent']:why.append('deterministic_intent_composition')
 for ref in e['context_references']:
  if ref['reference']!=r['utterance']:why.append('reference_span')
  if ref.get('resolved_value') is not None and ref['resolved_value']!=_first_context_value(r['context']):why.append('resolved_reference_value')
 for m in e['missing_information']:
  if m['description']!='Required information is unresolved':why.append('missing_information_description')
 if len(e['context_references'])>1 or len(e['missing_information'])>1:why.append('first_item_only')
 if why:structural.append({'record_id':r['id'],'original_id':r['original_id'],'reasons':sorted(set(why)),'gold_label_changed':False})
ceiling={'incompatible_records':len(structural),'denominator':len(val),'maximum_possible_exact_under_unchanged_adapter':(len(val)-len(structural))/len(val),'records':structural,'interpretation':'Conservative upper bound from deterministic output incompatibility; not a measured oracle/model accuracy. All original provisional targets retained. Not used to select or rewrite labels.'}
dump('deterministic_output_limits.json',ceiling)
# Analyze existing frozen request-vector similarities only as human triage.
with np.load(OUT/'frozen_features.npz') as ar:vec=ar['vectors']
cache=js('feature_cache_manifest.json');idx={x['text_sha256']:x['index'] for x in cache['texts']}
def req(r):return DecisionRequest(r['utterance'],r['context'],tuple(Capability(c['name'],c['description'],tuple(c['required_arguments'])) for c in r['capabilities']))
def request_vec(r):return vec[idx[hashlib.sha256(serialize_request(req(r)).encode()).hexdigest()]]
x=np.stack([request_vec(r) for r in train]);y=np.stack([request_vec(r) for r in val]);similarity=y@x.T
triage=[]
for i,r in enumerate(val):
 for j in np.argsort(similarity[i])[-3:][::-1]:
  t=train[int(j)];triage.append({'validation_id':r['id'],'training_id':t['id'],'validation_group':r['group_id'],'training_group':t['group_id'],'cosine_similarity':float(similarity[i,j]),'same_observed_script':observed_scripts(r['utterance'])==observed_scripts(t['utterance']),'status':'UNCERTAIN_SEMANTIC_NEIGHBOR_NOT_CONFIRMED_LEAKAGE','review_action':'Review paraphrase/translation/template relation; fixed validation never moved into training'})
table('semantic_overlap_triage.csv',triage,['validation_id','training_id','validation_group','training_group','cosine_similarity','same_observed_script','status','review_action'])
split=js('split_integrity_report.json');split['semantic_similarity_triage']={'nearest_pairs':len(triage),'pairs_cosine_ge_098':sum(r['cosine_similarity']>=.98 for r in triage),'confirmed_semantic_leakage':'NOT_VERIFIED','new_claim_independent_accuracy':False};dump('split_integrity_report.json',split)
languages=js('per_language_metrics.json');script_metrics={}
for script in sorted({'+'.join(observed_scripts(r['utterance'])) for r in train+val}):
 positions=[i for i,r in enumerate(val) if '+'.join(observed_scripts(r['utterance']))==script];n=len(positions);k=sum(best['predictions'][i]['exact_match'] for i in positions)
 script_metrics[script]={'agreement':k,'denominator':n,'value':k/n if n else None,'status':'INSUFFICIENT_INDEPENDENT_REVIEWED_EVIDENCE','scenario_groups':len({val[i]['group_id'] for i in positions}),'execution_path_support':dict(collections.Counter(val[i]['expected']['execution_path'] for i in positions))}
dump('per_script_metrics.json',script_metrics)
for lang,metric in languages.items():
 sub=[r for r in val if r['language']==lang];supports={}
 for field in compact_targets(val[0]):
  supports[field]=dict(collections.Counter(str(v) for r in sub for v in (compact_targets(r)[field] if isinstance(compact_targets(r)[field],list) else [compact_targets(r)[field]])))
 metric['all_head_class_support']=supports;metric['training_support']=sum(r['language']==lang for r in train)
dump('per_language_metrics.json',languages)
errors=[]
for dimension in ('language','script','selected_capability','intent','missing_information','context_presence','execution_path'):
 buckets=collections.defaultdict(list)
 for i,r in enumerate(val):
  if dimension=='language':key=r['language']
  elif dimension=='script':key='+'.join(observed_scripts(r['utterance']))
  elif dimension=='context_presence':key='nonempty' if r['context'] else 'empty'
  elif dimension=='missing_information':key=r['expected']['missing_information'][0]['name'] if r['expected']['missing_information'] else '__none__'
  else:key=str(r['expected'][dimension])
  buckets[key].append(i)
 for key,ix in buckets.items():errors.append({'dimension':dimension,'value':key,'exact_agreement':sum(best['predictions'][i]['exact_match'] for i in ix),'denominator':len(ix),'mismatches':sum(not best['predictions'][i]['exact_match'] for i in ix),'metric_type':'PROVISIONAL_LABEL_AGREEMENT'})
table('error_analysis_by_slice.csv',errors,['dimension','value','exact_agreement','denominator','mismatches','metric_type'])
# Count/train class support without silently dropping zero-support validation classes.
support=[]
for field in compact_targets(val[0]):
 universe=set(str(v) for records in (train,val) for r in records for v in (compact_targets(r)[field] if isinstance(compact_targets(r)[field],list) else [compact_targets(r)[field]]))
 for cl in sorted(universe):
  row={'target_field':field,'class_label':cl}
  for name,records in [('train',train),('validation',val)]:
   matching=[r for r in records if cl in list(map(str,compact_targets(r)[field] if isinstance(compact_targets(r)[field],list) else [compact_targets(r)[field]]))]
   row[name+'_count']=sum(sum(str(v)==cl for v in (compact_targets(r)[field] if isinstance(compact_targets(r)[field],list) else [compact_targets(r)[field]])) for r in records)
   row[name+'_groups']=len({r['group_id'] for r in matching})
  support.append(row)
table('class_support.csv',support,['target_field','class_label','train_count','train_groups','validation_count','validation_groups'])
# Run focused experiment checks and the prior pure regression suite, never fitting.
import test_experiment_integrity,test_dataset_tools
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(test_experiment_integrity),unittest.defaultTestLoader.loadTestsFromModule(test_dataset_tools)])
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
(OUT/'test_output.txt').write_text(stream.getvalue(),encoding='utf-8')
dump('test_results.json',{'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'scope':'13 actual experiment integrity checks plus 33 pure builder/gold/descriptor/reference checks; no new training/inference'})
if not result.wasSuccessful():raise RuntimeError('Final regression checks failed')
headtable='| Head (raw predictions) | Agreement | Provisional rate | Macro F1 |\n|---|---:|---:|---:|\n'
for head,m in best['heads'].items():
 if head.startswith('output:'):continue
 headtable+=f"| {head} | {m['agreement']}/{m['denominator']} | {m['accuracy']:.1%} | {m['macro_avg']['f1-score']:.3f} |\n"
fieldtable='| Output field | Baseline | Best experimental |\n|---|---:|---:|\n'+''.join(f"| {f} | {baseline['fields'][f]['agreement']}/40 | {best['fields'][f]['agreement']}/40 |\n" for f in fields)
langtable='| Language | Strict exact agreement | Validation scenarios | Evidence |\n|---|---:|---:|---|\n'+''.join(f"| {lang} | {m['agreement']}/{m['denominator']} | {m['independent_scenario_groups']} | insufficient independently reviewed evidence |\n" for lang,m in sorted(languages.items()))
report=f'''# Autonomous frozen multilingual BERT + Logistic Regression experiment

**90% provisional target NOT REACHED.** Baseline and best experimental candidate each achieved **8/40 = 20.0% exact full-record provisional-label agreement**. Training actually ran on **232 synthetic_unverified records**; validation was fixed at **40 records in seven groups**. Four controlled experiments completed, then stopped after three consecutive experiments without improving the primary metric, as explicitly required. No verified accuracy was measured: independently verified gold denominator is **0**.

## Baseline, inputs and authorization

The candidate directory `reports/multilingual_dataset_build_20261010_155129` and all its recorded output hashes matched before use. The baseline joblib and JSON metadata exist and were fingerprinted. Source inputs, candidate files, prior user-modified tracked files and local encoder artifacts remained unchanged after the run. The frozen baseline was never overwritten or installed anywhere. No production adapter/routing file or historical annotation was edited. This run contains only new models, analysis and separately versioned provisional datasets.

The user's experiment-specific authorization allows structurally valid synthetic_unverified fitting, distinct from verified gold readiness. Of 250 candidate TRAIN rows, **232** passed and were used; **18 pending-reconciliation rows** were excluded, with reasons in exclusions.json. Every one of the **40** candidate VALIDATION rows stayed in the denominator: 39 synthetic_unverified and one pending_review. No difficult validation example was moved into training or silently excluded. Original utterances/targets/context/capabilities/status/provenance are preserved; no labels were changed because of predictions. Ambiguous/rejected/protected rows did not enter training. There is no new sealed-test result.

Validation has zero original baseline training-ID membership, zero train/validation ID overlap, zero normalized text overlap, zero declared connected-group overlap and zero protected original-test IDs. Scenario/translation/template groups from the prior build are preserved. These mechanical checks are PASS; completeness of semantic grouping remains NOT_VERIFIED. Additional frozen-vector neighbor triage is saved in semantic_overlap_triage.csv, not treated as confirmed duplicate or gold label evidence. Results must not be described as independently verified generalization or real-world accuracy.

## Local encoder and feature parity

Local encoder and tokenizer were available at `{js('feature_parity_report.json')['snapshot']}`. The checkpoint/model/tokenizer files were hashed before/after, including the 669 MB model.safetensors. No downloads or external APIs were used. Hugging Face offline flags plus a network/write audit guard were enabled. Encoder is AutoModel for nlptown/bert-base-multilingual-uncased-sentiment; sentiment classifier ignored; eval/inference_mode, every parameter requires_grad=False; no encoder optimizer/backward/fine-tuning.

Actual offline extraction produced 288 unique serialized texts with finite 768-dimensional request/capability embeddings and finite 3072-dimensional matching features. Attention-mask mean pooling, L2 normalization, max_length=160 and independent descriptor encoding match the existing adapter. No serialized text exceeded 160 tokens. Full per-example descriptors are preserved: cache keys use complete serialized text, not just name. Gold targets, IDs, reviewer evidence and statuses never enter the model feature text. Corrected pair-feature construction exactly equals inference construction. One actual encoder-forward adapter prediction matched cached-feature replay; all baseline/experimental decisions otherwise replay those frozen vectors through the **unmodified real adapter's postprocessing**. Frozen feature-cache/input hashes make reuse explicit and versioned.

## Search and stopping

Predeclared search_plan.json contains at most 20 configurations for C, weighting, solver/penalty and seed/convergence options. The actual completed sequence used balanced L2/lbfgs, seed 42, max_iter=3000 and C=[4,1,16,0.25]. Exact agreements were **[8,7,8,6]/40**. Experiment 01 established the best experimental result; experiments 02–04 did not improve it, triggering plateau stopping. Unrun solver/weight/seed configurations are disclosed in the plan; they were not claimed tested. No validation-selected relabel, metric relaxation, threshold alteration or dataset swapping was applied.

Each experiment fitted all 11 Logistic Regression heads on TRAIN features only and saved a unique compressed joblib, configuration, classes, convergence diagnostics, predictions and metrics. At least two fitted classes per head were checked. Classes are compared through model.classes_, including the positive index in the actual capability/boolean inference code. The matcher is binary 0/1, not a per-tool classifier. Missing names/reference kinds retain their original older-contract class spaces. Training-only class inventory and zero-support validation classes appear in class_support.csv; reference kind **project** is absent from fitted training targets and remains counted as a validation error opportunity.

The best new artifact **ties the baseline on the primary metric**; it is not an established improvement and is not a deployment recommendation. Its mode/reference field agreements are slightly below baseline. Raw head/path improvements are not substituted for full-record agreement.

## Actual metrics and denominators

Primary exactness covers all eight supervised ClassifierDecision target fields: intent, execution_path, request_modes, selected_capability, context_references, missing_information, needs_clarification and needs_escalation. Reference spans/values and missing descriptions are exact; modes are compared as sets. Confidence/evidence are not supplied supervised targets and are not included. Missing/failed predictions count incorrect; no critical target/class was dropped. Baseline and experimental schema prediction failures are {baseline['prediction_failures']} and {best['prediction_failures']}.

{fieldtable}
{headtable}
Per-class precision, recall, F1, support and confusion matrices (including zero support) are in per_class_metrics.json and confusion_matrices/. Raw path head is **33/40**, whereas actual output path is **31/40** after clarification/escalation/capability postprocessing. Capability matcher has **36/77** binary pair agreement; selected capability output is **20/40**. Neither binary agreement nor a narrow successful field constitutes full-decision accuracy. Exact primary remains 8/40.

## Multilingual results

{langtable}
All ten language tags present in training are disclosed. Each script/language group has its actual denominator, field agreements, per-head class supports and training support. Per-script results are in per_script_metrics.json. Native Telugu/Hindi/Tamil each have only **one validation example**; 1/1 cannot establish language support. Latin transliteration groups have seven each, code-switching groups one/two, English twelve. Every group lacks independently reviewed gold; even favorable percentages remain insufficient evidence. Translations/template variants are not independent semantic examples.

## Why the primary metric remains low

**{len(structural)}/40 records have objectively incompatible deterministic outputs under the unchanged adapter**, establishing a conservative maximum strict exactness of **{ceiling['maximum_possible_exact_under_unchanged_adapter']:.1%}**, even if learned heads were perfect. Annotated reference spans differ from the adapter's complete-utterance output; explicitly annotated resolved values differ from arbitrary first-context-scalar traversal; missing-information descriptions differ from the fixed output description. This is an implementation/target alignment problem, not proof those labels are wrong. Targets were not rewritten to inflate agreement. Structural upper-bound analysis is diagnostic only, never a model score or fitting input.

Other remaining errors include poor capability compatibility/ranking across scenarios, original limited positive domain coverage, rare references (project missing from train), synthetic template dependence, path/clarification/escalation interactions, and unverified multilingual/semantic labels. Error slices are saved by language, script, capability, intent, missing information, context and path. All 32 nonexact validation records are in proposed_label_review.csv; predicted disagreement is explicitly not proof of annotation error. Corrected descriptor handling did not improve the primary metric in this fixed experiment. Strong semantic overlap or annotation inconsistency requires human confirmation; no automatic correction was inferred from similarity.

The existing production adapter was not modified. Proposed runtime/input/reference policy work would need independent semantics and source evidence, not an oracle that injects gold annotations into features. The prior dataset build's isolation and objective conversion/mapping repairs were retained. Further same-intervention fitting stopped at the mandated plateau; safe independent error/coverage/hash analysis continued.

## Artifacts, tests and reproducibility

Selected experimental model: `{meta['best_model_path']}`

SHA-256: `{meta['best_model_sha256']}`

Baseline model SHA-256: `{repro['input_hashes'][str(ROOT/'benchmarks/fast_classifier/models/v3/compact_heads.joblib')]['sha256']}`

Training dataset SHA-256: `{sha(OUT/'provisional_train.jsonl')}`

Validation dataset SHA-256: `{sha(OUT/'provisional_validation.jsonl')}`

Software versions, complete encoder/input/output fingerprints, seeds, CPU thread settings, exact training/validation IDs and elapsed time are recorded in training_manifest.json and reproducibility_manifest.json. Every candidate and the selected copy are newly versioned under this run. All recorded source/model/cache hashes were rechecked; originals unchanged. Git status before/after preserves pre-existing user changes. The manifests exclude their own recursive hash by design.

Focused offline regression/integrity suite: **{result.testsRun} tests passed**, zero failures/errors (13 actual experiment-output checks and 33 previously constructed pure builder/gold/descriptor/reference checks). Test fixtures never became real labels. No production routing was connected and no original artifact replaced.

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
'''
(OUT/'autonomous_training_report.md').write_text(report,encoding='utf-8')
git_after=subprocess.run(['git','status','--porcelain=v1'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
(OUT/'git_status_after.txt').write_text(git_after,encoding='utf-8')
repro['input_hashes_unchanged_after_finalization']={p:sha(p)==info['sha256'] for p,info in repro['input_hashes'].items()}
assert all(repro['input_hashes_unchanged_after_finalization'].values())
repro['test_results']=js('test_results.json');repro['scripts']['finalize_report.py']=sha(__file__);repro['scripts']['test_experiment_integrity.py']=sha(OUT/'test_experiment_integrity.py')
repro['timestamp_completed']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat()
names=sorted({p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file()}|{'FILES_CREATED.md'})
(OUT/'FILES_CREATED.md').write_text('# New run files\n\nEvery listed file was created only under this run directory. No existing project/model/data file was modified.\n\n'+'\n'.join('- `'+n+'`' for n in names)+'\n',encoding='utf-8')
repro['outputs']={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='reproducibility_manifest.json'}
repro['self_hash_policy']='Reproducibility manifest excluded from its own output hashes to avoid recursion.'
dump('reproducibility_manifest.json',repro)
print(json.dumps({'tests_passed':result.testsRun,'strict_upper_bound':ceiling['maximum_possible_exact_under_unchanged_adapter'],'structural_incompatibilities':len(structural),'source_hashes_unchanged':True,'selected_model_sha256':meta['best_model_sha256'],'report':str(OUT/'autonomous_training_report.md')},indent=2))
