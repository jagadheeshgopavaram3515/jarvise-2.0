"""Read-only forensic analysis. Writes exclusively beside this script; no inference."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, csv, hashlib, re, unicodedata, difflib, collections, datetime, ast
ROOT=Path('D:/JARVIS'); OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
inputs={}
def read(p):
 p=ROOT/p; b=p.read_bytes(); inputs[p.relative_to(ROOT).as_posix()]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}; return b.decode('utf-8-sig')
def js(p): return json.loads(read(p))
def rows(p): return [json.loads(x) for x in read(p).splitlines() if x.strip()]
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
def table(name,data,fields):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fields,extrasaction='ignore'); w.writeheader(); w.writerows(data)
def script(t):
 found=[]
 for n,a,b in [('telugu',0xc00,0xc7f),('devanagari',0x900,0x97f),('tamil',0xb80,0xbff)]:
  if any(a<=ord(c)<=b for c in t): found.append(n)
 if any(c.isalpha() and 'LATIN' in unicodedata.name(c,'') for c in t): found.append('latin')
 return '+'.join(found) or 'unknown'
def norm(t): return re.sub(r'\s+',' ',unicodedata.normalize('NFC',t).casefold()).strip()
meta=js('benchmarks/fast_classifier/models/v3/compact_heads.json'); tids=set(meta['training_example_ids'])
sources=['benchmarks/fast_classifier/v1/seed.jsonl']+[f'benchmarks/fast_classifier/v{v}/dataset.jsonl' for v in (2,3,4)]+['benchmarks/fast_classifier/fast_brain_v1/dataset.jsonl']
data={}; sealed_original=set(); counts={}
# Discard test target payloads before any label analysis; only IDs/text/group/split metadata survives.
for p in sources:
 raw=rows(p); counts[p]=len(raw); data[p]=[]
 for r in raw:
  if r.get('split')=='test':
   sealed_original.add(r['id']); r={k:r[k] for k in ('id','split','utterance','language','group_id','annotation_status','review_status','source','provenance') if k in r}; r['_protected']=True
  r['_source']=p; data[p].append(r)
hist=sources[-1]; v3=sources[2]; v4=sources[3]
for r in data[hist]:
 if r.get('split')=='sealed_candidate' or r.get('original_id') in sealed_original:
  for k in ('original_label','intent_family','capability_family','clarification_required','missing_information'): r.pop(k,None)
  r['_protected']=True
reviews=rows('benchmarks/fast_classifier/fast_brain_v1/review_decisions.jsonl')
byhist={r['id']:r for r in data[hist]}; review_counts=collections.Counter(x.get('reviewer_decision','') for x in reviews)
safe_reviews=[]
for x in reviews:
 if byhist.get(x.get('record_id'),{}).get('_protected'): x.pop('reviewed_label',None)
 else: safe_reviews.append(x)
reviewby={x['record_id']:x for x in reviews}
codepaths=['assistant/orchestration/compact_adapter.py','assistant/orchestration/decision.py','assistant/orchestration/fast_brain/contract.py','benchmarks/fast_classifier/train_candidate.py','benchmarks/fast_classifier/evaluate.py','benchmarks/fast_classifier/prepare_v2.py','benchmarks/fast_classifier/prepare_v3.py','benchmarks/fast_classifier/prepare_v4.py','benchmarks/fast_classifier/build_fast_brain_dataset.py','benchmarks/fast_classifier/fast_brain_v1/schema.py','benchmarks/fast_classifier/fast_brain_v1/split.py','benchmarks/fast_classifier/fast_brain_v1/review.py','assistant/tools/registry.py','assistant/orchestration/graph.py','benchmarks/fast_classifier/run_phase3c_evaluation.py','benchmarks/fast_classifier/run_phase4_comparison.py']
code={p:read(p) for p in codepaths}
manifests={f'v{v}':js(f'benchmarks/fast_classifier/v{v}/split_manifest.json') for v in (2,3,4)}
manifests['historical']=js('benchmarks/fast_classifier/fast_brain_v1/split_manifest.json')
analysis=js('benchmarks/fast_classifier/fast_brain_v1/dataset_analysis.json'); stale=js('benchmarks/fast_classifier/fast_brain_v1/review_analysis.json')
rejected=rows('benchmarks/fast_classifier/fast_brain_v1/rejected_records.jsonl')
reject_counts=collections.Counter(x.get('rejection_reason','') for x in rejected)
from benchmarks.fast_classifier.evaluate import _parse_example
from assistant.orchestration.compact_adapter import _compose_intent,_first_context_value,serialize_request,serialize_capability
from assistant.orchestration.decision import DecisionRequest
from assistant.orchestration.fast_brain.contract import FastBrainDecision
from benchmarks.fast_classifier.fast_brain_v1.schema import FastBrainRecord
from benchmarks.fast_classifier.fast_brain_v1.review import ReviewDecision
sus=[]; context=[]; checks=[]; dist=[]; lang=[]; dup=[]
def issue(r,field,label,typ,sev,evidence,action):
 if r.get('_protected'): return
 sus.append(dict(record_id=r['id'],source_file=r['_source'],split=r.get('split',''),training_overlap=(r.get('original_id',r['id']) in tids),language=r.get('language',r.get('language_group','')),script=script(r.get('utterance',r.get('text',''))),original_utterance=r.get('utterance',r.get('text','')),target_field=field,current_label=json.dumps(label,ensure_ascii=False),issue_type=typ,severity=sev,evidence=evidence,suggested_review_action=action,review_status=r.get('review_status',r.get('annotation_status',''))))
def ctx(r,typ,evidence):
 context.append(dict(record_id=r['id'],source_file=r['_source'],split=r.get('split',''),issue_type=typ,evidence=evidence,severity='WARNING'))
def labels(r):
 if r.get('_protected'): return {}
 if 'expected' not in r: return {k:r[k] for k in ('intent_family','capability_family','clarification_required') if k in r}
 e=r['expected']; refs=e['context_references']; miss=e['missing_information']
 result={'execution_path':e['execution_path'],'clarification':int(e['needs_clarification']),'escalation':int(e['needs_escalation']),'missing_information':miss[0]['name'] if miss else '__none__','reference_kind':refs[0]['kind'] if refs else '__none__','reference_resolved':int(bool(refs and refs[0].get('resolved_value') is not None)),'intent_contract':e['intent'],'selected_capability_contract':e.get('selected_capability') or '__none__'}
 for m in ('conversational','memory','research','action'): result['request_mode:'+m]=int(m in e['request_modes'])
 return result
valid_reviews=[]; invalid_reviews=[]
for x in safe_reviews:
 try:
  ReviewDecision(record_id=x['record_id'],reviewer_id=x.get('reviewer_id',''),decision=x.get('reviewer_decision',''),reviewed_label=x.get('reviewed_label'),adjudication_status=x.get('adjudication_status','')).validate()
  if x.get('reviewer_decision') in ('accept','correct') and x.get('reviewed_label') and x.get('adjudication_status')!='requires_adjudication': valid_reviews.append(x)
 except Exception as ex: invalid_reviews.append({'id':x.get('record_id'),'error':str(ex)})
for p,rs in data.items():
 for r in rs:
  if r.get('_protected'): continue
  try:
   if p==hist: FastBrainRecord.from_mapping(r)
   else: _parse_example(r)
  except Exception as ex: checks.append({'source':p,'id':r['id'],'error':str(ex)}); issue(r,'schema',None,'schema_violation','HIGH',str(ex),'Validate the source annotation against the original schema.')
  if p==hist:
   if r['id'] not in reviewby: issue(r,'review_status',r.get('review_status'),'no_human_review_decision','HIGH','No stable-ID review decision exists.','Obtain independent review; retain historical provenance.')
   old=json.loads(r.get('original_label','{}'))
   if old.get('selected_capability') in ('open_folder','recent_files','grep_code','run_project_tests') and r.get('capability_family')=='none': issue(r,'capability_family','none','builder_capability_omission','HIGH',f"Original selected_capability={old['selected_capability']}; builder capability() omits this real Registry capability.",'Review mapping against the Registry; do not automatically relabel.')
   ctx(r,'original_input_not_embedded','Consolidated record contains context_resolvable but no original context/capabilities; original_id joins v4 for reconstruction.')
   recorded=r.get('script_group'); actual=script(r['text'])
   if recorded=='mixed' and actual=='latin': issue(r,'script_group',recorded,'code_switch_not_mixed_script','MEDIUM','Utterance is Latin-only; mixed language does not imply mixed script.','Review language and script independently.')
   x=reviewby.get(r['id'],{}); lab=x.get('reviewed_label',{})
   if lab and lab.get('context_requirements')==['active_task']:
    original=next((a for a in data[v4] if a['id']==r.get('original_id')),None)
    if original and not original.get('context'): issue(r,'context_requirements',['active_task'],'context_requirement_unsubstantiated','MEDIUM','Reviewed label requires active_task; original input has empty context. Requirement could be intended future dependency.','Clarify annotation meaning with reviewer; preserve historical decision.')
   continue
  e=r['expected']; caps=r['capabilities']; names=[c['name'] for c in caps]
  if e.get('selected_capability') and e['selected_capability'] not in names: issue(r,'selected_capability',e['selected_capability'],'unadvertised_capability','HIGH','Selected name absent from advertised descriptors.','Review unsupported/executable boundary.')
  if e['needs_clarification'] and e['needs_escalation']: issue(r,'uncertainty_flags',True,'mutually_competing_flags','HIGH','Both flags true; adapter suppresses one.','Review intended clarification/escalation priority.')
  if p==v3:
   composed=_compose_intent(e['execution_path'],e.get('selected_capability'))
   if composed!=e['intent']: issue(r,'intent',e['intent'],'unreachable_exact_intent','HIGH',f"Adapter deterministically emits {composed!r} even with gold path/capability.",'Review contract and evaluation expectations; no intent head exists.')
   for ref in e['context_references']:
    if ref.get('resolved_value') is not None and _first_context_value(r.get('context',{}))!=ref['resolved_value']: issue(r,'context_references',ref,'resolver_first_scalar_mismatch','HIGH',f"Adapter chooses first context scalar {_first_context_value(r.get('context',{}))!r}, not annotated resolved_value.",'Review reference-resolution implementation separately from gold correctness.')
    if ref['reference']!=r['utterance']: issue(r,'context_references.reference',ref['reference'],'unreachable_exact_reference_span','MEDIUM','Adapter always emits the complete user utterance for reference span.','Review exact-match metric versus semantic reference contract.')
   for mi in e['missing_information']:
    if mi.get('description','')!='Required information is unresolved': issue(r,'missing_information.description',mi.get('description',''),'unreachable_exact_missing_description','MEDIUM','Adapter emits a fixed description; training consumes only first name.','Review whether descriptions should be scored verbatim.')
   if len(e['context_references'])>1 or len(e['missing_information'])>1: issue(r,'structured_targets',None,'first_item_only','HIGH','Training retains first item; adapter emits at most one.','Review lost multi-reference/missing-field supervision.')
  c=r.get('context',{})
  if not c: ctx(r,'empty_context_observation','Empty context is valid for standalone requests; not automatically defective.')
  if e['execution_path']=='memory_recall' and not c: ctx(r,'memory_without_context','Memory-recall label with no supplied memory/context.')
  if len(names)!=len(set(names)): ctx(r,'duplicate_capability_name','Name-keyed training descriptor cache overwrites duplicate names.')
  if e['needs_clarification'] and not e['missing_information']: ctx(r,'missing_clarification_information','Clarification target has no missing-information item.')
 for split in sorted({r.get('split','') for r in rs}):
  subset=[r for r in rs if r.get('split','')==split and not r.get('_protected')]; n=len(subset)
  fields=sorted({f for r in subset for f in labels(r)})
  for f in fields:
   co=collections.Counter(str(labels(r)[f]) for r in subset if f in labels(r))
   for cl,num in sorted(co.items()): dist.append(dict(source_file=p,split=split,target_field=f,class_label=cl,count=num,labelled_total=sum(co.values()),proportion=num/sum(co.values()),rare_lt_5=num<5,status='VERIFIED_COUNT_NOT_VERIFIED_GOLD'))
  if p!=hist:
   pairs=collections.Counter(int(c['name']==r['expected'].get('selected_capability')) for r in subset for c in r['capabilities'])
   for cl,num in pairs.items(): dist.append(dict(source_file=p,split=split,target_field='capability_matcher',class_label=cl,count=num,labelled_total=sum(pairs.values()),proportion=num/sum(pairs.values()),rare_lt_5=num<5,status='VERIFIED_COUNT_NOT_VERIFIED_GOLD'))
  for (lg,sc) in sorted({(r.get('language',r.get('language_group','')),script(r.get('utterance',r.get('text','')))) for r in subset}):
   group=[r for r in subset if r.get('language',r.get('language_group',''))==lg and script(r.get('utterance',r.get('text','')))==sc]
   for f in fields:
    co=collections.Counter(str(labels(r)[f]) for r in group if f in labels(r))
    for cl,num in co.items(): lang.append(dict(source_file=p,split=split,language=lg,observed_script=sc,target_field=f,class_label=cl,count=num,independent_scenario_groups=len({r.get('group_id',r['id']) for r in group}),independent_human_reviewed_count=sum(r['id'] in {x['record_id'] for x in valid_reviews} for r in group),coverage_status='NOT_VERIFIED_ADEQUATE'))
summary={}
for p,rs in data.items():
 stat={'records':len(rs),'splits':dict(collections.Counter(r.get('split','') for r in rs)),'id_duplicates':len(rs)-len({r['id'] for r in rs}),'training_id_overlap':sum(r.get('original_id',r['id']) in tids for r in rs),'protected_label_records':sum(bool(r.get('_protected')) for r in rs)}
 for kind,key in [('exact',lambda r:r.get('utterance',r.get('text',''))),('normalized',lambda r:norm(r.get('utterance',r.get('text','')))),('declared_group',lambda r:r.get('group_id',r['id']))]:
  groups=collections.defaultdict(list)
  for r in rs: groups[key(r)].append(r)
  gg=[g for g in groups.values() if len(g)>1]; cross=[g for g in gg if len({r.get('split','') for r in g})>1]
  stat[kind]={'groups':len(gg),'records_in_groups':sum(map(len,gg)),'excess_records':sum(len(g)-1 for g in gg),'cross_split_groups':len(cross),'cross_split_records':sum(map(len,cross))}
  for i,g in enumerate(gg):
   safe=[r for r in g if not r.get('_protected')]; fingerprints={json.dumps(labels(r),sort_keys=True) for r in safe}; conflict=len(fingerprints)>1
   ctxs={json.dumps(r.get('context'),sort_keys=True) for r in safe}
   for r in g: dup.append(dict(source_file=p,group_id=f'{kind}-{i}',match_type=kind,record_id=r['id'],original_id=r.get('original_id',''),split=r.get('split',''),training_overlap=r.get('original_id',r['id']) in tids,cross_split=len({a.get('split','') for a in g})>1,label_conflict=('NOT_VERIFIED_SEALED' if r.get('_protected') else conflict),context_differs=len(ctxs)>1,representative_utterance=g[0].get('utterance',g[0].get('text','')),evidence='Declared scenario group, not identical requests' if kind=='declared_group' else 'NFC/casefold/whitespace only' if kind=='normalized' else 'Exact original text'))
   if conflict and kind!='declared_group':
    for r in safe: issue(r,'targets',labels(r),'duplicate_different_targets','HIGH' if len(ctxs)==1 else 'MEDIUM',f'{kind} duplicate with different targets; context differs={len(ctxs)>1}.','Review context/capabilities before deciding whether labels conflict.')
 summary[p]=stat
# Conservative character similarity triage: same observed script, cross-split only, never a confirmed semantic duplicate.
for p in (v3,v4,hist):
 rs=data[p]; near=[]
 for i,a in enumerate(rs):
  ta=norm(a.get('utterance',a.get('text','')))
  for b in rs[i+1:]:
   tb=norm(b.get('utterance',b.get('text','')))
   if a.get('split')==b.get('split') or ta==tb or min(len(ta),len(tb))<12 or script(ta)!=script(tb): continue
   score=difflib.SequenceMatcher(None,ta,tb,autojunk=False).ratio()
   if score>=.85:
    near.append((a,b,score))
    for r in (a,b): dup.append(dict(source_file=p,group_id=f'near:{a["id"]}:{b["id"]}',match_type='uncertain_near_text',record_id=r['id'],original_id=r.get('original_id',''),split=r.get('split',''),training_overlap=r.get('original_id',r['id']) in tids,cross_split=True,label_conflict='NOT_VERIFIED',context_differs='',representative_utterance=a.get('utterance',a.get('text','')),evidence=f'Same script; character similarity={score:.3f}; human semantic review required'))
 summary[p]['uncertain_cross_split_near_pairs']=len(near)
# All records get an overlap row, including non-overlap; absence of ID is not proof of independence.
for p,rs in data.items():
 for r in rs: dup.append(dict(source_file=p,group_id='training_id_membership',match_type='training_id_overlap',record_id=r['id'],original_id=r.get('original_id',''),split=r.get('split',''),training_overlap=r.get('original_id',r['id']) in tids,cross_split='',label_conflict='',context_differs='',representative_utterance='',evidence='Exact stable-ID lookup only; non-membership does not rule out related-example leakage'))
artifactpath=ROOT/'benchmarks/fast_classifier/models/v3/compact_heads.joblib'
inputs[str(artifactpath.relative_to(ROOT))]={'sha256':hashlib.sha256(artifactpath.read_bytes()).hexdigest(),'bytes':artifactpath.stat().st_size}
import joblib, numpy as np
artifact=joblib.load(artifactpath)
headinfo={}
heads={k:artifact[k] for k in ('execution_path','clarification','escalation','missing_information','reference_kind','reference_resolved','capability_matcher')}; heads.update({'mode:'+k:v for k,v in artifact['request_modes'].items()})
for k,h in heads.items(): headinfo[k]={'type':type(h).__name__,'classes':h.classes_.tolist(),'n_features_in':int(h.n_features_in_),'finite_weights':bool(np.isfinite(h.coef_).all() and np.isfinite(h.intercept_).all()),'class_weight':h.class_weight,'C':h.C,'n_iter':h.n_iter_.tolist()}
metadata_match={k:artifact.get(k)==v for k,v in meta.items()}
trainrows=[r for r in data[v3] if r.get('split')=='train']; trainids={r['id'] for r in trainrows}
descriptor_variants=collections.defaultdict(set)
for r in trainrows:
 for c in r['capabilities']: descriptor_variants[c['name']].add(json.dumps(c,sort_keys=True))
variants={k:len(v) for k,v in descriptor_variants.items() if len(v)>1}
for r in trainrows:
 for c in r['capabilities']:
  if c['name'] in variants: issue(r,'capabilities',c,'name_keyed_descriptor_overwrite','HIGH','Independent training uses last descriptor per name; inference uses each request descriptor.','Review descriptor consistency and original training order.')
artifactintegrity={'metadata_equals_joblib':metadata_match,'dataset_hash_matches_artifact':inputs[v3]['sha256']==artifact['dataset_sha256'],'training_ids_match_train_split':trainids==tids,'missing_training_ids':sorted(tids-trainids),'extra_train_split_ids':sorted(trainids-tids),'heads':headinfo,'descriptor_name_variants':variants,'encoder_revision_and_weight_hash':'NOT_VERIFIED: artifact records hub ID, not resolved revision or weight hash','embedding_cache':'No persistent embedding cache in train_candidate.py; arrays generated per run','fine_tuning':'No encoder optimizer/backward; eval/inference_mode with frozen features; only LR fit calls','sealed_policy':'Test targets discarded before analysis; sealed_candidate and original test lineage labels excluded; no test outcomes read','historical_review_summary_stale':stale.get('reviewed_records',0)!=len(reviews),'schema_errors_nonsealed':checks,'review_validation_errors_nonsealed':invalid_reviews}
provenance={'status':'PASS' if artifactintegrity['dataset_hash_matches_artifact'] and artifactintegrity['training_ids_match_train_split'] else 'FAIL','sources':{p:{**inputs[p],**summary[p]} for p in sources},'model_metadata':meta,'artifact_sha256':inputs[str(artifactpath.relative_to(ROOT))]['sha256'],'metadata_matches_artifact':all(metadata_match.values()),'builders':['v1 seed -> prepare_v2.py (scenario synthesis/adjudication) -> prepare_v3.py (derive v2 + 84 targeted synthetic records) -> train_candidate.py split=train -> frozen BERT -> LR heads','v3 -> prepare_v4.py (additional test scenarios) -> build_fast_brain_dataset.py -> 359 historical queue records + 75 rejected; grouped re-split'],'human_review_decision_counts':dict(review_counts),'human_review_rows':len(reviews),'unique_review_ids':len(reviewby),'eligible_nonsealed_reviewed_labels':len(valid_reviews),'historical_analysis':{'records_imported':analysis.get('records_imported'),'records_quarantined':analysis.get('records_quarantined')},'rejection_reason_counts':dict(reject_counts),'verified_vs_inferred':'Hash/IDs and artifact contents verified. Historical training execution cannot be replay-proven; encoder revision, source-code-at-training hashes and independent label correctness not recorded.'}
dump('source_provenance.json',provenance); dump('integrity_checks.json',artifactintegrity)
table('label_distribution.csv',dist,['source_file','split','target_field','class_label','count','labelled_total','proportion','rare_lt_5','status'])
table('language_coverage.csv',lang,['source_file','split','language','observed_script','target_field','class_label','count','independent_scenario_groups','independent_human_reviewed_count','coverage_status'])
table('suspicious_labels.csv',sus,['record_id','source_file','split','training_overlap','language','script','original_utterance','target_field','current_label','issue_type','severity','evidence','suggested_review_action','review_status'])
table('context_quality.csv',context,['record_id','source_file','split','issue_type','evidence','severity'])
table('duplicate_and_leakage_report.csv',dup,['source_file','group_id','match_type','record_id','original_id','split','training_overlap','cross_split','label_conflict','context_differs','representative_utterance','evidence'])
dump('analysis_summary.json',{'datasets':summary,'review_decisions':dict(review_counts),'reviewed_rows':len(reviews),'eligible_nonsealed_labels':len(valid_reviews),'training_records_with_later_review':sum(r.get('original_id') in tids and r['id'] in reviewby for r in data[hist]),'training_records_with_eligible_later_review':sum(byhist[x['record_id']].get('original_id') in tids for x in valid_reviews),'suspect_types':dict(collections.Counter(x['issue_type'] for x in sus)),'unique_suspicious_records':len({(x['source_file'],x['record_id']) for x in sus}),'artifact_integrity':artifactintegrity})
dump('audit_manifest.json',{'audit_script_version':'1.0','timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat(),'python':sys.version,'read_only_scope':'Only writes in report directory; no inference/training/network/model downloads','normalization':'NFC, casefold, whitespace collapse; original text preserved; no transliteration, punctuation stripping or translation','near_duplicate_triage':'SequenceMatcher >=0.85, same observed script, cross split, length>=12; not ground truth','inputs':inputs,'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='audit_manifest.json'},'manifest_self_hash':'Excluded to avoid recursive hash; regenerate output hash list after report finalization'})
print(json.dumps({'summary':summary,'review_decisions':dict(review_counts),'eligible_nonsealed':len(valid_reviews),'types':dict(collections.Counter(x['issue_type'] for x in sus)),'integrity':{k:v for k,v in artifactintegrity.items() if k not in ('metadata_equals_joblib','heads')}},indent=2))
