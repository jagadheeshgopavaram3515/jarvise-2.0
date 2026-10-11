"""Reproducible offline construction of candidate-only compact dataset v1.

Original test labels are not copied or used. Outputs are limited to this new
version directory. This program never trains, predicts, or claims gold review.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,copy,csv,datetime,difflib,hashlib,json,collections,subprocess
from dataset_tools import *
OUT=Path(__file__).resolve().parent
GOLD='reports/fast_classification_gold_review_20261010_133541'
AUDIT='reports/bert_dataset_quality_audit_20261010_152955'
INPUTS={}; INVENTORY=[]
def read(path):
 p=ROOT/path; b=p.read_bytes(); INPUTS[path]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}; return b.decode('utf-8-sig')
def js(p): return json.loads(read(p))
def jl(p): return [json.loads(l) for l in read(p).splitlines() if l.strip()]
def dump(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
def jsonl(name,rs): (OUT/name).write_text(''.join(canonical(r)+'\n' for r in rs),encoding='utf-8')
def tab(name,rs,fields):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fields,extrasaction='ignore'); w.writeheader(); w.writerows(rs)

SOURCE_PATHS=['benchmarks/fast_classifier/v1/seed.jsonl']+[f'benchmarks/fast_classifier/v{i}/dataset.jsonl' for i in (2,3,4)]
source_data={}; protected=set(); lineage=collections.defaultdict(list); protected_metadata=[]
# Inspect only split/ID metadata on original test rows; never carry their target fields into a build pool.
for p in SOURCE_PATHS:
 raw=jl(p); source_data[p]=raw
 for r in raw:
  if r.get('split')=='test': protected.add(r['id'])
 for r in raw:
  INVENTORY.append({'source_file':p,'record_id':r['id'],'split':r.get('split'),'annotation_status':r.get('annotation_status'),'review_status':r.get('review_status'),'provenance':r.get('provenance'),'protected':r.get('split')=='test'})
for p,rs in source_data.items():
 for r in rs:
  if r['id'] in protected:
   if r.get('split')=='test': protected_metadata.append({'source_file':p,'record_id':r['id'],'group_id':r.get('group_id'),'split':r.get('split'),'source_sha256':INPUTS[p]['sha256']})
   continue
  lineage[r['id']].append({'source_file':p,'source_sha256':INPUTS[p]['sha256'],'source_record':copy.deepcopy(r)})
# Latest source version is the canonical candidate; multiple historical versions are not independent records.
latest={key:versions[-1]['source_record'] for key,versions in lineage.items()}
historic_path='benchmarks/fast_classifier/fast_brain_v1/dataset.jsonl'
hist=jl(historic_path); hist_by_id={r['id']:r for r in hist}; hist_by_original=collections.defaultdict(list)
for r in hist:
 hist_by_original[r['original_id']].append(r)
 INVENTORY.append({'source_file':historic_path,'record_id':r['id'],'original_id':r['original_id'],'split':r['split'],'review_status':r['review_status'],'provenance':r['provenance'],'protected':r['original_id'] in protected or r['split']=='sealed_candidate'})
ledger_path='benchmarks/fast_classifier/fast_brain_v1/review_decisions.jsonl'; ledger=jl(ledger_path)
review_by_original=collections.defaultdict(list); protected_reviews=[]
for r in ledger:
 h=hist_by_id.get(r['record_id']); original=h.get('original_id') if h else None
 if original in protected or (h and h['split']=='sealed_candidate'):
  protected_reviews.append({'record_id':r['record_id'],'original_id':original,'reviewer_decision':r.get('reviewer_decision'),'source_file':ledger_path,'source_sha256':INPUTS[ledger_path]['sha256'],'labels_withheld':True})
 elif original: review_by_original[original].append(copy.deepcopy(r))
gold_records=jl(GOLD+'/reviewer/records.jsonl'); gold_manifest=js(GOLD+'/steward/manifest.json'); gold_hashes=js(GOLD+'/reviewer/input-hashes.json')
from benchmarks.fast_classifier.gold_review import input_hash
gold_input_errors=[]; golden_by_original=collections.defaultdict(list)
for r in gold_records:
 if input_hash(r)!=gold_manifest['review_inputs'].get(r['review_id']): gold_input_errors.append(r['review_id'])
 h=hist_by_id.get(gold_manifest['mapping'][r['review_id']]['record_id']); original=h.get('original_id') if h else None
 if original and original not in protected and h['split']!='sealed_candidate': golden_by_original[original].append(copy.deepcopy(r))
 INVENTORY.append({'source_file':GOLD+'/reviewer/records.jsonl','record_id':r['review_id'],'original_id':original,'split':gold_manifest['mapping'][r['review_id']]['split'],'annotation_status':r['review_status'],'protected':original in protected or bool(h and h['split']=='sealed_candidate'),'judgment_count':len(r.get('judgments',[]))})
if gold_input_errors: raise ValueError('Golden input hashes do not match frozen mapping')
quarantine_path='benchmarks/fast_classifier/fast_brain_v1/rejected_records.jsonl'; quarantine=jl(quarantine_path)
specs=registry_specs(); read('assistant/tools/registry.py')
metadata=js('benchmarks/fast_classifier/models/v3/compact_heads.json'); train_ids=set(metadata['training_example_ids'])
# Inventory relevant evidence without opening protected labels or evaluation outcomes.
inventory_paths=[]
for base in (ROOT/GOLD,ROOT/AUDIT): inventory_paths.extend(p for p in base.rglob('*') if p.is_file())
inventory_paths.extend((ROOT/'benchmarks/fast_classifier').glob('*.py'))
inventory_paths.extend((ROOT/'benchmarks/fast_classifier/fast_brain_v1').glob('*.py'))
inventory_paths.extend((ROOT/'assistant/orchestration').glob('*adapter.py'))
inventory_paths.extend([ROOT/'assistant/orchestration/decision.py',ROOT/'assistant/orchestration/fast_brain/contract.py',ROOT/'benchmarks/fast_classifier/models/v3/compact_heads.joblib'])
for v in (2,3,4): inventory_paths.append(ROOT/f'benchmarks/fast_classifier/v{v}/split_manifest.json')
for p in inventory_paths:
 key=p.relative_to(ROOT).as_posix()
 if key not in INPUTS:
  b=p.read_bytes(); INPUTS[key]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'inspection':'hash_only' if '/steward/' in key and key!=GOLD+'/steward/manifest.json' else 'inventory'}
# Hash pre-existing tracked modifications as well as source inputs.
git_before=(OUT/'git_status_before.txt').read_text(encoding='utf-8-sig')
modified_files=[]
for line in git_before.splitlines():
 if line.startswith(' M '):
  path=line[3:]; p=ROOT/path
  if p.is_file(): INPUTS[path]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'inspection':'preexisting_user_change_hash_only'}; modified_files.append(path)

UF=UnionFind(latest); edges=[]
def union(a,b,reason,evidence):
 if a in latest and b in latest:
  UF.union(a,b); edges.append({'record_a':a,'record_b':b,'reason':reason,'evidence':evidence,'decision':'conservative_group_union_not_confirmed_semantic_duplicate'})
group_buckets=collections.defaultdict(list); text_buckets=collections.defaultdict(list)
for key,r in latest.items():
 group_buckets[r.get('group_id',key)].append(key); text_buckets[normalized(r['utterance'])].append(key)
for g,ids in group_buckets.items():
 for b in ids[1:]: union(ids[0],b,'source_scenario_translation_template',g)
for t,ids in text_buckets.items():
 for b in ids[1:]: union(ids[0],b,'normalized_duplicate','NFC/casefold/whitespace; original preserved')
# Group high-similarity candidates conservatively even across original train/dev; no split yet exists.
keys=sorted(latest)
for i,a in enumerate(keys):
 ta=normalized(latest[a]['utterance']); sa=observed_scripts(ta)
 for b in keys[i+1:]:
  if UF.root(a)==UF.root(b): continue
  tb=normalized(latest[b]['utterance'])
  if len(ta)<12 or len(tb)<12 or sa!=observed_scripts(tb): continue
  score=difflib.SequenceMatcher(None,ta,tb,autojunk=False).ratio()
  if score>=.85: union(a,b,'possible_paraphrase',f'same-script character similarity {score:.4f}; review required')
# Reuse known scenario topics as conservative family unions (never as label evidence).
# All translations stay with original source group; unresolved broader semantics go in collection plan.
pool=[]; mapping=[]; descriptor=[]; validation=[]; recovery=[]; recovery_ids={r.get('original_id') for r in quarantine}; protected_pool=[]
for old_id in keys:
 raw=latest[old_id]; r=copy.deepcopy(raw); source_path=lineage[old_id][-1]['source_file']
 r['schema_version']='3.0'; r['id']='compact-candidate-'+fingerprint({'original_id':old_id})[:20]
 r['original_id']=old_id; r['dataset_version']=VERSION; r['original_schema_version']=raw.get('schema_version')
 r['source']=source_path; r['source_record_id']=old_id; r['original_split']=raw['split']; r['split']='candidate'
 r['provenance']='derived_historical_synthetic_preserving_source'; r['source_lineage']=copy.deepcopy(lineage[old_id])
 r['original_context']=copy.deepcopy(raw['context']); r['original_capabilities']=copy.deepcopy(raw['capabilities']); r['original_annotation']=copy.deepcopy(raw['expected'])
 r['context']=copy.deepcopy(raw['context']); r['capabilities'],changes=aligned_capabilities(raw['capabilities'],specs)
 r['descriptor_changes']=changes; r['expected']['missing_information']=extract_missing(raw['expected']['missing_information'])
 r['group_id']='compact-group-'+fingerprint({'root_original_id':UF.root(old_id)})[:20]; r['original_group_id']=raw.get('group_id',old_id)
 r['annotation_status']='pending_review' if review_by_original[old_id] else 'synthetic_candidate'
 r['dataset_state']='pending_review' if review_by_original[old_id] else 'synthetic_unverified'; r['review_status']=r['dataset_state']
 r['review_history']=copy.deepcopy(review_by_original[old_id]); r['verification']={'contract':'ClassifierDecision','policy':'compact-two-reviewer-v1','reviewers':[],'status':'not_verified'}
 r['golden_review_ids']=[x['review_id'] for x in golden_by_original[old_id]]; r['golden_review_history']=golden_by_original[old_id]
 r['historical_queue_ids']=[h['id'] for h in hist_by_original[old_id]]
 r['label_sources']={'expected':'original compact annotation retained as unverified candidate','newer_review':'partial proposals only; never full-target approval'}
 lang=raw['language']; langs={'en':'english','te':'telugu','hi':'hindi','ta':'tamil'}
 required=[langs.get(lang.split('-')[0],'unknown')]
 if lang.startswith('mixed-en-'): required=['english',langs[lang.split('-')[-1]]]
 scripts=observed_scripts(raw['utterance'])
 r['language_metadata']={'source_tag':lang,'observed_scripts':scripts,'required_languages':required,'transliteration_expertise_required':lang.endswith('Latn') or (lang.startswith('mixed-') and scripts==['latin']),'verification':'unverified_source_metadata'}
 r['canonical_form']=raw.get('canonical_form'); r['translation']=raw.get('translation')
 r['capability_family_proposal']=capability_family(r['expected']['selected_capability'])
 r['recovered_quarantine_candidate']=old_id in recovery_ids
 r['protected_source']=False; r['original_test_lineage']=False; r['original_model_training_overlap']=old_id in train_ids
 # Never silently replace compact targets with newer labels.
 proposals=[]
 for review in review_by_original[old_id]:
  if review.get('reviewer_decision') in ('accept','correct') and review.get('reviewed_label'):
   proposed,notes=mapping_proposals(review['reviewed_label']); proposals.append({'source_record_id':review['record_id'],'source_review':copy.deepcopy(review),'proposed_partial_targets':proposed,'rationale':notes,'status':'pending_exact_contract_review'})
  else: proposals.append({'source_record_id':review['record_id'],'status':'not_promotable','reason':'skip/unresolved; never becomes gold'})
 r['reconciliation_proposals']=proposals
 for field in r['expected']: mapping.append({'record_id':r['id'],'original_id':old_id,'source_file':source_path,'source_field':'expected.'+field,'target_field':'expected.'+field,'source_value':canonical(raw['expected'][field]),'derived_value':canonical(r['expected'][field]),'mapping_status':'IDENTITY_UNVERIFIED' if raw['expected'][field]==r['expected'][field] else 'STRUCTURE_ONLY_UNVERIFIED','rationale':'Same compact source contract; no independent semantic verification'})
 for proposal in proposals: mapping.append({'record_id':r['id'],'original_id':old_id,'source_file':ledger_path,'source_field':'reviewed_label','target_field':'partial_only','source_value':'preserved in review_history','derived_value':canonical(proposal.get('proposed_partial_targets',{})),'mapping_status':proposal['status'],'rationale':'; '.join(proposal.get('rationale',[proposal.get('reason','')]))})
 for old,new in zip(raw['capabilities'],r['capabilities']): descriptor.append({'record_id':r['id'],'original_id':old_id,'capability':old['name'],'original_descriptor':canonical(old),'derived_descriptor':canonical(new),'registry_required':canonical(specs[old['name']]['required_arguments']),'required_arguments_aligned':new['required_arguments']==specs[old['name']]['required_arguments'],'feature_key':'full serialized per-example descriptor','status':'SOURCE_PRESERVED_DERIVED_ALIGNED_REVIEW_REQUIRED' if old!=new else 'ALIGNED'})
 try: validate_candidate(r); validation.append({'record_id':r['id'],'status':'PASS_REAL_PARSER','gold_gate':'BLOCKED_UNVERIFIED'})
 except Exception as error:
  r['dataset_state']='rejected'; r['review_status']='rejected'; r['rejection_diagnostic']=str(error); validation.append({'record_id':r['id'],'status':'FAIL','diagnostic':str(error)})
 pool.append(r)

# Old tests and historical sealed_candidates remain protected source metadata, not new final test examples.
for p,rs in source_data.items():
 for r in rs:
  if r['id'] in protected:
   protected_pool.append({'record_id':r['id'],'source_file':p,'source_sha256':INPUTS[p]['sha256'],'split':r.get('split'),'group_id':r.get('group_id'),'status':'protected_original_test_lineage','targets':'NOT_COPIED','new_test_eligibility':False})
for q in quarantine:
 original=q['original_id']; derived=next((r for r in pool if r['original_id']==original),None)
 recovery.append({'original_id':original,'source_file':q['source'],'source_line':q['line'],'previous_rejection':q['rejection_reason'],'derived_record_id':derived['id'] if derived else '',
 'recovery_status':'RECOVERED_UNVERIFIED_CANDIDATE' if derived else 'PROTECTED_ORIGINAL_TEST_CANDIDATE_METADATA_ONLY','gold_status':'NOT_VERIFIED','diagnostic':'Structured name/description extraction preserves original compact names, including prior_request; FastBrain enum incompatibility is explicit.' if derived else 'Original test labels protected; recovery input/target must be steward-reviewed separately.'})

verified=[r for r in pool if r['dataset_state']=='verified_gold']; rejected=[r for r in pool if r['dataset_state']=='rejected']; pending=[r for r in pool if r['dataset_state']=='pending_review']; synthetic=[r for r in pool if r['dataset_state']=='synthetic_unverified']; ambiguous=[r for r in pool if r['dataset_state'] in ('ambiguous','verified_gold_needs_adjudication')]
if verified: raise RuntimeError('This build has no submitted exact-schema approvals; unexpected gold promotion')
jsonl('train.jsonl',[]); jsonl('validation.jsonl',[]); jsonl('sealed_test.jsonl',[])
jsonl('pending_review.jsonl',pending); jsonl('synthetic_unverified.jsonl',synthetic); jsonl('ambiguous_records.jsonl',ambiguous); jsonl('rejected_records.jsonl',rejected)
jsonl('adjudication_proposals.jsonl',[{'record_id':r['id'],'original_id':r['original_id'],'proposal_version':'1.0','decision':'pending_review_not_human_adjudication','proposals':r['reconciliation_proposals']} for r in pool if r['reconciliation_proposals']])
jsonl('protected_source_manifest.jsonl',protected_pool)
dump('protected_test_manifest.json',{'version':VERSION,'new_final_test':{'locked':False,'independently_reviewed':False,'record_ids':[],'status':'NOT_READY_NO_VERIFIED_INDEPENDENT_RECORDS'},'original_protected_record_ids':sorted(protected),'protected_review_metadata':protected_reviews,'policy':'No original test targets copied. Old model training IDs and old 359 queue cannot be the new independent final test. Metadata-only references retained.'})
dump('source_provenance.json',{'dataset_version':VERSION,'source_files':INPUTS,'inventory':INVENTORY,'deduplication':'Canonical source ID with all nonsealed source revisions preserved; latest compact v4/v3 record wins only as an unverified candidate','protected_ids':sorted(protected),'gold_bundle':{'path':GOLD,'population':len(gold_records),'submitted_judgments':sum(len(r.get('judgments',[])) for r in gold_records),'adjudications':sum(bool(r.get('adjudication')) for r in gold_records),'input_hashes_pass':not gold_input_errors},'historical_ledger':{'rows':len(ledger),'decisions':dict(collections.Counter(r['reviewer_decision'] for r in ledger))},'preexisting_modified_files':modified_files})
tab('label_mapping_report.csv',mapping,['record_id','original_id','source_file','source_field','target_field','source_value','derived_value','mapping_status','rationale'])
tab('quarantined_recovery_report.csv',recovery,['original_id','source_file','source_line','previous_rejection','derived_record_id','recovery_status','gold_status','diagnostic'])
tab('descriptor_alignment_report.csv',descriptor,['record_id','original_id','capability','original_descriptor','derived_descriptor','registry_required','required_arguments_aligned','feature_key','status'])
tab('duplicate_group_report.csv',edges,['record_a','record_b','reason','evidence','decision'])
dump('label_mapping_manifest.json',{'version':VERSION,'compact_contract':'ClassifierDecision','compact_semantic_paths':sorted(PATHS),'request_mode_enum':['conversational','memory','research','action'],'missing_information':'Compact name is an open nonempty string; no FastBrain enum is forced into it. prior_request retained, not aliased to reference.','capability_family_proposals':CAPABILITY_FAMILIES,'safe_partial_path_proposals':{'respond':'direct_response','memory':'memory_recall','clarify':'clarify'},'unmappable_without_review':['fast_path','qwen_agent','research workflow vs single/multi-step','escalation reasons vs unsupported capability','capability_family -> exact selected_capability','uncertainty -> calibrated confidence','context_requirements -> context_references','intent_family -> exact intent/request_modes'],'approval_policy':'No conversion is promoted automatically; original labels and decisions preserved','field_mappings':len(mapping)})
groups=collections.defaultdict(list)
for r in pool: groups[r['group_id']].append(r['id'])
group_manifest={'version':VERSION,'split_status':'BLOCKED_PENDING_HUMAN_REVIEW','split_assignments':{'train':[],'validation':[],'test':[]},'candidate_groups':dict(groups),'candidate_split':'unassigned; not a supervised split','split_policy':{'unit':'audited connected scenario/translation/template group','proposed_ratios':[.7,.15,.15],'assignment':'not performed until approvals and independent new test collection','test_lock':'new reviewed groups only; old test/training IDs and old benchmark excluded'},'old_overlap_ids':sorted(train_ids),'protected_ids':sorted(protected)}
dump('split_manifest.json',group_manifest)
reviewqueue=[]
for r in pool:
 reasons=['Exact ClassifierDecision target approval missing','two independent qualified reviewers required']
 if r['recovered_quarantine_candidate']: reasons.append('recovered clarification candidate')
 if r['review_history']: reasons.append('historical newer-contract annotation needs reconciliation')
 if r['descriptor_changes']: reasons.append('Registry argument descriptor corrected in derived copy')
 if r['expected']['context_references']: reasons.append('current deterministic reference/span resolver cannot guarantee gold emission')
 reviewqueue.append({'record_id':r['id'],'original_id':r['original_id'],'state':r['dataset_state'],'group_id':r['group_id'],'language':r['language'],'script':'+'.join(r['language_metadata']['observed_scripts']),'utterance':r['utterance'],'training_overlap':r['original_model_training_overlap'],'recovered_quarantine':r['recovered_quarantine_candidate'],'review_requirements':'; '.join(reasons),'review_status':'NOT_APPROVED','source_file':r['source']})
tab('label_review_queue.csv',reviewqueue,['record_id','original_id','state','group_id','language','script','utterance','training_overlap','recovered_quarantine','review_requirements','review_status','source_file'])

# Coverage: candidate labels are explicitly unverified; absent gold rows are explicit zeroes.
actual_head_info=js(AUDIT+'/integrity_checks.json')['heads']
def fitted_classes(field):
 if field.startswith('request_modes:'): key='mode:'+field.split(':',1)[1]
 else: key={'needs_clarification':'clarification','needs_escalation':'escalation'}.get(field,field)
 return set(map(str,actual_head_info.get(key,{}).get('classes',[])))
observed=collections.defaultdict(set)
for r in pool:
 for field,label in compact_targets(r).items(): observed[field].update(map(str,label if isinstance(label,list) else [label]))
observed['missing_information'].update(x.value for x in FamilyMissing)
observed['execution_path'].update(PATHS)
for field,key in [('needs_clarification','clarification'),('needs_escalation','escalation'),('missing_information','missing_information'),('reference_kind','reference_kind'),('reference_resolved','reference_resolved'),('capability_matcher','capability_matcher')]: observed[field].update(map(str,actual_head_info[key]['classes']))
coverage=[]; langcover=[]
for split,bucket in [('train',[]),('validation',[]),('test',[]),('candidate_unverified',pool)]:
 for field,classes in sorted(observed.items()):
  for label in sorted(classes):
   matching=[]; count=denom=0
   for r in bucket:
    vals=compact_targets(r)[field]; vals=vals if isinstance(vals,list) else [vals]
    denom+=len(vals); c=sum(str(v)==label for v in vals); count+=c
    if c: matching.append(r)
   coverage.append({'split':split,'target_field':field,'class_label':label,'count':count,'denominator':denom,'proportion':count/denom if denom else 0,'scenario_groups':len({r['group_id'] for r in matching}),'label_status':'UNVERIFIED_CANDIDATE' if split=='candidate_unverified' else 'VERIFIED_GOLD_EMPTY','coverage_status':'ABSENT' if count==0 else 'UNDERREPRESENTED_GROUPS' if len({r['group_id'] for r in matching})<5 else 'UNVERIFIED','class_in_existing_head':label in fitted_classes(field)})
 for (lg,script) in sorted({(r['language'],'+'.join(r['language_metadata']['observed_scripts'])) for r in pool}):
  sub=[r for r in bucket if r['language']==lg and '+'.join(r['language_metadata']['observed_scripts'])==script]
  langcover.append({'split':split,'language':lg,'script':script,'records':len(sub),'split_denominator':len(bucket),'proportion':len(sub)/len(bucket) if bucket else 0,'scenario_groups':len({r['group_id'] for r in sub}),'verified_gold_records':0,'status':'UNVERIFIED' if split=='candidate_unverified' else 'EMPTY_NOT_READY'})
  for field,classes in observed.items():
   for label in sorted(classes):
    n=sum(sum(str(v)==label for v in (compact_targets(r)[field] if isinstance(compact_targets(r)[field],list) else [compact_targets(r)[field]])) for r in sub)
    # Parallel language/head coverage artifact with denominators; no fake records generated.
    if split=='candidate_unverified': langcover.append({'split':split,'language':lg,'script':script,'target_field':field,'class_label':label,'records':n,'split_denominator':sum(len(compact_targets(r)[field]) if isinstance(compact_targets(r)[field],list) else 1 for r in sub),'proportion':None,'scenario_groups':len({r['group_id'] for r in sub}),'verified_gold_records':0,'status':'ABSENT' if n==0 else 'UNVERIFIED'})
tab('class_coverage_by_split.csv',coverage,['split','target_field','class_label','count','denominator','proportion','scenario_groups','label_status','coverage_status','class_in_existing_head'])
tab('language_script_coverage.csv',langcover,['split','language','script','target_field','class_label','records','split_denominator','proportion','scenario_groups','verified_gold_records','status'])
dump('schema_validation_report.json',{'records':validation,'candidate_real_parser_pass':sum(x['status']=='PASS_REAL_PARSER' for x in validation),'gold_records':0,'gold_supervised_files':'EMPTY_BLOCKED; real load_dataset deliberately rejects empty selection; no fake example added','gold_gate':'enforced in new load_verified_training helper; existing trainer patch proposed only','original_context_preserved':all(r['context']==r['original_context'] for r in pool),'no_gold_fields_in_serialization':'serialize_request uses utterance/context; exact_pair_features only serializes advertised descriptors'})
integrity={'version':VERSION,'original_inputs_unchanged':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==x['sha256'] for p,x in INPUTS.items()},'original_model_training_ids':len(train_ids),'new_gold_training_ids':0,'old_test_targets_copied':False,'protected_records_copied_as_training':False,'golden_inputs_verified':not gold_input_errors,'golden_review_submissions_found':sum(len(x.get('judgments',[])) for x in gold_records),'original_source_files_modified':[],'production_routing_modified':False,'model_trained':False,'encoder_instantiated':False,'network_used':False,'original_joblib_changed':False}
dump('integrity_checks.json',integrity)
dump('split_integrity_report.json',{'status':'PASS_CANDIDATE_ISOLATION_NOT_READY_FINAL_EVALUATION','train_validation_test_group_crossings':0,'new_supervised_rows':0,'new_final_test_locked':False,'independent_final_test_status':'NOT_VERIFIED_NOT_CREATED','original_training_id_overlap_in_candidates':sum(r['original_id'] in train_ids for r in pool),'old_test_lineage_excluded_from_candidate_pools':sorted(protected),'confirmed_downloads_duplicate':{'original_ids':['train_open_downloads-en','v4_open_downloads-en'],'derived_action':'training source retained only as unverified candidate; original test record isolated in protected manifest; cannot enter new final test'},'uncertain_semantic_grouping':'Same-source translations/templates and normalized/near-text connected groups; broader semantic matches still require reviewer confirmation','candidate_group_count':len(groups),'candidate_group_hash':fingerprint(dict(groups))})
dump('dataset_manifest.json',{'version':VERSION,'readiness':'BLOCKED_PENDING_HUMAN_REVIEW','counts':{'verified_gold':len(verified),'pending_review':len(pending),'ambiguous':len(ambiguous),'rejected':len(rejected),'synthetic_unverified':len(synthetic),'eligible_for_supervised_fitting':0,'held_out_new_final_evaluation':0,'canonical_nonsealed_candidates':len(pool),'candidate_scenario_groups':len(groups),'quarantine_total':len(quarantine),'quarantine_recovered_nonsealed':sum(bool(r['recovered_quarantine_candidate']) for r in pool),'quarantine_protected_metadata_only':sum(x['recovery_status'].startswith('PROTECTED') for x in recovery),'original_test_unique_ids':len(protected),'older_versions_collapsed':sum(len(v)-1 for v in lineage.values())},'states':STATES,'source_hashes':INPUTS,'scripts':{'build_dataset.py':'1.0','dataset_tools.py':'1.0'},'files':{},'timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat(),'do_not_train':'All candidate files are non-gold; train/validation/test intentionally empty. Existing trainer is unsafe for candidates until proposed gate patch is applied.'})
dump('build_summary.json',{'counts':json.loads((OUT/'dataset_manifest.json').read_text())['counts'],'languages':[r for r in langcover if r['split']=='candidate_unverified' and 'target_field' not in r],'regression_test_status':'NOT_RUN_YET','recovery_counts':dict(collections.Counter(r['recovery_status'] for r in recovery)),'gold_bundle':'359 blank judgments; exact-schema gold unavailable','registry':specs})
print(json.dumps(json.loads((OUT/'dataset_manifest.json').read_text())['counts'],indent=2))
