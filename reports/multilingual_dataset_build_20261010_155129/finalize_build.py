"""Finalize candidate-only splits, blinded compact packet, tests and manifests."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import collections,copy,csv,datetime,hashlib,io,json,subprocess,unittest
from dataset_tools import *
OUT=Path(__file__).resolve().parent
def js(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
def jl(name): return [json.loads(l) for l in (OUT/name).read_text(encoding='utf-8').splitlines() if l.strip()]
def dump(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def jsonl(name,rs): (OUT/name).write_text(''.join(canonical(r)+'\n' for r in rs),encoding='utf-8')
def table(name,rs,fields):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fields,extrasaction='ignore'); w.writeheader(); w.writerows(rs)
def rowscsv(name): return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
manifest=js('dataset_manifest.json'); pool=jl('pending_review.jsonl')+jl('synthetic_unverified.jsonl'); groups=collections.defaultdict(list)
for r in pool: groups[r['group_id']].append(r)
train_old=set(js('split_manifest.json')['old_overlap_ids']); protected=set(js('protected_test_manifest.json')['original_protected_record_ids'])
# Provisional candidate partition only. Existing model's training IDs cannot enter this candidate validation.
train_groups={g for g,rs in groups.items() if any(r['original_id'] in train_old for r in rs)}
other=sorted(set(groups)-train_groups,key=lambda g:fingerprint({'seed':42,'group':g}))
validation_group_count=min(len(other),max(1,round(.2*len(groups)))) if other else 0
validation_groups=set(other[:validation_group_count]); train_groups.update(other[validation_group_count:])
candidates_train=[]; candidates_validation=[]
for r in sorted(pool,key=lambda r:r['id']):
 derived=copy.deepcopy(r); derived['split']='train' if r['group_id'] in train_groups else 'dev'; derived['candidate_split_role']='PROVISIONAL_UNVERIFIED_NOT_SUPERVISED'
 (candidates_train if derived['split']=='train' else candidates_validation).append(derived)
jsonl('candidate_train.jsonl',candidates_train); jsonl('candidate_validation.jsonl',candidates_validation)
jsonl('schema_failures.jsonl',jl('rejected_records.jsonl')); jsonl('adjudication_ledger.jsonl',[])
split=js('split_manifest.json')
split['candidate_split_assignments']={'candidate_train':[r['id'] for r in candidates_train],'candidate_validation':[r['id'] for r in candidates_validation],'candidate_test':[]}
split['candidate_group_assignments']={g:('candidate_train' if g in train_groups else 'candidate_validation') for g in groups}
split['candidate_assignment_method']='Seed 42 SHA-256 group ordering; all original model-training-overlap groups pinned to candidate_train; approximately 20% of groups assigned validation from the remaining groups. No test assignment.'
split['candidate_supervised_eligibility']=False
split['candidate_split_file_hashes']={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ('candidate_train.jsonl','candidate_validation.jsonl')}
dump('split_manifest.json',split)
split_integrity=js('split_integrity_report.json')
split_integrity.update({'candidate_train_records':len(candidates_train),'candidate_validation_records':len(candidates_validation),'candidate_train_groups':len(train_groups),'candidate_validation_groups':len(validation_groups),'candidate_group_crossings':len(train_groups&validation_groups),'candidate_id_overlap':len({r['id'] for r in candidates_train}&{r['id'] for r in candidates_validation}),'candidate_normalized_text_overlap':len({normalized(r['utterance']) for r in candidates_train}&{normalized(r['utterance']) for r in candidates_validation}),'old_training_id_overlap_in_candidate_validation':sum(r['original_id'] in train_old for r in candidates_validation),'protected_id_overlap_in_candidate_files':sum(r['original_id'] in protected for r in pool),'candidate_splits_are_gold':False,'final_test_inference':'NOT_READY; no new gold/test sample; provisional validation is not independently reviewed accuracy evidence'})
assert not any(split_integrity[x] for x in ('candidate_group_crossings','candidate_id_overlap','candidate_normalized_text_overlap','old_training_id_overlap_in_candidate_validation','protected_id_overlap_in_candidate_files'))
dump('split_integrity_report.json',split_integrity)

# Blinded compact review: inputs only, no original targets, decisions, groups or source notes.
reviewer=OUT/'reviewer'; reviewer.mkdir(exist_ok=True)
blind=[]; blind_mapping=[]
for i,r in enumerate(sorted(pool,key=lambda r:fingerprint({'blind_order_seed':42,'id':r['id']})),1):
 rid=f'compact-blind-{i:04d}'
 record={'review_id':rid,'utterance':r['utterance'],'context':r['context'],'capabilities':r['capabilities'],'language':r['language_metadata'],'review_requirements':{'contract':'ClassifierDecision','qualified_independent_humans':2,'required_languages':r['language_metadata']['required_languages'],'required_scripts':r['language_metadata']['observed_scripts'],'transliteration_expertise_required':r['language_metadata']['transliteration_expertise_required']},'input_sha256':review_input_hash(r),'judgments':[],'adjudication':None}
 blind.append(record); blind_mapping.append({'review_id':rid,'record_id':r['id'],'original_id':r['original_id'],'group_id':r['group_id'],'input_sha256':record['input_sha256'],'original_annotations':'in candidate source_lineage; never disclose before blind initial review'})
jsonl('reviewer/records.jsonl',blind); jsonl('reviewer_mapping.jsonl',blind_mapping)
fields=sorted(pool[0]['expected']) if pool else ['intent','execution_path','request_modes','selected_capability','context_references','missing_information','needs_clarification','needs_escalation']
template={'review_id':None,'judgment':'pending','review_kind':None,'reviewer_type':None,'reviewer_id':None,'prediction_blinded':None,'input_sha256':None,'contract':'ClassifierDecision','label':None,'qualification':{'languages':[],'scripts':[],'transliteration':None},'field_rationale':dict.fromkeys(fields,''),'timestamp':None,'notes':''}
dump('reviewer/judgment-template.json',template)
schema={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Compact independent annotation v1; schema shape is not verification','type':'object','required':list(template),'additionalProperties':False,'properties':{'review_id':{'type':'string'},'judgment':{'enum':['pending','labeled','ambiguous','unresolved','rejected']},'review_kind':{'enum':['independent','self_review','historical_reference',None]},'reviewer_type':{'enum':['human','AI',None]},'reviewer_id':{'type':['string','null']},'prediction_blinded':{'type':['boolean','null']},'input_sha256':{'type':['string','null']},'contract':{'const':'ClassifierDecision'},'label':{'anyOf':[{'type':'null'},{'type':'object','required':fields,'additionalProperties':False,'properties':{'intent':{'type':'string','minLength':1},'execution_path':{'enum':sorted(PATHS)},'request_modes':{'type':'array','minItems':1,'uniqueItems':True,'items':{'enum':['conversational','memory','research','action']}},'selected_capability':{'type':['string','null']},'context_references':{'type':'array','items':{'type':'object','required':['reference','kind','resolved_value'],'additionalProperties':False,'properties':{'reference':{'type':'string','minLength':1},'kind':{'type':'string','minLength':1},'resolved_value':{'type':['string','null']}}}},'missing_information':{'type':'array','items':{'type':'object','required':['name','description'],'additionalProperties':False,'properties':{'name':{'type':'string','minLength':1},'description':{'type':'string'}}}},'needs_clarification':{'type':'boolean'},'needs_escalation':{'type':'boolean'}}}]},'qualification':{'type':'object','required':['languages','scripts','transliteration'],'properties':{'languages':{'type':'array','items':{'type':'string'}},'scripts':{'type':'array','items':{'type':'string'}},'transliteration':{'type':['boolean','null']}}},'field_rationale':{'type':'object','required':fields,'additionalProperties':False,'properties':{k:{'type':'string'} for k in fields}},'timestamp':{'type':['string','null'],'format':'date-time'},'notes':{'type':'string'}},'allOf':[{'if':{'properties':{'judgment':{'const':'labeled'}}},'then':{'properties':{'label':{'type':'object'}}},'else':{'properties':{'label':{'type':'null'}}}}]}
dump('reviewer/annotation.schema.json',schema)
instructions='''# Compact contract independent review

Share only this reviewer directory. Other build files contain historical labels and reconciliation proposals; do not show them before independent initial judgments. The old Golden Review seven-field taxonomy is a different task.

Review exact utterance, supplied context and advertised capability descriptors. Never add an imagined active_task, memory, file, URL, query, location or referent. Each reviewer records qualifications for the actual language/script/transliteration. No reviewer identity has been confirmed; templates do not represent actual approvals.

Label all eight compact training targets using annotation.schema.json. Modes are conversational/memory/research/action. Supported semantic execution paths are direct_response, memory_recall, research, single_action, multi_step_task, clarify, escalate. Intent/reference kind/missing-information name are nonempty strings in the actual older contract, not the newer family enums. A selected capability must be advertised. Capability family is not a tool name. Clarification needs a concrete missing name; unsupported action needs escalation according to actual available capabilities. The Registry's empty required list does not eliminate semantic location/time/reference prerequisites.

Preserve reference spans and resolved values faithfully; do not replace them to imitate the current adapter's first-scalar output. Explain each target field and leave the label null for pending/ambiguous/unresolved/rejected judgments. Partial labels do not qualify for gold. No model predictions may determine the label.

Save each reviewer submission as a new file with exact input_sha256, timezone timestamp, identity, qualifications and blinded independent status. Require two independent qualified initial reviewers for this build policy. Consequential disagreements and multilingual semantic judgments require a separate qualified adjudicator after initial reviews are preserved. Append new adjudication records; never rewrite earlier ledger entries. Consistent labels alone do not authenticate identities/competence.

No verified rows will be promoted until an approved reconciliation step validates the exact input AND target hashes, original contract, all eight rationales and provenance. Current candidate splits are provisional; group membership may need conservative enlargement after semantic/translation review. New final-test examples must be independently reviewed and locked before tuning; none are supplied here.
'''
(reviewer/'reviewer-instructions.md').write_text(instructions,encoding='utf-8')
dump('reviewer/input-hashes.json',{r['review_id']:r['input_sha256'] for r in blind})
native=[r for r in blind if r['review_requirements']['required_languages']!=['english']]
jsonl('native_speaker_review.jsonl',native)
table('native_speaker_review_queue.csv',[{'review_id':r['review_id'],'languages':'+'.join(r['review_requirements']['required_languages']),'scripts':'+'.join(r['review_requirements']['required_scripts']),'transliteration_required':r['review_requirements']['transliteration_expertise_required'],'status':'AWAITING_QUALIFIED_HUMANS'} for r in native],['review_id','languages','scripts','transliteration_required','status'])

# Model-output limitations are evidence, not automatic edits to genuine targets.
from assistant.orchestration.compact_adapter import _first_context_value
reference_rows=[]
for r in pool:
 issues=[]
 for ref in r['expected']['context_references']:
  if ref.get('resolved_value') is not None and _first_context_value(r['context'])!=ref['resolved_value']: issues.append('first_scalar_resolver_differs_from_annotated_value')
  if ref['reference']!=r['utterance']: issues.append('adapter_emits_full_utterance_not_gold_span')
 for missing in r['expected']['missing_information']:
  if missing['description']!='Required information is unresolved': issues.append('adapter_fixed_description_differs')
 if issues: reference_rows.append({'record_id':r['id'],'original_id':r['original_id'],'issues':'; '.join(issues),'targets_preserved':True,'runtime_fix_status':'PROPOSAL_ONLY_NOT_APPLIED','review_action':'Review semantic gold and approve runtime/reference policy; do not force annotation to wrong deterministic output.'})
table('context_reference_alignment_report.csv',reference_rows,['record_id','original_id','issues','targets_preserved','runtime_fix_status','review_action'])

# Add provisional split distributions without confusing unverified rows with gold.
existing=rowscsv('class_coverage_by_split.csv'); original_classes=collections.defaultdict(set)
for r in existing: original_classes[r['target_field']].add(r['class_label'])
existing=[r for r in existing if r['split'] not in ('candidate_train','candidate_validation')]
for split_name,bucket in [('candidate_train',candidates_train),('candidate_validation',candidates_validation)]:
 for field,classes in original_classes.items():
  values=[v for r in bucket for v in (compact_targets(r)[field] if isinstance(compact_targets(r)[field],list) else [compact_targets(r)[field]])]
  for label in sorted(classes):
   matching=[r for r in bucket if label in list(map(str,compact_targets(r)[field] if isinstance(compact_targets(r)[field],list) else [compact_targets(r)[field]]))]
   count=sum(str(v)==label for v in values)
   head_class=next(r['class_in_existing_head'] for r in existing if r['target_field']==field and r['class_label']==label)
   existing.append({'split':split_name,'target_field':field,'class_label':label,'count':count,'denominator':len(values),'proportion':count/len(values) if values else 0,'scenario_groups':len({r['group_id'] for r in matching}),'label_status':'UNVERIFIED_CANDIDATE','coverage_status':'ABSENT' if not count else 'UNVERIFIED','class_in_existing_head':head_class})
table('class_coverage_by_split.csv',existing,['split','target_field','class_label','count','denominator','proportion','scenario_groups','label_status','coverage_status','class_in_existing_head'])
languages=sorted({(r['language'],'+'.join(r['language_metadata']['observed_scripts'])) for r in pool}); language_rows=[]
for split_name,bucket in [('train',[]),('validation',[]),('test',[]),('candidate_unverified',pool),('candidate_train',candidates_train),('candidate_validation',candidates_validation)]:
 for lg,sc in languages:
  subset=[r for r in bucket if r['language']==lg and '+'.join(r['language_metadata']['observed_scripts'])==sc]
  language_rows.append({'split':split_name,'language':lg,'script':sc,'target_field':'ALL_RECORDS','class_label':'','records':len(subset),'split_denominator':len(bucket),'proportion':len(subset)/len(bucket) if bucket else 0,'scenario_groups':len({r['group_id'] for r in subset}),'verified_gold_records':0,'status':'UNVERIFIED' if bucket else 'EMPTY_NOT_READY'})
  if not bucket: continue
  for field,classes in original_classes.items():
   for label in sorted(classes):
    vals=[v for r in subset for v in (compact_targets(r)[field] if isinstance(compact_targets(r)[field],list) else [compact_targets(r)[field]])]
    matching=[r for r in subset if label in list(map(str,compact_targets(r)[field] if isinstance(compact_targets(r)[field],list) else [compact_targets(r)[field]]))]
    count=sum(str(v)==label for v in vals)
    language_rows.append({'split':split_name,'language':lg,'script':sc,'target_field':field,'class_label':label,'records':count,'split_denominator':len(vals),'proportion':count/len(vals) if vals else 0,'scenario_groups':len({r['group_id'] for r in matching}),'verified_gold_records':0,'status':'ABSENT' if not count else 'UNVERIFIED'})
table('language_script_coverage.csv',language_rows,['split','language','script','target_field','class_label','records','split_denominator','proportion','scenario_groups','verified_gold_records','status'])

# Collection task list, never generated examples or invented target labels.
collection=[]
for field,classes in original_classes.items():
 for label in sorted(classes):
  supports=[r for r in existing if r['split']=='candidate_unverified' and r['target_field']==field and r['class_label']==label]
  support=int(supports[0]['count']); scenario=int(supports[0]['scenario_groups'])
  collection.append({'target_field':field,'requested_class':label,'verified_support':0,'candidate_support':support,'candidate_groups':scenario,'priority':'CRITICAL' if support==0 else 'HIGH','action':'Collect independent real/consented or expert-authored requests; obtain exact compact-schema human approvals in every relevant language/script; do not oversample templates.','status':'NOT_READY'})
for name in registry_specs():
 positive=sum(r['expected']['selected_capability']==name for r in pool)
 collection.append({'target_field':'capability_matcher_positive_domain','requested_class':name,'verified_support':0,'candidate_support':positive,'candidate_groups':len({r['group_id'] for r in pool if r['expected']['selected_capability']==name}),'priority':'CRITICAL' if positive==0 else 'HIGH','action':'Collect matched AND unrelated advertised-capability examples. Tool names are domain coverage, not separate matcher classes.','status':'NOT_READY'})
table('data_collection_plan.csv',collection,['target_field','requested_class','verified_support','candidate_support','candidate_groups','priority','action','status'])

# Focused offline regression tests; write only inside the new build directory.
import test_dataset_tools
test_stream=io.StringIO(); result=unittest.TextTestRunner(stream=test_stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_dataset_tools))
(OUT/'test_output.txt').write_text(test_stream.getvalue(),encoding='utf-8')
test_report={'framework':'stdlib unittest','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'success':result.wasSuccessful(),'fixtures':'Synthetic unit fixtures only; never included in datasets or human approval counts','training_main_executed':False,'encoder_or_provider_called':False}
dump('test_results.json',test_report)
if not result.wasSuccessful(): raise RuntimeError('Focused regression tests failed')
# Validate actual loader on the generated candidate split files, never the sealed source.
from benchmarks.fast_classifier.evaluate import load_dataset
candidate_parses={name:len(load_dataset(OUT/name).examples) if (OUT/name).stat().st_size else 0 for name in ('candidate_train.jsonl','candidate_validation.jsonl')}
schema_validation=js('schema_validation_report.json'); schema_validation.update({'candidate_split_loader_counts':candidate_parses,'regression_tests':test_report,'verified_training_guard':'rejects all unverified rows and empty files before any encoder creation','schema_failure_count':0})
dump('schema_validation_report.json',schema_validation)
integrity=js('integrity_checks.json'); source_hashes=manifest['source_hashes']; changed=[p for p,x in source_hashes.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=x['sha256']]
integrity.update({'original_inputs_unchanged':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==x['sha256'] for p,x in source_hashes.items()},'input_files_checked':len(source_hashes),'changed_original_files':changed,'tests':test_report,'golden_original_labels_not_read':'Golden steward annotation/prediction/outcome files hashed only; frozen manifest metadata read for source IDs','no_sealed_targets_in_derived_jsonl':all(r['original_id'] not in protected for r in pool),'current_source_patches_applied':False,'new_trainer_added_but_not_executed':True})
assert not changed
dump('integrity_checks.json',integrity)
manifest['counts'].update({'candidate_train':len(candidates_train),'candidate_validation':len(candidates_validation),'candidate_train_groups':len(train_groups),'candidate_validation_groups':len(validation_groups),'native_speaker_review_queue':len(native),'schema_failures':0,'protected_downloads_duplicate_records':1,'retained_downloads_candidate_records':1,'regrouping_edges':len(jl('reviewer_mapping.jsonl')) and len(rowscsv('duplicate_group_report.csv'))})
manifest['verification_policy']='Exact compact contract, two distinct independently qualified blinded reviewers; hash-bound input/targets and full-field rationales; separate adjudicator for disagreements. No reviewers confirmed.'
manifest['candidate_files_not_gold']=['candidate_train.jsonl','candidate_validation.jsonl','pending_review.jsonl','synthetic_unverified.jsonl']
counts=manifest['counts']; recovery=rowscsv('quarantined_recovery_report.csv'); edges=rowscsv('duplicate_group_report.csv'); near=sum(x['reason']=='possible_paraphrase' for x in edges)
lang_totals=[r for r in language_rows if r['split']=='candidate_unverified' and r['target_field']=='ALL_RECORDS']
lang_table='| Language tag | Observed script | Candidates | Share of 290 | Gold |\n|---|---|---:|---:|---:|\n'+''.join(f"| {r['language']} | {r['script']} | {r['records']} | {r['proportion']:.1%} | 0 |\n" for r in lang_totals)
path_table='| Execution path | Candidate TRAIN | Candidate VALIDATION | Verified TRAIN |\n|---|---:|---:|---:|\n'
for cl in sorted(PATHS):
 ns=[next(int(r['count']) for r in existing if r['split']==sp and r['target_field']=='execution_path' and r['class_label']==cl) for sp in ('candidate_train','candidate_validation')]
 path_table+=f'| {cl} | {ns[0]} | {ns[1]} | 0 |\n'
report=f'''# Compact multilingual dataset build v1

Verdict: **BLOCKED_PENDING_HUMAN_REVIEW**. A structurally valid, versioned candidate dataset and corrected isolated build/feature implementations are complete. There are **zero verified gold records**, **zero eligible supervised fitting records**, and **zero new final evaluation records**. Do not train on the candidate files or treat their provisional splits as independent accuracy evidence.

All outputs are in this new timestamped directory. Existing datasets, Golden Review bundle, human ledger, audit report, frozen heads, production source and pre-existing user changes are unchanged. No retraining, inference, encoder loading, network/provider calls or production integration occurred. {len(source_hashes)} input files were fingerprinted and rechecked.

## Exact counts and distinct states

| State/category | Count | Interpretation |
|---|---:|---|
| verified_gold | 0 | No approved exact ClassifierDecision references |
| verified_gold_needs_adjudication | 0 | No submitted exact-contract approvals to adjudicate; potential issues remain in review queue |
| pending_review | {counts['pending_review']} | Historical decision/reconciliation evidence; not approved compact targets |
| ambiguous | 0 | No confirmed human ambiguous judgment; this does not establish absence of ambiguity |
| rejected/schema failures | 0 | All retained nonsealed candidates pass real parser; old builder rejections were recoverable structure errors |
| synthetic_unverified | {counts['synthetic_unverified']} | Authored/synthetic candidates with unverified exact targets |
| canonical nonsealed candidates | {counts['canonical_nonsealed_candidates']} | Stable original-ID records; multiple revisions retained as provenance, not examples |
| candidate scenario groups | {counts['candidate_scenario_groups']} | Conservative connected components; independent semantic status unverified |
| original versions collapsed | {counts['older_versions_collapsed']} | Historical source revisions are not counted as independent examples |
| provisional candidate TRAIN | {counts['candidate_train']} | {counts['candidate_train_groups']} groups; contains unverified labels only |
| provisional candidate VALIDATION | {counts['candidate_validation']} | {counts['candidate_validation_groups']} groups; no original model training IDs |
| new final TEST | 0 | Not collected, not reviewed and not locked |
| quarantine entries | 75 | All accounted for |
| recovered nonsealed candidates | {counts['quarantine_recovered_nonsealed']} | Retained within the 290, never counted again as new gold |
| protected quarantine entries | {counts['quarantine_protected_metadata_only']} | Metadata-only; sealed target payloads not copied or used |
| protected original-test IDs | {counts['original_test_unique_ids']} | Across v1/v2/v3/v4; excluded from candidate fitting pools |
| native-speaker review queue | {counts['native_speaker_review_queue']} | Inputs only; identities/expertise still unconfirmed |

The pool has 290 rather than all 294 v4 train/dev records because **four records have original v1 sealed-test lineage**, even though later versions moved them to development. Those historical test IDs remain protected. In all, 452 distinct historical source IDs resolve to 290 nonsealed candidates plus 162 protected IDs. All 75 quarantined records resolve to 52 recoverable nonsealed candidates plus 23 protected references. No sealed targets were exposed in build reports or copied into candidate records.

## Provenance and Golden Review reconciliation

Inventory covers v1/v2/v3/v4 compact sources, consolidation dataset/rejections/ledger, original builders and schemas, model metadata/joblib hash, audit evidence and complete Golden Review file hashes. `source_provenance.json` records source hashes, rows/IDs, split/status/provenance metadata and the working-tree modifications recorded before the build. `git_status_before.txt` preserves pre-existing user state.

The Golden bundle has 359 blinded records and **zero judgments/adjudications**. Its input hashes match the frozen steward manifest. Steward prediction, comparison and label files were hashed only, not used to choose labels. The historical ledger has 31 decisions (15 accept, 3 correct, 13 skip) for the newer FastBrainDecision taxonomy. Original identity, timestamps, label, notes and decisions are retained for nonprotected candidates. Protected decisions retain metadata and source/hash pointers without their labels.

No newer family/path review is silently transformed into complete compact gold. Only partial semantic proposals (respond→direct_response, memory→memory_recall, clarify→clarify; clarification booleans/names) are recorded with rationale. qwen_agent/fast_path cannot determine semantic compact paths; research can need one/multiple steps; escalation policy is broader than unavailable action; family is not selected tool; uncertainty is not compact confidence; requirements are not resolved references. Remaining targets require exact-schema human review. `adjudication_proposals.jsonl` contains AI-authored pending proposals, explicitly not decisions. `adjudication_ledger.jsonl` is empty and reserved for append-only approved human adjudication. No historical decision was rewritten.

`pending_review.jsonl` and `synthetic_unverified.jsonl` preserve exact utterance, original context and capabilities, original annotation, all nonsealed source revisions, original IDs/group/split, review evidence and current candidate state. Loader annotation_status remains a valid original enum; dataset_state separately expresses the six verification states. Derived files use schema_version 3.0 plus dataset_version {VERSION}; the old loader accepts these provenance additions. None is labeled verified just because shape validates. No translation or canonical text was invented; available source metadata is retained separately and absent forms remain null.

## Corrected implementation and remaining boundaries

1. **Missing-information conversion:** new `extract_missing` reads documented name/description fields and preserves them. The compact contract allows arbitrary nonempty missing names, so prior_request remains prior_request. FastBrain conversion is separately strict and diagnoses unknown enum values instead of stringifying objects or silently aliasing prior_request to reference. All 75 prior quarantine entries are accounted for; none becomes gold automatically.
2. **Capability omission:** explicit mappings cover all 19 registered actions, including open_folder/recent_files→file and grep_code/run_project_tests→project. Family metadata is a proposal distinct from execution path/intent/tool selection, not a new classifier class.
3. **Descriptor mismatch:** `exact_pair_features` serializes each example's own descriptor in original order and computes exactly the same request/capability/absolute-difference/product vector as inference. Same name/different description variants cannot overwrite one another. New trainer uses this branch by default with independent encoding. Focused tests establish structural parity with deterministic fake vectors; no BERT features were computed.
4. **Context/reference:** full original context/capabilities are preserved. Annotated reference values/spans and missing descriptions are not replaced to accommodate wrong current outputs. A pure tested resolver proposal requires a unique explicit inference-time resolved_references entry and returns unresolved for absent/ambiguous context; it never traverses the first arbitrary scalar or reads gold targets. The existing adapter remains unchanged. {len(reference_rows)} candidates have deterministic output-alignment concerns documented separately; full-decision readiness also needs approved runtime/span/description policy work.
5. **Registry arguments:** current Registry is extracted statically without loading handlers. Derived copies align weather required=[]; search_files required=[keyword]; list_directory/run_project_tests required=[]. Original descriptors remain preserved. Empty Registry required lists do not imply that weather is semantically executable without a city/coordinates/location source. Such semantic prerequisites require review, not fabricated context.

Implemented files are new and isolated; the existing train_candidate.py remains unsafe for pending candidates. `train_verified_candidate.py` is a new, unexecuted entry point: it checks exact-contract gold status, two independent qualified reviewers, input/target hashes and class fitability before encoder loading; it preserves per-example descriptors and refuses existing model destinations. `proposed_training_patch.diff` and `proposed_reference_patch.diff` are unapplied integration proposals. No application/production file was changed. See approval_boundaries.md.

## Coverage and multilingual preservation

{lang_table}
All ten requested language/script categories are preserved, including native Telugu/Hindi/Tamil, Latin forms and English–Telugu/Hindi/Tamil mixing. Original language tags are retained; observed Unicode scripts are separate (Latin-only code switching is not called mixed script). Labels, spelling, naturalness, semantic translations and transliteration equivalents remain unverified. The English queue also needs independent review; multilingual/native/Latin expertise is explicitly requested in the blinded packet.

{path_table}
`class_coverage_by_split.csv` covers path, four modes, clarification/escalation, missing names, reference kind/resolved and binary capability matching. Gold train/validation/test rows explicitly have support zero. Candidate distributions are tagged UNVERIFIED_CANDIDATE and include denominators and scenario groups. `language_script_coverage.csv` gives per-split language/script and head/class denominators. `data_collection_plan.csv` requests missing target/file/application/url/query/location/time/scope/permission/reference cases and uncommon reference kinds plus actual Registry domain coverage. Zero-count rows are inventory, not fabricated examples or labels. Binary matcher classes are 0/1; capability names are domain coverage, not invented per-tool heads. Five groups is only a triage indicator for sparse scenarios, not a scientifically established 90% support threshold.

## Group splits and leakage checks

Source scenario IDs group paraphrases, translations, transliterations and template variants. NFC/casefold/whitespace duplicates are unioned, preserving text. Same-script >=0.85 character similarity across candidate groups causes conservative transitive union ({near} such edges). This is leakage-prevention triage, not confirmed semantic identity; wider paraphrases/translations still need humans. Candidate group connected components are frozen in the provisional manifest with stable IDs/hash. No test labels or outcomes determine grouping, mapping, selection or fitting.

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

Focused offline regression suite: **{result.testsRun} tests passed**, zero failures/errors. Covers structured missing objects/unknown FastBrain names, every Registry action family, four Registry argument discrepancies, same-name descriptor feature parity, full-context preservation, explicit/ambiguous reference behavior, no label serialization, exact gold hash/qualification gate, candidate loader acceptance, empty/unverified fit blocking, Downloads protection and transitive groups. Synthetic unit approvals are fixtures only and are never included in datasets or human-review counts. Proposed source patches compile but are not applied or end-to-end validated.

## Next approved work

No reviewers are confirmed. Per the user's direction, this safe candidate build is finished without waiting for names or requesting them again. Use only reviewer/ for blind initial review; withhold source targets/ledger/proposals until initial judgments are frozen. Obtain two qualified independent exact-contract labels and separate adjudication for meaningful disagreement/multilingual semantics. Collect fresh diverse scenarios for absent classes and undercovered scripts without duplicate oversampling. Approve runtime reference/span/description policy and integration of the isolated gold/descriptor gate. Then rebuild approved group splits and independently collect/lock the final test. Stop before any model fitting until a separate training experiment is approved.

No 90% improvement is promised or measured. This is a structurally valid candidate dataset, **not a verified trainable gold dataset**.
'''
(OUT/'dataset_build_report.md').write_text(report,encoding='utf-8')
(OUT/'experiment_readiness.md').write_text(f'''# Experiment readiness

**BLOCKED_PENDING_HUMAN_REVIEW**

290 canonical nonsealed candidate records in 34 conservatively grouped components; {len(candidates_train)} provisional candidate train and {len(candidates_validation)} provisional candidate validation. All are pending_review/synthetic_unverified. Verified supervised TRAIN/VALIDATION/TEST files are intentionally empty; zero labels are eligible for fitting and zero new final-test records are locked. Empty-file schema handling does not claim successful fitting.

{result.testsRun} offline regression tests passed. Original sources/model/user changes are unchanged. Corrected builder, pair features and gold gate exist as new isolated implementation. Existing trainer/reference adapter patches are proposals only, not applied. No training or encoder inference ran.

Needed: exact ClassifierDecision approvals by two qualified independent humans; separate adjudication for disagreements and multilingual meanings; broader semantic/group verification; diverse coverage of absent/rare classes; approved runtime reference/span/description policy; a fresh independent reviewed test locked before tuning. No reviewer identities are confirmed and this build does not wait for them or invent approvals.

Do not pass candidate_train.jsonl to the legacy trainer. New train_verified_candidate.py refuses empty/non-gold/insufficient-class data before loading the encoder and refuses overwriting existing artifacts. Any next training experiment requires separate approval; no accuracy gain is claimed.
''',encoding='utf-8')

git_after=subprocess.run(['git','status','--porcelain=v1'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
(OUT/'git_status_after.txt').write_text(git_after,encoding='utf-8')
before_lines=(OUT/'git_status_before.txt').read_text(encoding='utf-8-sig').splitlines()
integrity['git_preexisting_status_entries_preserved']=set(before_lines).issubset(set(git_after.splitlines()))
integrity['source_and_test_hashes_unchanged_after_finalization']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==x['sha256'] for p,x in source_hashes.items())
dump('integrity_checks.json',integrity)
manifest['tests']=test_report; manifest['timestamp_completed']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat()
manifest['script_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')}
file_names=sorted({p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file()}|{'FILES_CREATED.md'})
(OUT/'FILES_CREATED.md').write_text('# Files created by this build\n\nAll files listed below are new under `'+OUT.as_posix()+'`. No pre-existing file outside this directory was modified. New build scripts and reports were revised only within this directory while completing the task.\n\n'+'\n'.join('- `'+name+'`' for name in file_names)+'\n',encoding='utf-8')
manifest['files']={p.relative_to(OUT).as_posix():{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in OUT.rglob('*') if p.is_file() and p.name!='dataset_manifest.json'}
manifest['self_hash_policy']='dataset_manifest.json excluded to avoid recursive hash; all other outputs listed'
dump('dataset_manifest.json',manifest)
print(json.dumps({'verdict':manifest['readiness'],'counts':counts,'tests':test_report,'inputs_unchanged':not changed,'git_entries_preserved':integrity['git_preexisting_status_entries_preserved'],'outputs':len(manifest['files'])},indent=2))
