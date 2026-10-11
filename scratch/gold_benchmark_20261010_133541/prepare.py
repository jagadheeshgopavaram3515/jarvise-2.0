import collections,datetime as dt,difflib,hashlib,json,pathlib,re,subprocess,sys
ROOT=pathlib.Path('D:/JARVIS');WORK=pathlib.Path(__file__).resolve().parent
STAMP=WORK.name.removeprefix('gold_benchmark_');OUT=ROOT/'reports'/('fast_classification_gold_review_'+STAMP)
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from benchmarks.fast_classifier.gold_review import FIELDS,baseline_modules,input_hash,score
def load(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def jsonl(path,rows):path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
files=[p for p in ROOT.iterdir() if p.is_file()]
for name in ('assistant','benchmarks','benchmark-results','reports','tests','scratch','logs','memory_store','cache'):
 files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.stat().st_size<20000000 and WORK not in p.parents)
created={'benchmarks/fast_classifier/gold_review.py','tests/test_gold_review.py'}
before={p.relative_to(ROOT).as_posix():digest(p) for p in files if p.relative_to(ROOT).as_posix() not in created}
write(WORK/'integrity-before.json',before)
(WORK/'status-before.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=ROOT))
OUT.mkdir(exist_ok=False);(OUT/'reviewer').mkdir();(OUT/'steward').mkdir()
queue=load(ROOT/'benchmarks/fast_classifier/fast_brain_v1/review_queue.jsonl')
v1=load(ROOT/'benchmarks/fast_classifier/fast_brain_v1/dataset.jsonl')
source=load(ROOT/'benchmarks/fast_classifier/v4/dataset.jsonl')
decisions=load(ROOT/'benchmarks/fast_classifier/fast_brain_v1/review_decisions.jsonl')
rejected=load(ROOT/'benchmarks/fast_classifier/fast_brain_v1/rejected_records.jsonl')
adjudications=load(ROOT/'benchmarks/fast_classifier/fast_brain_v1/adjudication_queue.jsonl')
v1by={r['id']:r for r in v1};sourceby={r['id']:r for r in source};decisionby={r['record_id']:r for r in decisions}
assert len(queue)==len(v1)==359 and len({r['record_id'] for r in queue})==359
mods=baseline_modules();c=mods['contract']
taxonomy={cls.__name__:[v.value for v in cls] for cls in (c.IntentFamily,c.CapabilityFamily,c.ExecutionPath,c.ContextRequirement,c.MissingInformation,c.EscalationReason,c.Uncertainty)}
paths=['assistant/orchestration/fast_brain/'+x+'.py' for x in ('contract','validator','deterministic','adapter','shadow','encoder_adapter')]+['assistant/orchestration/router.py','assistant/orchestration/graph.py','assistant/orchestration/decision.py','assistant/orchestration/compact_adapter.py','benchmarks/fast_classifier/build_fast_brain_dataset.py','benchmarks/fast_classifier/review_fast_brain_dataset.py','benchmarks/fast_classifier/gold_review.py']
hashes={p:digest(ROOT/p) for p in paths}
target={'target':'DeterministicFastBrain','backend_name':'deterministic-v1','entry_point':'assistant.orchestration.fast_brain.deterministic.DeterministicFastBrain.decide(FastBrainRequest)','inputs':['transcript: exact original string','context: original source mapping','capabilities: original advertised names (no executable handlers)'],'outputs':'Frozen seven-field FastBrainDecision','checkpoint':None,'model_calls':0,'generation_settings':None,'execution':'direct decide + actual validate_decision, before safe_decide can conceal failures','production_connected':False,'not_evaluated':['Legacy QwenClassifier LOCAL/GEMINI/TASK','compact-mbert-frozen-v3 older structured adapter','raw-logit encoder foundation','historical label builder'],'source_hashes':hashes,'benchmark_status':'NOT_MEASURED: independent reference labels unavailable'}
write(OUT/'steward/evaluation-target.json',target)
taxonomy_text='# Frozen taxonomy for blinded independent review\n\nSource enums and parser: contract.py:13–87,137–160; constraints: validator.py:16–38. The meanings are operational guidance, not additional enum values.\n\n'
for cls,values in taxonomy.items():taxonomy_text+='## '+cls+'\n\n'+', '.join('`'+v+'`' for v in values)+'\n\n'
taxonomy_text+='''## All required decision fields

intent_family, execution_path, capability_family, context_requirements, clarification, uncertainty, escalation are required. clarification has required (boolean) and missing (array); escalation has candidate (boolean) and reason (enum or null). Context is a nonempty array. All enums must use exact spellings above. Gold annotations require exact keys and boolean/array types even where the runtime parser is permissive; this does not change the runtime contract.

## Cross-field constraints

none cannot coexist with another context requirement. clarification.required iff missing is nonempty. escalation.candidate iff reason is nonnull. clarify execution path and clarification intent require clarification. fast_path cannot use unknown capability. The implementation does not enforce all semantic intent/path/capability combinations; legal shape alone is insufficient.

## Operational meaning to adjudicate

conversation is dialogue; information is an explanatory/factual response. deterministic_action denotes a directly handled action, local_task a local capability task, research information gathering, memory stored-memory retrieval, multi_step_task a composed goal, clarification unresolved essential input, unsupported unavailable capability. Capability values designate domains, not tool names. file is for folder/file operations, application for launchable programs, browser for browser actions, project for project/code work. research, weather, location, reminder, memory and system name their domains; none means no capability, unknown means an unresolved domain.

Execution fast_path indicates a documented deterministic handler; respond a direct answer; qwen_agent an agent task (an enum value, not a command to run Qwen); memory/research their workflows; clarify asks for missing information; escalate flags an escalation workflow. Folder opening can use file/local_task/qwen_agent under an appropriate local-task policy; launching an app can use application. Research is distinct from simply opening a known URL; multistep research can need multi_step_task. Do not infer that any schema-valid path is executable in production.

Context requirements indicate needed context, not merely the existence of a context object. conversation/memory/project/current_information/observations/active_task identify needed sources; none excludes other entries. Empty supplied context must not automatically imply active_task. A command can depend on an unresolved referent even when context is empty. Clarification missing entries identify target/file/application/url/query/location/time/scope/permission/reference. Escalation reason names are above; actual capability availability and runtime policy must support a claim of unsupported or escalation.

## Important unresolved rubric issue

Uncertainty high/medium/low/ambiguous/outside_envelope has no calibrated thresholds in the frozen validator. The design proposes calibrated bands but does not implement their boundaries. Do not invent precision or infer the correct band from a classifier output. Reviewers/steward must agree a versioned operational rubric for uncertainty and route/context ownership before full-field labels become defensible gold. If a required field cannot be justified, leave the whole record ambiguous/unresolved; partial judgments are supporting evidence only.

No computer_control IntentFamily or local ExecutionPath exists. Do not create replacement enum values to accommodate a request. Tool names in the input capability list are supporting availability evidence, not capability_family enum values.
'''
taxonomy_text='\n'.join(line.rstrip() for line in taxonomy_text.splitlines())+'\n'
(OUT/'reviewer/taxonomy.md').write_text(taxonomy_text,encoding='utf-8');rubric_sha=digest(OUT/'reviewer/taxonomy.md')
language_codes={'en':'english','te':'telugu','hi':'hindi','ta':'tamil'}
ordered=sorted(queue,key=lambda r:hashlib.sha256(('gold-review-v1:'+r['record_id']).encode()).hexdigest())
forms=[];orig=[];mapping={};known=set()
audit=load(ROOT/'reports/fast_brain_classification_records.jsonl')
known={r['record_id'] for r in audit if r['domain_mapping_error']}
for index,row in enumerate(ordered,1):
 rid=row['record_id'];src=sourceby[v1by[rid]['original_id']];review_id=f'blind-{index:03d}'
 rawlanguage=src.get('language','en');tokens=rawlanguage.replace('mixed-','').split('-')
 languages=sorted({language_codes[x] for x in tokens if x in language_codes})
 if not languages:raise ValueError('Unknown source language requiring explicit resolution: '+rawlanguage)
 text=src['utterance'];scripts=[]
 for script,lo,hi in [('telugu',0xC00,0xC7F),('devanagari',0x900,0x97F),('tamil',0xB80,0xBFF)]:
  if any(lo<=ord(ch)<=hi for ch in text):scripts.append(script)
 if any(ch.isascii() and ch.isalpha() for ch in text):scripts.append('latin')
 if not scripts:scripts=['latin']
 form={'review_id':review_id,'utterance':text,'context':src.get('context',{}),'capabilities':src.get('capabilities',[]),
       'language':{**row['language'],'source_language_tag':rawlanguage,'observed_scripts':scripts},
       'review_requirements':{'required_languages':languages,'required_scripts':scripts,'independent_human_required':True,'qualified_language_review_required':languages!=['english'],'transliteration_expertise_required':bool('latin' in scripts and any(x!='english' for x in languages))},
       'rubric_sha256':rubric_sha,'review_status':'pending','judgments':[],'adjudication':None}
 forms.append(form);mapping[review_id]={'record_id':rid,'group_id':row['group_id'],'split':v1by[rid]['split'],'known_domain_mapping_concern':rid in known,'empty_source_context':not bool(src.get('context')),'language':row['language']}
 orig.append({'record_id':rid,'review_id':review_id,'historical_canonical_label':row['canonical_label'],'historical_legacy_label':row['original_label'],'source_expected':src['expected'],'historical_provenance':row['provenance'],'historical_review_decision':decisionby.get(rid),'not_gold':True})
for row in orig:
 if row['historical_review_decision']:
  row['historical_review_decision']=dict(row['historical_review_decision'])
  row['historical_review_decision']['reviewer_id']='historical-reviewer-'+hashlib.sha256(row['historical_review_decision']['reviewer_id'].encode()).hexdigest()[:8]
jsonl(OUT/'reviewer/records.jsonl',forms)
jsonl(OUT/'steward/original-labels.jsonl',orig)
jsonl(OUT/'steward/gold-evaluation-set.jsonl',[{'review_id':r['review_id'],'record_id':mapping[r['review_id']]['record_id'],'utterance':r['utterance'],'gold_label':None,'proposed_gold_label':None,'field_rationale':dict.fromkeys(FIELDS,None),'reviewer_id':None,'judgment_source':None,'review_status':'awaiting_independent_qualified_review','uncertainty_status':'unresolved','adjudication_notes':'No independent reviewer available. Historical/AI/self-reviewed labels not promoted.','prepared_at':dt.datetime.now(dt.timezone(dt.timedelta(hours=5,minutes=30))).isoformat()} for r in forms])
template={'judgment':'pending','review_kind':None,'reviewer_type':None,'reviewer_id':None,'prediction_blinded':None,'input_sha256':None,'rubric_sha256':rubric_sha,'qualification':{'languages':[],'scripts':[],'transliteration':None},'label':None,'field_rationale':dict.fromkeys(FIELDS,''),'notes':'','timestamp':None}
write(OUT/'reviewer/judgment-template.json',template)
label_schema={'type':'object','additionalProperties':False,'required':['intent_family','execution_path','capability_family','context_requirements','clarification','uncertainty','escalation'],'properties':{
 'intent_family':{'enum':taxonomy['IntentFamily']},'execution_path':{'enum':taxonomy['ExecutionPath']},'capability_family':{'enum':taxonomy['CapabilityFamily']},'uncertainty':{'enum':taxonomy['Uncertainty']},
 'context_requirements':{'type':'array','minItems':1,'items':{'enum':taxonomy['ContextRequirement']}},
 'clarification':{'type':'object','additionalProperties':False,'required':['required','missing'],'properties':{'required':{'type':'boolean'},'missing':{'type':'array','items':{'enum':taxonomy['MissingInformation']}}}},
 'escalation':{'type':'object','additionalProperties':False,'required':['candidate','reason'],'properties':{'candidate':{'type':'boolean'},'reason':{'enum':taxonomy['EscalationReason']+[None]}}}}}
schema={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Separate offline human annotation; not a production schema','type':'object','required':list(template),'additionalProperties':False,'properties':{
 'judgment':{'enum':['pending','labeled','ambiguous','unresolved']},'review_kind':{'enum':['independent','self_review','historical_reference',None]},'reviewer_type':{'enum':['human','AI',None]},'reviewer_id':{'type':['string','null']},'prediction_blinded':{'type':['boolean','null']},'input_sha256':{'type':['string','null']},'rubric_sha256':{'type':'string'},'qualification':{'type':'object','required':['languages','scripts','transliteration'],'properties':{'languages':{'type':'array','items':{'type':'string'}},'scripts':{'type':'array','items':{'type':'string'}},'transliteration':{'type':['boolean','null']}}},'label':{'anyOf':[label_schema,{'type':'null'}]},'field_rationale':{'type':'object','required':list(FIELDS),'additionalProperties':False,'properties':{field:{'type':'string'} for field in FIELDS}},'notes':{'type':'string'},'timestamp':{'type':['string','null'],'format':'date-time'}},'allOf':[{'if':{'properties':{'judgment':{'const':'labeled'}}},'then':{'properties':{'label':label_schema}}},{'if':{'properties':{'judgment':{'enum':['pending','ambiguous','unresolved']}}},'then':{'properties':{'label':{'type':'null'}}}}], 'description':'Actual validate_label/check_judgment additionally enforce frozen cross-field constraints, rationale, qualification and provenance; schema shape alone does not establish independent review.'}
write(OUT/'reviewer/annotation.schema.json',schema)
instructions='''# Blinded offline gold-label review

Share only this reviewer directory. Do not open the steward directory, old audit reports, previous human decisions, Qwen proposals or baseline predictions before an initial judgment. File permissions do not enforce blinding: separation is procedural. Labels are all blank; there are no machine-suggested translations.

1. Read taxonomy.md and settle a shared, versioned policy for fields whose operational ownership is unclear (especially uncertainty bands). Record that policy before initial labeling; regenerate rubric hashes if changed. Do not change production schemas.
2. Assign anonymized IDs to qualified human reviewers. English needs independent review too. For multilingual/code-switched/transliterated records, cover every required language/script and transliteration expertise. If expertise is unavailable, choose unresolved, explain the language gap and leave label null.
3. Review records.jsonl in its fixed blind order. Consider exact utterance, supplied source context and capability descriptors. Missing context is not permission to invent a referent or infer active_task. Source annotations and source notes have deliberately been withheld.
4. Copy judgment-template.json into a record's judgments array. Use judgment=labeled only for a complete defensible seven-field interpretation (a proposed correct reference label, not acceptance of a prediction). Otherwise use ambiguous or unresolved and label=null. Give a rationale for each of the nine dimensions; partial field opinions belong in notes and do not qualify for strict gold.
5. Mark review_kind=independent only if the judgment is actually independent, reviewer_type=human, prediction_blinded=true only if true, and record qualification. Self-review uses self_review and remains separate/non-gold. Record timezone-bearing timestamp, input_sha256 supplied in input-hashes.json and rubric_sha256. Competence is attested by the reviewer, not automatically authenticated by the tool.
6. Save a new annotation file such as reviews_reviewer_A.jsonl; preserve records.jsonl, previous submissions, original datasets and history. A second reviewer makes their initial judgment separately without seeing the first label. The steward then combines judgments by review_id. Duplicate IDs/identities, incomplete fields and disputed labels must not be silently merged.
7. A qualified separate adjudicator resolves disagreements only after both initial judgments are preserved. Put the full chosen label and per-field rationale in adjudication, plus notes explaining the disagreement. The adjudicator may see initial labels but must not substitute classifier agreement for evidence. Irresolvable or out-of-taxonomy meanings stay unresolved.
8. Freeze accepted independent labels, review/provenance records and hashes before revealing predictions. Validate against annotation.schema.json and the real schema/gold gate. Only then run the offline baseline command in the steward instructions. Never append these new annotations to the old review_decisions.jsonl automatically.

All pending/ambiguous/unresolved/self-reviewed/historical/AI cases are excluded from genuine accuracy. Scoring cannot verify a human's identity or linguistic skill; the steward must verify the review process. No network, model, application, desktop or microphone action is required.
'''
(OUT/'reviewer/reviewer-instructions.md').write_text(instructions,encoding='utf-8')
write(OUT/'reviewer/input-hashes.json',{r['review_id']:input_hash(r) for r in forms})
inventory={'queue_records':len(queue),'source_v4_records':len(source),'v1_records':len(v1),'excluded_during_consolidation':len(rejected),'adjudication_artifact_records':len(adjudications),'unique_ids':len(set(r['record_id'] for r in queue)),'duplicate_record_ids':dict((k,n) for k,n in collections.Counter(r['record_id'] for r in queue).items() if n>1),'historical_decisions':len(decisions),'decision_counts':dict(collections.Counter(r['reviewer_decision'] for r in decisions)),'pending':sum(r['record_id'] not in decisionby for r in queue),'historical_labels_with_per_field_review_documentation':sum(bool(r.get('field_rationale')) for r in decisions),'independent_gold_records':0,'language_script':dict(collections.Counter(r['language']['semantic_language']+'/'+r['language']['input_script'] for r in queue)),'source_language_tags':dict(collections.Counter(r['language']['source_language_tag'] for r in forms)),'intent':dict(collections.Counter(r['canonical_label']['intent_family'] for r in queue)),'capability':dict(collections.Counter(r['canonical_label']['capability_family'] for r in queue)),'execution_path':dict(collections.Counter(r['canonical_label']['execution_path'] for r in queue)),'groups':dict(collections.Counter(r['group_id'] for r in queue)),'splits':dict(collections.Counter(r['split'] for r in v1)),'provenance':dict(collections.Counter(r['provenance'] for r in queue)),'known_domain_mapping_cases':len(known),'empty_source_context_active_task':sum(not r['context'] and q['canonical_label']['context_requirements']==['active_task'] for r,q in zip(forms,ordered)),'missing_required_fields':0,'invalid_canonical_labels':0,'qualified_language_review_records':sum(r['review_requirements']['qualified_language_review_required'] for r in forms),'baseline_pattern_coverage':'No reference labels for stop/time/date/open Chrome/YouTube handlers in the 359 audit population; do not infer handler accuracy.'}
for row in queue:c.FastBrainDecision.from_mapping(row['canonical_label'])
duplicates={text:n for text,n in collections.Counter(r['text'] for r in queue).items() if n>1}
inventory['duplicate_utterances']=duplicates
near=[]
for i,a in enumerate(queue):
 for b in queue[i+1:]:
  if a['language']!=b['language']:continue
  x=re.sub(r'\s+',' ',a['text'].casefold()).strip();y=re.sub(r'\s+',' ',b['text'].casefold()).strip()
  if x!=y and difflib.SequenceMatcher(None,x,y).ratio()>=.9:near.append({'a':a['record_id'],'b':b['record_id'],'same_group':a['group_id']==b['group_id']})
inventory['lexical_near_duplicate_pairs']=len(near)
write(OUT/'steward/near-duplicate-screen.json',near)
write(OUT/'steward/inventory.json',inventory)
manifest={'selection':'all 359 queue records; census, not sampled real-world distribution','order':'sha256(gold-review-v1:record_id), ascending','rubric_sha256':rubric_sha,'review_inputs':{r['review_id']:input_hash(r) for r in forms},'mapping':mapping,'source_hashes':hashes,'original_file_hashes':{p:digest(ROOT/p) for p in ('benchmarks/fast_classifier/v4/dataset.jsonl','benchmarks/fast_classifier/fast_brain_v1/dataset.jsonl','benchmarks/fast_classifier/fast_brain_v1/review_queue.jsonl','benchmarks/fast_classifier/fast_brain_v1/review_decisions.jsonl','benchmarks/fast_classifier/fast_brain_v1/adjudication_queue.jsonl','benchmarks/fast_classifier/fast_brain_v1/rejected_records.jsonl')},'reviewer_identity_policy':'anonymized; independence/qualification require human verification','accuracy_status':'NOT_MEASURED','predictions_executed':0}
write(OUT/'steward/manifest.json',manifest)
metrics=score(forms,{})
assert metrics['independently_adjudicated']==0 and metrics['exact_match']['value'] is None
metrics.update(predictions_executed=0,qwen_used=False,confidence_intervals='NOT_MEASURED: zero independent labels; translations within 49 groups are correlated. Future finite-population descriptive scores do not support real-world generalization.',precision_recall_f1='NOT_MEASURED',per_language_script='NOT_MEASURED',known_error_vs_ordinary='NOT_MEASURED',historical_consistency='NOT_MEASURED in this gold-gated run',abstention='NOT_MEASURED',readiness='BLOCKED_INSUFFICIENT_INDEPENDENT_REFERENCE_LABELS')
write(OUT/'steward/aggregate-metrics.json',metrics)
(OUT/'steward/predictions.jsonl').write_text('',encoding='utf-8')
jsonl(OUT/'steward/per-record-comparison.jsonl',[{'record_id':mapping[r['review_id']]['record_id'],'review_id':r['review_id'],'gold_label':None,'prediction':None,'schema_validation':None,'field_correctness':dict.fromkeys(FIELDS,None),'exact_match':None,'status':'NOT_MEASURED','reason':'Independent qualified reference label not available.'} for r in forms])
import csv
with (OUT/'steward/error-analysis.csv').open('w',encoding='utf-8-sig',newline='') as handle:
 writer=csv.writer(handle);writer.writerow(['record_id','review_id','status','reason'])
 for r in forms:writer.writerow([mapping[r['review_id']]['record_id'],r['review_id'],'NOT_MEASURED','Requires independent qualified language review' if r['review_requirements']['qualified_language_review_required'] else 'Requires independent English review'])
command='python -B -m benchmarks.fast_classifier.gold_review --reviews <new-reviewed-jsonl> --manifest '+str(OUT/'steward/manifest.json')+' --out <new-benchmark-output-directory>'
(OUT/'steward/adjudication-instructions.md').write_text('# Steward workflow\n\nDo not share this directory or old reports/predictions before initial judgments. Keep original-labels.jsonl as delayed supporting evidence only. The file split is procedural, not an ACL. Verify human identities/independence and language competence; the code checks attestations and completeness only.\n\nFreeze field semantics (especially uncertainty bands), retain both initial reviews and all disagreements, and use a separate qualified adjudicator when they disagree. Preserve self-review separately. Never overwrite an original queue or decision log. Hash accepted references before revealing predictions.\n\nAfter reference labels qualify, run from D:\\JARVIS:\n\n```powershell\n'+command+'\n```\n\nThe command refuses an existing output directory, verifies input/rubric hashes, runs only eligible records, and uses the real deterministic baseline with no network/model. Failed/missing/invalid predictions count as incorrect in each eligible denominator. Ambiguous/unresolved/self/AI/historical references are excluded and listed. The initial blank bundle has N=0 and must remain NOT_MEASURED. Do not advertise one qualified single-review label as consensus; retain reviewer count and qualifications.\n\nFuture reporting: exact seven-field/nested match; nine per-dimension counts; confusion matrices for enum/boolean fields and exact-set-list diagnostics for context/missing; macro and per-class precision/recall/F1 with zero-support classes disclosed; language/script, intent/capability, known-error and ordinary subgroup sizes. A full census gives descriptive correctness on this fixed set, not sampling generalization. Correlated translations require group-aware uncertainty analyses; do not attach naive iid Wilson intervals to real-world claims. Separately report historical consistency, classifier gold accuracy and Qwen-review quality; Qwen remains out of scope.\n',encoding='utf-8')
write(WORK/'preparation-summary.json',{'bundle':str(OUT),'timestamp':STAMP,'inventory':inventory,'report':str(ROOT/'reports'/('fast_classification_gold_benchmark_'+STAMP+'.md')),'baseline_predictions_executed':0})
print(json.dumps({'bundle':str(OUT),'inventory':inventory,'gold_adjudicated':0,'accuracy':'NOT_MEASURED'},ensure_ascii=False,indent=2))
