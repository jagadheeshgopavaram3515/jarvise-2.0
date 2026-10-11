"""Authorized offline frozen-BERT/LR provisional-label experiment loop.
All writes stay in this new run; original annotations/artifacts remain immutable.
"""
import os,sys
sys.dont_write_bytecode=True
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
from pathlib import Path
import types,json,csv,hashlib,datetime,collections,copy,platform,traceback,time,subprocess,warnings
ROOT=Path('D:/JARVIS'); OUT=Path(__file__).resolve().parent; BUILD=ROOT/'reports/multilingual_dataset_build_20261010_155129'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BUILD))
# Isolate orchestration initialization; no production log/provider/router startup.
package=types.ModuleType('assistant.orchestration');package.__path__=[str(ROOT/'assistant/orchestration')];sys.modules['assistant.orchestration']=package
sys.modules['readline']=None
platform.uname()
INPUTS={};START=time.time()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def watch(p):
 p=Path(p).resolve();INPUTS[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size};return p
def js(p):return json.loads(watch(p).read_text(encoding='utf-8-sig'))
def jl(p):return [json.loads(l) for l in watch(p).read_text(encoding='utf-8').splitlines() if l.strip()]
def dump(n,x):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
def jsonl(n,rs):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rs),encoding='utf-8')
def table(n,rs,fields):
 with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rs)
def progress(message):print(message,flush=True);dump('progress.json',{'message':message,'elapsed_seconds':round(time.time()-START,2)})

# Audit guard blocks writes outside this run and all network/provider boundaries.
def guard(event,args):
 if event in ('socket.connect','socket.getaddrinfo','os.system'):raise RuntimeError('Blocked offline boundary: '+event)
 if event=='open':
  p,mode,flags=args;writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if writing and isinstance(p,(str,bytes,os.PathLike)) and str(p).lower() not in ('nul','\\\\.\\nul'):
   target=Path(p).resolve()
   if target!=OUT and OUT not in target.parents:raise RuntimeError('Blocked write outside run: '+str(target))
sys.addaudithook(guard)
from dataset_tools import validate_candidate,compact_targets,normalized,observed_scripts,registry_specs,exact_pair_features
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder,CompactMultilingualAdapter,serialize_request,serialize_capability,_pair_features,_first_context_value
from assistant.orchestration.decision import DecisionRequest,ClassifierDecision
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import numpy as np,joblib,sklearn,torch,transformers
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(42);np.random.seed(42)

def request(r):
 e=validate_candidate(r);return DecisionRequest(e.utterance,e.context,e.capabilities)
def target_key(x):
 y={k:copy.deepcopy(x[k]) for k in FIELDS};y['request_modes']=sorted(set(y['request_modes']));return json.dumps(y,ensure_ascii=False,sort_keys=True,separators=(',',':'))
FIELDS=('intent','execution_path','request_modes','selected_capability','context_references','missing_information','needs_clarification','needs_escalation')

manifest=js(BUILD/'dataset_manifest.json')
verified_files={n:sha(watch(BUILD/n))==v['sha256'] for n,v in manifest['files'].items()}
if not all(verified_files.values()):raise RuntimeError('Candidate output hash mismatch')
split_manifest=js(BUILD/'split_manifest.json'); protected=set(js(BUILD/'protected_test_manifest.json')['original_protected_record_ids'])
baseline_path=watch(ROOT/'benchmarks/fast_classifier/models/v3/compact_heads.joblib');baseline_meta=js(ROOT/'benchmarks/fast_classifier/models/v3/compact_heads.json')
old_train_ids=set(baseline_meta['training_example_ids'])
for p in ['assistant/orchestration/compact_adapter.py','assistant/orchestration/decision.py','benchmarks/fast_classifier/train_candidate.py','benchmarks/fast_classifier/evaluate.py','assistant/tools/registry.py']:watch(ROOT/p)
for line in (OUT/'git_status_before.txt').read_text(encoding='utf-8-sig').splitlines():
 if line.startswith(' M ') and (ROOT/line[3:]).is_file():watch(ROOT/line[3:])
train_candidates=jl(BUILD/'candidate_train.jsonl');validation=jl(BUILD/'candidate_validation.jsonl');findings=[];reviews=[];exclusions=[];train=[]
specs=registry_specs();seen={}
for r in train_candidates:
 reason=None
 if r['original_id'] in protected or r.get('original_test_lineage') or r.get('protected_source'):reason='protected_test_lineage'
 elif r['dataset_state']!='synthetic_unverified':reason='pending_reconciliation_not_authorized_as_synthetic_fitting'
 else:
  try:validate_candidate(r)
  except Exception as ex:reason='schema_invalid:'+str(ex)
 key=json.dumps({'text':normalized(r['utterance']),'context':r['context'],'capabilities':r['capabilities']},ensure_ascii=False,sort_keys=True)
 if reason is None and key in seen:
  if target_key(r['expected'])!=target_key(seen[key]['expected']):reason='contradictory_identical_request_equivalent_context'
  else:reason='redundant_duplicate'
 if reason:exclusions.append({'record_id':r['id'],'original_id':r['original_id'],'reason':reason,'state':r['dataset_state']})
 else:seen[key]=r;train.append(copy.deepcopy(r))
for r in validation:
 validate_candidate(r)
 if r['original_id'] in protected or r['original_id'] in old_train_ids:raise RuntimeError('Validation protected/training-ID overlap')
if {r['group_id'] for r in train}&{r['group_id'] for r in validation}:raise RuntimeError('Group leakage')
if {normalized(r['utterance']) for r in train}&{normalized(r['utterance']) for r in validation}:raise RuntimeError('Normalized-text leakage')
for split,records in [('provisional_train',train),('provisional_validation',validation)]:
 for r in records:
  for c in r['capabilities']:
   if c['name'] not in specs or c['required_arguments']!=specs[c['name']]['required_arguments']:raise RuntimeError('Registry descriptor drift')
  for ref in r['expected']['context_references']:
   if ref.get('resolved_value') is not None and _first_context_value(r['context'])!=ref['resolved_value']:
    findings.append({'record_id':r['id'],'split':split,'issue_type':'deterministic_reference_value_mismatch','severity':'HIGH','evidence':'Annotated value differs from existing adapter first-context-scalar choice; no relabel applied'})
   if ref['reference']!=r['utterance']:findings.append({'record_id':r['id'],'split':split,'issue_type':'deterministic_reference_span_mismatch','severity':'HIGH','evidence':'Existing adapter emits complete user utterance; gold source span preserved'})
  for m in r['expected']['missing_information']:
   if m['description']!='Required information is unresolved':findings.append({'record_id':r['id'],'split':split,'issue_type':'deterministic_missing_description_mismatch','severity':'HIGH','evidence':'Existing fixed output description differs; source description preserved'})
  if r['language'].startswith('mixed-') and observed_scripts(r['utterance'])==['latin']:findings.append({'record_id':r['id'],'split':split,'issue_type':'code_switch_latin_script_observation','severity':'INFO','evidence':'Language mixing and script mixing are distinct; observed script retained separately'})
  for f in ('missing_information','reference_kind'):
   pass
jsonl('provisional_train.jsonl',train);jsonl('provisional_validation.jsonl',validation);dump('exclusions.json',exclusions)
dump('split_integrity_report.json',{'train':len(train),'validation':len(validation),'training_groups':len({r['group_id'] for r in train}),'validation_groups':len({r['group_id'] for r in validation}),'ID_overlap':0,'group_overlap':0,'normalized_text_overlap':0,'baseline_training_ID_overlap_in_validation':0,'protected_test_overlap':0,'validation_states':dict(collections.Counter(r['dataset_state'] for r in validation)),'validation_fixed':True,'independent_semantic_group_completeness':'NOT_VERIFIED; source scenario/translations/template components plus conservative character triage','metric_type':'PROVISIONAL_LABEL_AGREEMENT_NOT_VERIFIED_ACCURACY'})
dump('training_manifest.json',{'dataset_version':manifest['version'],'authorization':'User explicitly permits structurally valid synthetic_unverified fitting for this provisional experiment only; human-gold gate is not bypassed for gold claims','train':len(train),'validation':len(validation),'train_ids':[r['id'] for r in train],'validation_ids':[r['id'] for r in validation],'exclusions':exclusions,'seed':42,'primary_metric':'exact eight-target full-record agreement, preserving reference spans/descriptions; confidence/evidence are not dataset supervised targets','encoder_frozen':True,'max_experiments':20,'patience':3,'inputs':INPUTS,'train_sha256':sha(OUT/'provisional_train.jsonl'),'validation_sha256':sha(OUT/'provisional_validation.jsonl')})
progress(f'Dataset verified: {len(train)} synthetic training / {len(validation)} fixed provisional validation; loading local encoder')
snapshot=Path.home()/'.cache/huggingface/hub/models--nlptown--bert-base-multilingual-uncased-sentiment/snapshots/8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d'
cache_files={str(p):{'sha256':sha(watch(p)),'bytes':p.stat().st_size} for p in snapshot.iterdir() if p.is_file()}
if not all((snapshot/n).is_file() for n in ('config.json','model.safetensors','vocab.txt','tokenizer_config.json')):raise RuntimeError('Local encoder/tokenizer incomplete')
encoder=FrozenMultilingualEncoder(str(snapshot),max_length=160)
for parameter in encoder.model.parameters():parameter.requires_grad_(False)
all_records=train+validation;texts=[]
for r in all_records:
 req=request(r);texts.append(serialize_request(req))
 texts.extend(serialize_capability(c.name,c.description,c.required_arguments) for c in req.capabilities)
unique_texts=list(dict.fromkeys(texts));cache_index={t:i for i,t in enumerate(unique_texts)}
vectors=encoder.encode(unique_texts,batch_size=16)
if vectors.shape!=(len(unique_texts),768) or not np.isfinite(vectors).all():raise RuntimeError('Encoder dimensions/nonfinite feature failure')
np.savez_compressed(OUT/'frozen_features.npz',vectors=vectors)
dump('feature_cache_manifest.json',{'key':'exact complete serialized input text; no name-only descriptor keys','encoder_snapshot':str(snapshot),'encoder_files':cache_files,'max_length':160,'pooling':'attention-mask mean then L2 normalization','texts':[{'text_sha256':hashlib.sha256(t.encode()).hexdigest(),'index':i} for i,t in enumerate(unique_texts)],'vectors_sha256':sha(OUT/'frozen_features.npz'),'shape':list(vectors.shape),'target_labels_in_features':False})
class CachedEncoder:
 def __init__(self):self.forward_calls=0;self.last_timings={};self.load_ms=0;self.model_id=baseline_meta['model_id'];self.max_length=160
 def encode(self,texts,**kw):
  self.forward_calls+=1;return np.stack([vectors[cache_index[t]] for t in texts])
cached=CachedEncoder()
Xtrain=np.stack([vectors[cache_index[serialize_request(request(r))]] for r in train]);Xval=np.stack([vectors[cache_index[serialize_request(request(r))]] for r in validation])
train_examples=[validate_candidate(r) for r in train];pair_train,Ypair,_=exact_pair_features(train_examples,Xtrain,cached)
val_examples=[validate_candidate(r) for r in validation];pair_val,Ypair_val,_=exact_pair_features(val_examples,Xval,cached)
parity=[]
offset=0
for i,e in enumerate(train_examples):
 for c in e.capabilities:
  direct=_pair_features(Xtrain[i],vectors[cache_index[serialize_capability(c.name,c.description,c.required_arguments)]])
  parity.append(np.array_equal(direct,pair_train[offset]));offset+=1
tokens=encoder.tokenizer(unique_texts,truncation=False,add_special_tokens=True)['input_ids'];truncated=sum(len(t)>160 for t in tokens)
dump('feature_parity_report.json',{'status':'PASS' if all(parity) else 'FAIL','local_encoder_available':True,'snapshot':str(snapshot),'request_feature_dimensions':768,'pair_feature_dimensions':3072,'request_training_shape':list(Xtrain.shape),'pair_training_shape':list(pair_train.shape),'pair_validation_shape':list(pair_val.shape),'pair_features_exactly_equal_to_inference_construction':all(parity),'finite_request_features':bool(np.isfinite(vectors).all()),'finite_pair_features':bool(np.isfinite(pair_train).all()),'encoder_eval_mode':not encoder.model.training,'encoder_parameters_require_grad':any(p.requires_grad for p in encoder.model.parameters()),'encoder_optimizer_or_backward':False,'serialized_texts_exceeding_160_tokens':truncated,'serialized_text_denominator':len(unique_texts),'cached_inference':'Exact text-index vectors replayed through unmodified adapter postprocessor; no labels/IDs/review metadata in keys','model_hashes':cache_files})
if not all(parity):raise RuntimeError('Feature parity failed; fitting prohibited')
progress(f'Offline frozen features extracted: {len(unique_texts)} unique texts, 768/3072 dimensions; running baseline')

def adapter_for(artifact):
 a=CompactMultilingualAdapter.__new__(CompactMultilingualAdapter);a.artifact=artifact;a.artifact_path='in_memory_read_only';a.encoder=CachedEncoder();a._adapter_id=artifact['adapter_id'];a.inference_count=0;a.last_inference_metrics={};return a
def cmetric(y,p,classes=None):
 y=list(map(str,y));p=list(map(str,p));labs=sorted(set(y+p+(list(map(str,classes)) if classes is not None else [])))
 rep=classification_report(y,p,labels=labs,output_dict=True,zero_division=0)
 return {'agreement':sum(a==b for a,b in zip(y,p)),'denominator':len(y),'accuracy':sum(a==b for a,b in zip(y,p))/len(y) if y else None,'label_type':'PROVISIONAL_LABEL_AGREEMENT','classes':labs,'per_class':{k:rep[k] for k in labs},'macro_avg':rep['macro avg'],'weighted_avg':rep['weighted avg'],'confusion_matrix':confusion_matrix(y,p,labels=labs).tolist()}
def evaluate(artifact):
 a=adapter_for(artifact);predictions=[];errors=[]
 for r in validation:
  try:predictions.append(dict(a.predict(request(r))));errors.append(None)
  except Exception as ex:predictions.append(None);errors.append(str(ex))
 matches=[p is not None and target_key(p)==target_key(r['expected']) for r,p in zip(validation,predictions)]
 field_metrics={}
 for field in FIELDS:
  correct=sum(p is not None and (sorted(p[field])==sorted(r['expected'][field]) if field=='request_modes' else p[field]==r['expected'][field]) for r,p in zip(validation,predictions))
  field_metrics[field]={'agreement':correct,'denominator':len(validation),'value':correct/len(validation)}
 heads={}
 for field,key in [('execution_path','execution_path'),('needs_clarification','clarification'),('needs_escalation','escalation'),('missing_information','missing_information'),('reference_kind','reference_kind'),('reference_resolved','reference_resolved')]:
  h=artifact[key];heads[field]=cmetric([compact_targets(r)[field] for r in validation],h.predict(Xval),list(h.classes_)+[compact_targets(r)[field] for r in train])
 for mode,h in artifact['request_modes'].items():heads['request_modes:'+mode]=cmetric([compact_targets(r)['request_modes:'+mode] for r in validation],h.predict(Xval),h.classes_)
 h=artifact['capability_matcher'];heads['capability_matcher']=cmetric(Ypair_val,h.predict(pair_val),h.classes_)
 for field in ('intent','execution_path','selected_capability','needs_clarification','needs_escalation'):
  heads['output:'+field]=cmetric([str(r['expected'][field]) for r in validation],[str(p[field]) if p else '__PREDICTION_FAILURE__' for p in predictions])
 per_language={}
 for lang in sorted({r['language'] for r in all_records}):
  ix=[i for i,r in enumerate(validation) if r['language']==lang];n=len(ix);k=sum(matches[i] for i in ix)
  per_language[lang]={'agreement':k,'denominator':n,'value':k/n if n else None,'language_status':'INSUFFICIENT_EVIDENCE' if n<20 else 'PROVISIONAL_ONLY_NOT_VERIFIED','independent_scenario_groups':len({validation[i]['group_id'] for i in ix}),'observed_scripts':dict(collections.Counter('+'.join(observed_scripts(validation[i]['utterance'])) for i in ix)),'execution_path_support':dict(collections.Counter(validation[i]['expected']['execution_path'] for i in ix)),'per_field':{f:{'agreement':sum(predictions[i] is not None and (sorted(predictions[i][f])==sorted(validation[i]['expected'][f]) if f=='request_modes' else predictions[i][f]==validation[i]['expected'][f]) for i in ix),'denominator':n} for f in FIELDS}}
 metric={'metric_type':'PROVISIONAL_LABEL_AGREEMENT_NOT_VERIFIED_ACCURACY','exact_full_record':{'agreement':sum(matches),'denominator':len(validation),'value':sum(matches)/len(validation)},'fields':field_metrics,'heads':heads,'languages':per_language,'prediction_failures':sum(p is None for p in predictions),'gold_verified_accuracy':{'status':'NOT_MEASURED','denominator':0},'predictions':[{'record_id':r['id'],'original_id':r['original_id'],'language':r['language'],'group_id':r['group_id'],'gold_status':r['dataset_state'],'expected':r['expected'],'prediction':p,'error':er,'exact_match':m} for r,p,er,m in zip(validation,predictions,errors,matches)]}
 return metric
baseline=joblib.load(baseline_path)
baseline_metrics=evaluate(baseline);dump('baseline_metrics.json',baseline_metrics)
# Verify cached-replay baseline predictions against one actual encoder forward.
real_adapter=adapter_for(baseline);real_adapter.encoder=encoder
real_prediction=dict(real_adapter.predict(request(validation[0])))
if target_key(real_prediction)!=target_key(baseline_metrics['predictions'][0]['prediction']):raise RuntimeError('Actual encoder vs cached baseline parity failed')
feature_report=json.loads((OUT/'feature_parity_report.json').read_text());feature_report['actual_encoder_vs_cached_adapter_prediction_parity']=True;dump('feature_parity_report.json',feature_report)
progress(f'Baseline exact provisional agreement {baseline_metrics["exact_full_record"]["agreement"]}/{len(validation)}; beginning LR search')

targets={f:np.array([compact_targets(r)[f] for r in train]) for f in compact_targets(train[0]) if f!='capability_matcher'}
missing_fit=[f for f,y in targets.items() if len(np.unique(y))<2]
if len(np.unique(Ypair))<2:missing_fit.append('capability_matcher')
if missing_fit:raise RuntimeError('Training heads lack two classes: '+repr(missing_fit))
for field,y in targets.items():
 valclasses=set(str(compact_targets(r)[field]) for r in validation);absent=valclasses-set(map(str,np.unique(y)))
 if absent:findings.append({'record_id':'','split':'provisional_validation','issue_type':'missing_training_class','severity':'HIGH','evidence':f'{field}: validation classes absent from train {sorted(absent)}; kept in denominator'})
history=[];best=None;best_artifact=None;best_name=None;best_score=-1;plateau=0
# Predeclared bounded sequence: corrected original setting, regularization, weighting, supported solver/penalty.
configs=[{'C':c,'class_weight':weight,'solver':'lbfgs','penalty':'l2','max_iter':3000,'seed':42} for weight in ('balanced',None) for c in (4.,1.,16.,.25,64.)]
configs += [{'C':c,'class_weight':weight,'solver':'liblinear','penalty':pen,'max_iter':3000,'seed':42} for pen in ('l1','l2') for weight in ('balanced',None) for c in (1.,4.)]
configs += [{'C':4.,'class_weight':'balanced','solver':'lbfgs','penalty':'l2','max_iter':6000,'seed':seed} for seed in (7,17)]
dump('search_plan.json',{'configurations':configs,'max_experiments':20,'plateau_patience':3,'primary':'strict eight-target exact full-record agreement','selection_tie_break':'No score gain resets plateau; preserve first equal-scoring model; supplemental fields do not redefine primary','validation_frozen_hash':sha(OUT/'provisional_validation.jsonl')})
for number,cfg in enumerate(configs[:20],1):
 exp=f'experiment_{number:02d}';directory=OUT/exp;directory.mkdir();start=time.time();artifact={k:copy.deepcopy(v) for k,v in baseline.items() if k not in ('execution_path','request_modes','clarification','escalation','missing_information','reference_kind','reference_resolved','capability_matcher')}
 def fit(X,y):
  h=LogisticRegression(C=cfg['C'],class_weight=cfg['class_weight'],solver=cfg['solver'],penalty=cfg['penalty'],max_iter=cfg['max_iter'],random_state=cfg['seed']);return h.fit(X,y)
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always',ConvergenceWarning)
  for field,key in [('execution_path','execution_path'),('needs_clarification','clarification'),('needs_escalation','escalation'),('missing_information','missing_information'),('reference_kind','reference_kind'),('reference_resolved','reference_resolved')]:artifact[key]=fit(Xtrain,targets[field])
  artifact['request_modes']={m:fit(Xtrain,targets['request_modes:'+m]) for m in ('conversational','memory','research','action')};artifact['capability_matcher']=fit(pair_train,Ypair)
 artifact.update({'adapter_id':'compact-mbert-frozen-provisional-'+exp,'training_examples':len(train),'training_example_ids':[r['id'] for r in train],'dataset_path':str(OUT/'provisional_train.jsonl'),'dataset_sha256':sha(OUT/'provisional_train.jsonl'),'training_split':'train','capability_encoding':'independent','training_configuration':{**cfg,'encoder_frozen':True,'feature_pipeline':'per-example descriptor parity','label_status':'synthetic_unverified'},'encoder_snapshot':str(snapshot),'encoder_file_hashes':cache_files,'label_status':'PROVISIONAL_SYNTHETIC_UNVERIFIED_NOT_GOLD','validation_sha256':sha(OUT/'provisional_validation.jsonl'),'source_record_training_ids':[r['original_id'] for r in train]})
 model_path=directory/'compact_heads.joblib';joblib.dump(artifact,model_path,compress=3)
 metrics=evaluate(artifact);dump(exp+'/metrics.json',metrics)
 info={'experiment':exp,'configuration':cfg,'model_sha256':sha(model_path),'training_records':len(train),'validation_records':len(validation),'encoder_frozen':True,'training_seconds':time.time()-start,'convergence_warnings':[str(w.message) for w in caught],'head_classes':{k:v.classes_.tolist() for k,v in artifact.items() if hasattr(v,'classes_')},'request_mode_classes':{k:v.classes_.tolist() for k,v in artifact['request_modes'].items()},'primary':metrics['exact_full_record'],'fields':metrics['fields'],'weight_finiteness':all(np.isfinite(h.coef_).all() for h in [artifact['execution_path'],artifact['capability_matcher'],*artifact['request_modes'].values()])}
 dump(exp+'/metadata.json',info)
 score=metrics['exact_full_record']['value'];improved=score>best_score
 if improved:best_score=score;best=metrics;best_artifact=artifact;best_name=exp;plateau=0
 else:plateau+=1
 history.append({'experiment':exp,'C':cfg['C'],'class_weight':str(cfg['class_weight']),'solver':cfg['solver'],'penalty':cfg['penalty'],'max_iter':cfg['max_iter'],'seed':cfg['seed'],'exact_agreement':metrics['exact_full_record']['agreement'],'denominator':len(validation),'exact_value':score,'path_agreement':metrics['fields']['execution_path']['agreement'],'capability_agreement':metrics['fields']['selected_capability']['agreement'],'improved_primary':improved,'plateau_count':plateau,'training_seconds':info['training_seconds'],'model_sha256':info['model_sha256']})
 table('experiment_history.csv',history,list(history[0]));progress(f'{exp}: full-record {metrics["exact_full_record"]["agreement"]}/{len(validation)}, path {metrics["fields"]["execution_path"]["agreement"]}/{len(validation)}; best={best_score:.1%}; plateau={plateau}/3')
 if score>=.90:stop_reason='PROVISIONAL_TARGET_REACHED';break
 if plateau>=3:stop_reason='THREE_CONSECUTIVE_PRIMARY_PLATEAU_EXPERIMENTS';break
else:stop_reason='TWENTY_EXPERIMENT_LIMIT'

bestdir=OUT/'best_candidate';bestdir.mkdir();joblib.dump(best_artifact,bestdir/'compact_heads.joblib',compress=3)
best_meta={'selected_experiment':best_name,'selection_metric':'exact full eight-target provisional-label agreement','best':best['exact_full_record'],'baseline':baseline_metrics['exact_full_record'],'target_reached':best_score>=.90,'stop_reason':stop_reason,'experiments_run':len(history),'best_model_path':str(bestdir/'compact_heads.joblib'),'best_model_sha256':sha(bestdir/'compact_heads.joblib'),'configuration':history[int(best_name.split('_')[-1])-1],'training_records':len(train),'validation_records':len(validation),'gold_verified_accuracy':'NOT_MEASURED','encoder_frozen':True,'no_original_model_replacement':True}
dump('best_candidate_metadata.json',best_meta);dump('best_candidate/compact_heads.json',best_meta)
dump('per_head_metrics.json',best['heads']);dump('per_class_metrics.json',{k:v['per_class'] for k,v in best['heads'].items()});dump('per_language_metrics.json',best['languages'])
for key,value in best['heads'].items():dump('confusion_matrices/'+key.replace(':','_')+'.json',{'classes':value['classes'],'matrix':value['confusion_matrix'],'denominator':value['denominator'],'metric_type':'PROVISIONAL_LABEL_AGREEMENT'})
for row in best['predictions']:
 if row['exact_match']:continue
 r=next(r for r in validation if r['id']==row['record_id']);p=row['prediction'];causes=[]
 if r['expected']['context_references']:causes.append('context/reference deterministic output limitation or ambiguity')
 if r['expected']['missing_information']:causes.append('fixed missing description or missing-name head limitation')
 if p and p['execution_path']!=r['expected']['execution_path']:causes.append('path/uncertainty head or capability postprocessing error')
 if p and p['selected_capability']!=r['expected']['selected_capability']:causes.append('capability matcher coverage or feature limitation')
 reviews.append({'record_id':r['id'],'original_id':r['original_id'],'language':r['language'],'script':'+'.join(observed_scripts(r['utterance'])),'utterance':r['utterance'],'current_target':json.dumps(r['expected'],ensure_ascii=False),'prediction':json.dumps(p,ensure_ascii=False),'probable_causes':'; '.join(causes) or 'semantic annotation uncertainty or feature limitation','review_action':'Human semantic review; prediction disagreement alone is not proof of wrong target','label_changed':False})
table('proposed_label_review.csv',reviews,['record_id','original_id','language','script','utterance','current_target','prediction','probable_causes','review_action','label_changed'])
table('data_quality_findings.csv',findings,['record_id','split','issue_type','severity','evidence'])
unchanged={p:sha(p)==v['sha256'] for p,v in INPUTS.items()}
dump('reproducibility_manifest.json',{'python':sys.version,'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'sklearn':sklearn.__version__,'numpy':np.__version__,'joblib':joblib.__version__},'seeds':[42],'CPU_threads':4,'encoder_checkpoint':cache_files,'input_hashes':INPUTS,'inputs_unchanged':unchanged,'original_git_changes':'git_status_before.txt','scripts':{'run_experiments.py':sha(__file__)},'duration_seconds':time.time()-START,'outputs':{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='reproducibility_manifest.json'},'network_disabled':True,'production_code_modified':False,'sealed_labels_read_or_used':False})
progress(f'Loop complete: {len(history)} experiments, best exact {best["exact_full_record"]["agreement"]}/{len(validation)}; {stop_reason}')
