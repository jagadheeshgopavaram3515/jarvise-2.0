import os,sys
sys.dont_write_bytecode=True
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
from pathlib import Path
import types,json,hashlib,csv,copy,time,collections,subprocess,platform
ROOT=Path('D:/JARVIS');OUT=Path(__file__).resolve().parent;PRIOR=ROOT/'reports/autonomous_bert_training_20261010_161410';BUILD=ROOT/'reports/multilingual_dataset_build_20261010_155129'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BUILD));sys.path.insert(0,str(OUT))
pkg=types.ModuleType('assistant.orchestration');pkg.__path__=[str(ROOT/'assistant/orchestration')];sys.modules['assistant.orchestration']=pkg
sys.modules['readline']=None;platform.uname()
INPUTS={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):INPUTS[str(Path(p).resolve())]=sha(p);return Path(p).read_text(encoding='utf-8-sig')
def js(p):return json.loads(read(p))
def jl(p):return [json.loads(l) for l in read(p).splitlines() if l.strip()]
def dump(n,x):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
def tab(n,rs):
 with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(rs[0]) if rs else ['status']);w.writeheader();w.writerows(rs)
def guard(event,args):
 if event in ('socket.connect','socket.getaddrinfo','os.system'):raise RuntimeError('offline boundary blocked')
 if event=='open':
  p,m,flags=args;write=(isinstance(m,str) and any(c in m for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if write and isinstance(p,(str,bytes,os.PathLike)) and str(p).lower() not in ('nul','\\\\.\\nul'):
   path=Path(p).resolve()
   if path!=OUT and OUT not in path.parents:raise RuntimeError('write outside experiment blocked: '+str(path))
sys.addaudithook(guard)
import joblib,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from assistant.orchestration.compact_adapter import CompactMultilingualAdapter,serialize_request,serialize_capability,FrozenMultilingualEncoder
from assistant.orchestration.decision import DecisionRequest,Capability,ClassifierDecision
from dataset_tools import validate_candidate,compact_targets,observed_scripts,normalized
from experimental_adapter import ExperimentalCompactAdapter,pair_features,typed_resolve
FIELDS=('intent','execution_path','request_modes','selected_capability','context_references','missing_information','needs_clarification','needs_escalation')
def request(r):return DecisionRequest(r['utterance'],r['context'],tuple(Capability(c['name'],c['description'],tuple(c['required_arguments'])) for c in r['capabilities']))
def exact(r,p):return all((sorted(p[k])==sorted(r['expected'][k]) if k=='request_modes' else p[k]==r['expected'][k]) for k in FIELDS)
prior_manifest=js(PRIOR/'reproducibility_manifest.json')
if not all(sha(PRIOR/n)==v for n,v in prior_manifest['outputs'].items()):raise ValueError('prior artifact hashes changed')
train=jl(PRIOR/'provisional_train.jsonl');val=jl(PRIOR/'provisional_validation.jsonl');limits=js(PRIOR/'deterministic_output_limits.json')
for r in train+val:validate_candidate(r)
protected=set(js(BUILD/'protected_test_manifest.json')['original_protected_record_ids']);baseline_ids=set(js(ROOT/'benchmarks/fast_classifier/models/v3/compact_heads.json')['training_example_ids'])
assert len(train)==232 and len(val)==40
assert not ({r['group_id'] for r in train}&{r['group_id'] for r in val})
assert not ({normalized(r['utterance']) for r in train}&{normalized(r['utterance']) for r in val})
assert all(r['original_id'] not in protected for r in train+val)
assert not (baseline_ids&{r['original_id'] for r in val})
cache=js(PRIOR/'feature_cache_manifest.json');INPUTS[str((PRIOR/'frozen_features.npz').resolve())]=sha(PRIOR/'frozen_features.npz')
for p,info in cache['encoder_files'].items():
 INPUTS[p]=sha(p)
 if INPUTS[p]!=info['sha256']:raise ValueError('encoder hash mismatch')
with np.load(PRIOR/'frozen_features.npz') as a:vectors=a['vectors']
index={x['text_sha256']:x['index'] for x in cache['texts']}
def encoded(text):return vectors[index[hashlib.sha256(text.encode()).hexdigest()]]
class ReplayEncoder:
 def __init__(self):self.forward_calls=0;self.last_timings={}
 def encode(self,texts,**kwargs):self.forward_calls+=1;return np.stack([encoded(t) for t in texts])
X=np.stack([encoded(serialize_request(request(r))) for r in train]);XV=np.stack([encoded(serialize_request(request(r))) for r in val])
def pairs(records,style):
 xx=[];yy=[]
 for r in records:
  q=encoded(serialize_request(request(r)))
  for c in request(r).capabilities:xx.append(pair_features(q,encoded(serialize_capability(c.name,c.description,c.required_arguments)),style));yy.append(int(c.name==r['expected']['selected_capability']))
 return np.stack(xx),np.array(yy)
def make_adapter(artifact,experimental=False):
 cls=ExperimentalCompactAdapter if experimental else CompactMultilingualAdapter
 a=cls.__new__(cls);a.artifact=artifact;a.encoder=ReplayEncoder();a._adapter_id=artifact['adapter_id'];a.inference_count=0;a.last_inference_metrics={};return a
def metric(y,p,classes=[]):
 y=list(map(str,y));p=list(map(str,p));labels=sorted(set(y+p+list(map(str,classes))))
 return {'agreement':sum(a==b for a,b in zip(y,p)),'denominator':len(y),'classes':labels,'report':classification_report(y,p,labels=labels,output_dict=True,zero_division=0),'confusion':confusion_matrix(y,p,labels=labels).tolist()}
def evaluate(artifact,experimental):
 a=make_adapter(artifact,experimental);pred=[];latency=[];errors=[]
 for r in val:
  start=time.perf_counter()
  try:p=dict(a.predict(request(r)));ClassifierDecision.from_mapping(p);error=None
  except Exception as ex:p=None;error=str(ex)
  latency.append((time.perf_counter()-start)*1000);pred.append(p);errors.append(error)
 agreement=[p is not None and exact(r,p) for r,p in zip(val,pred)]
 fields={f:sum(p is not None and (sorted(p[f])==sorted(r['expected'][f]) if f=='request_modes' else p[f]==r['expected'][f]) for r,p in zip(val,pred)) for f in FIELDS}
 heads={}
 for field,key in [('execution_path','execution_path'),('needs_clarification','clarification'),('needs_escalation','escalation'),('missing_information','missing_information'),('reference_kind','reference_kind'),('reference_resolved','reference_resolved')]:
  h=artifact[key];heads[field]=metric([compact_targets(r)[field] for r in val],h.predict(XV),h.classes_)
 for m,h in artifact['request_modes'].items():heads['mode:'+m]=metric([int(m in r['expected']['request_modes']) for r in val],h.predict(XV),h.classes_)
 xx,yy=pairs(val,artifact.get('pair_feature_style','full'));h=artifact['capability_matcher'];heads['capability_matcher']=metric(yy,h.predict(xx),h.classes_)
 if experimental:
  for name,field,key in [('reference_span','context_references','reference'),('missing_description','missing_information','description')]:
   ix=[i for i,r in enumerate(val) if r['expected'][field]];gold=[val[i]['expected'][field][0][key] for i in ix]
   if name=='reference_span':gold=[s if s!=val[i]['utterance'] else '__utterance__' for i,s in zip(ix,gold)]
   heads[name]=metric(gold,artifact[name].predict(XV[ix]),artifact[name].classes_)
 output={f:metric([json.dumps(r['expected'][f],sort_keys=True) for r in val],[json.dumps(p[f],sort_keys=True) if p else '__ERROR__' for p in pred]) for f in FIELDS if f not in ('context_references','missing_information')}
 languages={}
 for lang in sorted({r['language'] for r in train+val}):
  ix=[i for i,r in enumerate(val) if r['language']==lang];languages[lang]={'exact_agreement':sum(agreement[i] for i in ix),'denominator':len(ix),'scenario_groups':len({val[i]['group_id'] for i in ix}),'class_support':dict(collections.Counter(val[i]['expected']['execution_path'] for i in ix)),'scripts':dict(collections.Counter('+'.join(observed_scripts(val[i]['utterance'])) for i in ix)),'status':'INSUFFICIENT_INDEPENDENTLY_VERIFIED_EVIDENCE'}
 return {'metric_type':'PROVISIONAL_LABEL_AGREEMENT','exact_agreement':sum(agreement),'denominator':len(val),'fields':fields,'raw_heads':heads,'output_metrics':output,'languages':languages,'inference_failures':sum(x is not None for x in errors),'abstentions':sum(bool(p and p.get('abstained')) for p in pred),'cached_latency_ms':{'median':float(np.median(latency)),'p95':float(np.percentile(latency,95))},'predictions':[{'record_id':r['id'],'original_id':r['original_id'],'expected':r['expected'],'prediction':p,'exact':ok,'error':er} for r,p,ok,er in zip(val,pred,agreement,errors)]}
baseline_path=ROOT/'benchmarks/fast_classifier/models/v3/compact_heads.joblib';prior_path=PRIOR/'best_candidate/compact_heads.joblib'
for p in (baseline_path,prior_path):INPUTS[str(p.resolve())]=sha(p)
baseline=joblib.load(baseline_path);previous=joblib.load(prior_path)
reproduction={}
def canonical_prediction(p):
 p=copy.deepcopy(p)
 if p is not None:p['request_modes']=sorted(p['request_modes'])
 return p
for name,artifact,saved in [('baseline',baseline,js(PRIOR/'baseline_metrics.json')),('previous_best',previous,js(PRIOR/'experiment_01/metrics.json'))]:
 m=evaluate(artifact,False);dump(name+'_reproduction.json',m)
 checks=[canonical_prediction(p['prediction'])==canonical_prediction(q['prediction']) for p,q in zip(m['predictions'],saved['predictions'])]
 if not all(checks) or m['exact_agreement']!=8:raise ValueError('Baseline reproduction failed; do not continue')
 reproduction[name]={'exact_prediction_reproduction':True,'exact_agreement':m['exact_agreement'],'denominator':40,'serialization_discrepancy':'request_modes list order varies with process hash seed; actual contract is frozenset, so comparison uses sorted modes. No field value or score changed.'}
dump('baseline_reproduction_checks.json',reproduction);print('Baseline and prior candidate reproduced exactly: 8/40 each.',flush=True)
# Rendering-label heads use TRAIN only; retain arbitrary source wording without validation lookup.
span_labels=[('__none__' if not r['expected']['context_references'] else '__utterance__' if r['expected']['context_references'][0]['reference']==r['utterance'] else r['expected']['context_references'][0]['reference']) for r in train]
description_ix=[i for i,r in enumerate(train) if r['expected']['missing_information']]
description_labels=[train[i]['expected']['missing_information'][0]['description'] for i in description_ix]
def fit(xx,yy,C,weight):return LogisticRegression(C=C,class_weight=weight,max_iter=3000,solver='lbfgs',random_state=42).fit(xx,yy)
configs=[{'C':C,'weight':weight,'style':style,'capability_threshold':threshold} for C,weight,style,threshold in [(4,'balanced','full',.5),(4,'balanced','interaction',.5),(4,None,'interaction',.5),(4,'balanced','interaction',.25),(1,'balanced','interaction',.25),(16,'balanced','interaction',.25),(4,None,'full',.25),(16,None,'interaction',.25)]]
dump('search_plan.json',{'configs':configs,'fixed_validation_sha256':sha(PRIOR/'provisional_validation.jsonl'),'rendering_targets':'TRAIN only; no validation labels in renderer vocabulary','new_heads':'reference_span and missing_description are LR renderers preserving ClassifierDecision output shape','encoder_frozen':True})
history=[];best=None;best_art=None;best_name=None
for i,cfg in enumerate(configs,1):
 name=f'experiment_{i:02d}';(OUT/name).mkdir();a=copy.deepcopy(previous)
 for f,k in [('execution_path','execution_path'),('needs_clarification','clarification'),('needs_escalation','escalation'),('missing_information','missing_information'),('reference_kind','reference_kind'),('reference_resolved','reference_resolved')]:a[k]=fit(X,[compact_targets(r)[f] for r in train],cfg['C'],cfg['weight'])
 a['request_modes']={m:fit(X,[int(m in r['expected']['request_modes']) for r in train],cfg['C'],cfg['weight']) for m in ('conversational','memory','research','action')}
 xx,yy=pairs(train,cfg['style']);a['capability_matcher']=fit(xx,yy,cfg['C'],cfg['weight'])
 a['reference_span']=fit(X,span_labels,cfg['C'],cfg['weight']);a['missing_description']=fit(X[description_ix],description_labels,cfg['C'],cfg['weight'])
 a.update({'adapter_id':'compact-experimental-contract-v2-'+name,'artifact_schema_version':'experimental-3.0','pair_feature_style':cfg['style'],'capability_threshold':cfg['capability_threshold'],'rendering_vocabulary_source':'training targets only','experiment_configuration':cfg,'experimental_adapter_required':'experimental_adapter.ExperimentalCompactAdapter'})
 path=OUT/name/'compact_heads.joblib';joblib.dump(a,path,compress=3);m=evaluate(a,True);dump(name+'/metrics.json',m);dump(name+'/config.json',{**cfg,'artifact_sha256':sha(path)})
 score=(m['exact_agreement'],m['fields']['execution_path'],m['fields']['selected_capability'])
 if best is None or score>(best['exact_agreement'],best['fields']['execution_path'],best['fields']['selected_capability']):best=m;best_art=a;best_name=name
 history.append({'experiment':name,**cfg,'strict_exact':m['exact_agreement'],'denominator':40,'path':m['fields']['execution_path'],'capability':m['fields']['selected_capability'],'reference':m['fields']['context_references'],'missing':m['fields']['missing_information'],'failures':m['inference_failures'],'abstentions':m['abstentions'],'model_sha256':sha(path)})
 print(f'{name}: exact {m["exact_agreement"]}/40; capability {m["fields"]["selected_capability"]}/40',flush=True)
tab('experiment_history.csv',history);(OUT/'best_candidate').mkdir();joblib.dump(best_art,OUT/'best_candidate/compact_heads.joblib',compress=3)
dump('best_candidate_metadata.json',{'selected_experiment':best_name,'strict_exact':best['exact_agreement'],'denominator':40,'model_sha256':sha(OUT/'best_candidate/compact_heads.joblib'),'model_path':str(OUT/'best_candidate/compact_heads.joblib'),'metric':'PROVISIONAL_LABEL_AGREEMENT','verified_accuracy':'NOT_MEASURED','configuration':best_art['experiment_configuration'],'experimental_adapter_path':str(OUT/'experimental_adapter.py')})
dump('best_metrics.json',best)
abstention=[]
for threshold in (0,.25,.5,.65,.8,.9):
 a=make_adapter(best_art,True);accepted=correct=full=0
 for r in val:
  p=a.predict_with_abstention(request(r),threshold);accept=not p.get('abstained',False);accepted+=accept;correct+=accept and exact(r,p);full+=exact(r,p)
 abstention.append({'threshold':threshold,'accepted':accepted,'denominator':40,'coverage':accepted/40,'accepted_exact':correct,'accepted_correctness':correct/accepted if accepted else None,'full_set_exact':full,'full_set_denominator':40})
dump('abstention_metrics.json',abstention)
cases=[];ids={r['id']:r for r in val}
for item in limits['records']:
 r=ids[item['record_id']];prediction=next(x['prediction'] for x in best['predictions'] if x['record_id']==r['id'])
 subtype='C adapter defect plus D inconsistent rendering convention'
 if item['original_id'] in ('en-memory-001','hi-codeswitch-memory-001'):subtype+='; E unseen exact span vocabulary'
 cases.append({'record_id':r['id'],'original_id':r['original_id'],'old_issues':'; '.join(item['reasons']),'classification':subtype,'general_fix':'training-only span/description heads; typed context resolver; explicit unresolved fallback','after_reference_exact':prediction['context_references']==r['expected']['context_references'],'after_missing_exact':prediction['missing_information']==r['expected']['missing_information'],'after_full_exact':exact(r,prediction),'source_labels_changed':False,'unresolved':'Head error/unseen wording remains if after flags false; no validation phrase added to vocabulary'})
tab('eleven_incompatibilities.csv',cases)
distribution=[]
for split,rs in [('train',train),('validation',val)]:
 for field in compact_targets(rs[0]):
  vals=[str(v) for r in rs for v in (compact_targets(r)[field] if isinstance(compact_targets(r)[field],list) else [compact_targets(r)[field]])]
  for cl,n in collections.Counter(vals).items():distribution.append({'split':split,'head':field,'class':cl,'count':n,'denominator':len(vals)})
tab('class_distributions.csv',distribution)
dump('data_quality_report.json',{'rows':{'train':232,'validation':40},'schema_invalid':0,'training_status':'synthetic_unverified','group_overlap':0,'normalized_text_overlap':0,'protected_overlap':0,'baseline_training_ID_overlap_validation':0,'semantic_grouping':'Prior connected scenario/translation/template groups; broader semantic relation completeness NOT_VERIFIED','unseen_span_labels':sorted({ref['reference'] for r in val for ref in r['expected']['context_references'] if ref['reference']!=r['utterance']}-set(span_labels)),'missing_reference_kind_training':'project; addressed only by explicit project-alias context rule, not fabricated training samples','exclusions':'18 pending training records retained in prior exclusion ledger; fixed validation never filtered','family_mapping':'Prior corrected isolated builder maps all19 Registry actions; no source changes','feature_schema':{'request':768,'full_pairs':3072,'interaction_pairs':1536,'target_leakage':False,'descriptor_key':'per-example full serialized string'}})
# Actual local encoder timing on fixed validation, no downloads; cached timing separate.
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
encoder=FrozenMultilingualEncoder(cache['encoder_snapshot'],160)
for p in encoder.model.parameters():p.requires_grad_(False)
actual=make_adapter(best_art,True);actual.encoder=encoder;times=[];actual_preds=[]
for r in val:
 start=time.perf_counter();actual_preds.append(dict(actual.predict(request(r))));times.append((time.perf_counter()-start)*1000)
actual_exact=sum(exact(r,p) for r,p in zip(val,actual_preds))
dump('actual_encoder_evaluation.json',{'denominator':40,'strict_exact':actual_exact,'cache_vs_actual_all_predictions_equal':all(p==s['prediction'] for p,s in zip(actual_preds,best['predictions'])),'load_ms':encoder.load_ms,'latency_ms':{'median':float(np.median(times)),'p95':float(np.percentile(times,95))},'encoder_frozen':not any(p.requires_grad for p in encoder.model.parameters()),'predictions':actual_preds})
for p in ['assistant/orchestration/compact_adapter.py','assistant/orchestration/decision.py','benchmarks/fast_classifier/train_candidate.py','assistant/tools/registry.py']:INPUTS[str((ROOT/p).resolve())]=sha(ROOT/p)
dump('run_manifest.json',{'inputs':INPUTS,'unchanged':{p:sha(p)==v for p,v in INPUTS.items()},'train_sha256':sha(PRIOR/'provisional_train.jsonl'),'validation_sha256':sha(PRIOR/'provisional_validation.jsonl'),'encoder_files':cache['encoder_files'],'baseline_model_sha256':sha(baseline_path),'previous_candidate_sha256':sha(prior_path),'training_experiments':len(history),'seed':42,'production_changes':False,'sealed_labels_used':False,'reference_projects_accessed':False,'selection':'strict exact first, then path/capability tie-break disclosed'})
print('Completed fits and actual offline encoder evaluation.',flush=True)
