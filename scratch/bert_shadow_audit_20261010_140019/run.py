import os, sys, json, hashlib, time, traceback, types, pathlib, collections, statistics, subprocess, platform
ROOT=pathlib.Path('D:/JARVIS'); WORK=ROOT/'scratch/bert_shadow_audit_20261010_140019'
OUT=ROOT/'reports/bert_shadow_classifier_20261010_140019'
OUT.mkdir(exist_ok=True)
def dump(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
protected={str(p.relative_to(ROOT)):sha(p) for directory in ['assistant','benchmarks','reports','tests','logs'] for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in str(p) and OUT not in p.parents}
dump(WORK/'integrity-before.json',protected)
(WORK/'status-before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),encoding='utf-8')
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
sys.path.insert(0,str(ROOT))
pkg=types.ModuleType('assistant.orchestration'); pkg.__path__=[str(ROOT/'assistant/orchestration')];sys.modules['assistant.orchestration']=pkg
sys.modules['readline']=None
platform.uname()
def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'): raise RuntimeError('Blocked external boundary: '+event)
    if event=='open':
        path,mode,flags=args
        writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing and isinstance(path,(str,bytes,os.PathLike)) and str(path).lower() not in ('nul','\\\\.\\nul'):
            target=pathlib.Path(path).resolve()
            if not any(target==d or d in target.parents for d in [WORK.resolve(),OUT.resolve()]): raise RuntimeError('Blocked write: '+str(target))
sys.addaudithook(guard)
population=[json.loads(s) for s in (ROOT/'benchmarks/fast_classifier/fast_brain_v1/dataset.jsonl').read_text(encoding='utf-8').splitlines()]
source={r['id']:r for r in map(json.loads,(ROOT/'benchmarks/fast_classifier/v4/dataset.jsonl').read_text(encoding='utf-8').splitlines())}
metadata=json.loads((ROOT/'benchmarks/fast_classifier/models/v3/compact_heads.json').read_text())
dump(OUT/'protocol.json',dict(population=len(population),primary_target='Existing offline compact adapter; no production shadow invocation found',network=False,retries=0,gold_records=0,accuracy='NOT MEASURED',metadata=metadata))
results=[];load_error=None;adapter=None
try:
    from assistant.orchestration.compact_adapter import CompactMultilingualAdapter
    from assistant.orchestration.decision import DecisionRequest,Capability,ClassifierDecision
    started=time.perf_counter();adapter=CompactMultilingualAdapter(ROOT/'benchmarks/fast_classifier/models/v3/compact_heads.joblib');load_seconds=time.perf_counter()-started
    heads={k:dict(type=type(v).__name__,classes=getattr(v,'classes_',None).tolist() if hasattr(v,'classes_') else None) for k,v in adapter.artifact.items() if hasattr(v,'predict')}
    dump(OUT/'model.json',dict(adapter_id=adapter.adapter_id,load_seconds=load_seconds,heads=heads,artifact_sha256=sha(ROOT/'benchmarks/fast_classifier/models/v3/compact_heads.joblib'),model_id=adapter.encoder.model_id,max_length=adapter.encoder.max_length,device=str(next(adapter.encoder.model.parameters()).device)))
except Exception:
    load_error=traceback.format_exc();dump(OUT/'load-error.json',dict(error=load_error));print(load_error,flush=True)
fields=['intent','execution_path','request_modes','selected_capability','context_references','missing_information','needs_clarification','needs_escalation']
with (OUT/'predictions.jsonl').open('w',encoding='utf-8') as handle:
    for index,row in enumerate(population):
        original=source[row['original_id']]
        result=dict(record_id=row['id'],source_id=row['original_id'],utterance=row['text'],language=row['language_group'],script=row['script_group'],split=row['split'],context=original['context'],capabilities=original['capabilities'],historical_label=original['expected'],training_overlap=row['original_id'] in metadata['training_example_ids'],status='not_attempted',prediction=None,error=load_error)
        if adapter:
            result['status']='failed';start=time.perf_counter();captured=[];real_encode=adapter.encoder.encode
            def capture(texts,**kwargs): captured.extend(texts);return real_encode(texts,**kwargs)
            adapter.encoder.encode=capture
            try:
                caps=tuple(Capability(c['name'],c['description'],tuple(c.get('required_arguments',[]))) for c in original['capabilities'])
                prediction=dict(adapter.predict(DecisionRequest(row['text'],original['context'],caps)))
                ClassifierDecision.from_mapping(prediction)
                result.update(status='success',prediction=prediction,schema_valid=True,error=None,inference_metrics=adapter.last_inference_metrics)
                result['historical_field_agreement']={f:(set(prediction[f])==set(original['expected'][f]) if f=='request_modes' else prediction[f]==original['expected'][f]) for f in fields}
                result['historical_exact_agreement']=all(result['historical_field_agreement'].values())
            except Exception: result.update(error=traceback.format_exc(),schema_valid=False)
            finally: adapter.encoder.encode=real_encode
            result.update(latency_ms=(time.perf_counter()-start)*1000,encoder_inputs=captured)
        handle.write(json.dumps(result,ensure_ascii=False,default=str)+'\n');handle.flush();results.append(result)
        if adapter and (index+1)%10==0: print(f'Completed {index+1}/{len(population)}; success {sum(r["status"]=="success" for r in results)}',flush=True)
success=[r for r in results if r['status']=='success'];lat=sorted(r['latency_ms'] for r in success)
groups={}
for dimension in ['language','script','split']:
    groups[dimension]={v:dict(total=sum(r[dimension]==v for r in results),successful=sum(r[dimension]==v for r in success),historical_exact_agreements=sum(r[dimension]==v and r['historical_exact_agreement'] for r in success)) for v in sorted({r[dimension] for r in results})}
metrics=dict(total=len(results),attempted=sum(r['status']!='not_attempted' for r in results),successful=len(success),failed=sum(r['status']=='failed' for r in results),not_attempted=sum(r['status']=='not_attempted' for r in results),gold_accuracy='NOT MEASURED',historical_exact_agreements=sum(r['historical_exact_agreement'] for r in success),historical_field_agreements={f:sum(r['historical_field_agreement'][f] for r in success) for f in fields},training_overlap=sum(r['training_overlap'] for r in results),distribution={f:dict(collections.Counter(str(r['prediction'][f]) for r in success)) for f in ['intent','execution_path','selected_capability']},groups=groups,median_ms=statistics.median(lat) if lat else None,p95_ms=lat[min(len(lat)-1,int(len(lat)*.95))] if lat else None)
dump(OUT/'metrics.json',metrics)
changed=[p for p,h in protected.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
dump(OUT/'integrity.json',dict(protected_files=len(protected),changed_preexisting_files=changed,network_blocked=True,retries=0))
print(json.dumps(metrics,ensure_ascii=False),flush=True)
