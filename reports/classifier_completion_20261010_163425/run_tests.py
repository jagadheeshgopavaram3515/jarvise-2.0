"""Entire relevant offline suite with protected-label/network/write barriers."""
import os,sys
sys.dont_write_bytecode=True
os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
from pathlib import Path
import types,json,platform
ROOT=Path('D:/JARVIS');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
pkg=types.ModuleType('assistant.orchestration');pkg.__path__=[str(ROOT/'assistant/orchestration')];sys.modules['assistant.orchestration']=pkg
sys.modules['readline']=None;platform.uname()
import pytest
import tempfile,time
import logging
log_module=types.ModuleType('assistant.core.log');log_module.get=logging.getLogger;sys.modules['assistant.core.log']=log_module
scratch=OUT/'test_scratch';scratch.mkdir(exist_ok=True);tempfile.tempdir=str(scratch)
os.environ['TMP']=str(scratch);os.environ['TEMP']=str(scratch)
reports=[];blocked=[]
class Recorder:
 def pytest_runtest_logreport(self,report):
  if report.when=='call' or (report.when=='setup' and report.outcome!='passed'):
   reports.append({'test':report.nodeid,'phase':report.when,'outcome':report.outcome,'reason':str(report.longrepr) if report.longrepr else ''})
protected={ROOT/'benchmarks/fast_classifier/v1/seed.jsonl',*[ROOT/f'benchmarks/fast_classifier/v{i}/dataset.jsonl' for i in (2,3,4)],ROOT/'benchmarks/fast_classifier/fast_brain_v1/dataset.jsonl',ROOT/'benchmarks/fast_classifier/fast_brain_v1/review_decisions.jsonl'}
def guard(event,args):
 if event in ('socket.connect','socket.getaddrinfo'):
  pytest.skip('BLOCKED: real network prohibited; offline fixtures only',allow_module_level=True)
 if event=='open':
  p,mode,flags=args
  if not isinstance(p,(str,bytes,os.PathLike)):return
  if str(p).lower() in ('nul','\\\\.\\nul'):return
  path=Path(p).resolve();write=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if write and path!=OUT and OUT not in path.parents:pytest.skip('BLOCKED: existing-tree write prohibited: '+str(path),allow_module_level=True)
  if not write and (path in protected or 'benchmark-results' in path.parts or ('steward' in path.parts and 'fast_classification_gold_review_20261010_133541' in path.parts)):
   blocked.append(str(path));pytest.skip('BLOCKED: test would read original sealed-label or historical-outcome source; not allowed in this experiment',allow_module_level=True)
sys.addaudithook(guard)
names=['test_compact_classifier.py','test_classifier_benchmark.py','test_fast_classifier.py','test_fast_brain_dataset.py','test_fast_brain_foundation.py','test_fast_brain_encoder_adapter.py','test_fast_brain_review.py','test_fast_brain_review_cli.py','test_gold_review.py','test_shadow_review_evaluation.py','test_shadow_review_regression.py','test_phase4_benchmark.py']
paths=[str(ROOT/'tests'/n) for n in names]+[str(OUT/'test_completion.py')]
result=pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(OUT/('pytest_tmp_'+str(time.time_ns()))),'--junitxml',str(OUT/'pytest_results.xml'),*paths],plugins=[Recorder()])
(OUT/'test_results.json').write_text(json.dumps({'exit_code':int(result),'passed':sum(r['outcome']=='passed' for r in reports),'failed':sum(r['outcome']=='failed' for r in reports),'skipped_or_blocked':sum(r['outcome']=='skipped' for r in reports),'reports':reports,'blocked_source_reads':sorted(set(blocked)),'suite_files':paths,'protection':'Tests requiring original sealed labels/outcomes or existing-tree writes are explicitly skipped; never used for fitting/selection'},indent=2)+'\n',encoding='utf-8')
raise SystemExit(result)
