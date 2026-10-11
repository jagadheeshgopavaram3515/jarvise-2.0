import pathlib,sys,types,logging,socket,os
ROOT=pathlib.Path('D:/JARVIS');WORK=pathlib.Path(__file__).resolve().parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT));os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
def guard(event,args):
 if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):raise RuntimeError('External action prohibited')
 if event=='open':
  path,mode,flags=args;writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and isinstance(path,(str,bytes,os.PathLike)):
   target=pathlib.Path(os.fsdecode(path)).resolve()
   if target==ROOT or (ROOT in target.parents and WORK not in target.parents):raise RuntimeError('Write outside test work directory')
sys.addaudithook(guard)
pkg=types.ModuleType('assistant.orchestration');pkg.__path__=[str(ROOT/'assistant/orchestration')];sys.modules['assistant.orchestration']=pkg
log=types.ModuleType('assistant.core.log');log.get=lambda name:logging.getLogger(name);log.setup=lambda:None;sys.modules['assistant.core.log']=log
sys.modules['readline']=None
import pytest
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(WORK/'pytest-tmp'),str(ROOT/'tests/test_gold_review.py'),str(ROOT/'tests/test_fast_brain_foundation.py'),str(ROOT/'tests/test_fast_brain_dataset.py')]))
