import sys,types,pathlib,os,platform,tempfile
root=pathlib.Path('D:/JARVIS');work=root/'scratch/bert_shadow_audit_20261010_140019'
sys.path.insert(0,str(root));os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1';sys.modules['readline']=None
tempfile.tempdir=str(work)
platform.uname()
package=types.ModuleType('assistant.orchestration');package.__path__=[str(root/'assistant/orchestration')];sys.modules['assistant.orchestration']=package
log=types.ModuleType('assistant.core.log');log.logger=__import__('logging').getLogger('isolated-offline-tests');sys.modules['assistant.core.log']=log
log.get=lambda *args,**kwargs:log.logger
def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):raise RuntimeError('External boundary blocked: '+event)
    if event=='open':
        path,mode,flags=args
        writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing and isinstance(path,(str,bytes,os.PathLike)) and str(path).lower() not in ('nul','\\\\.\\nul'):
            target=pathlib.Path(path).resolve()
            if target!=work and work not in target.parents:raise RuntimeError('Write blocked: '+str(target))
sys.addaudithook(guard)
import pytest
sys.exit(pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(work/'pytest-temp'),'tests/test_compact_classifier.py','tests/test_fast_brain_encoder_adapter.py']))
