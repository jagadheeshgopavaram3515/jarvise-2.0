"""Offline candidate builder and fail-closed compact gold/feature gate.

No encoder or classifier is constructed. No historical reviewer decision is
converted to a complete compact gold label. Runtime changes are proposals only.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import ast, copy, datetime, hashlib, json, re, unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.fast_classifier.evaluate import _parse_example, load_dataset
from assistant.orchestration.decision import ClassifierDecision, DecisionRequest
from assistant.orchestration.compact_adapter import serialize_capability, _pair_features
from assistant.orchestration.fast_brain.contract import CapabilityFamily, MissingInformation as FamilyMissing

STATES=('verified_gold','verified_gold_needs_adjudication','pending_review','ambiguous','rejected','synthetic_unverified')
PATHS=frozenset(('direct_response','memory_recall','research','single_action','multi_step_task','clarify','escalate'))
VERSION='compact-dataset-candidates-1.0.0'
def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def fingerprint(x): return hashlib.sha256(canonical(x).encode('utf-8')).hexdigest()
def normalized(text): return re.sub(r'\s+',' ',unicodedata.normalize('NFC',text).casefold()).strip()
def observed_scripts(text):
 result=[]
 for name,a,b in [('telugu',0xc00,0xc7f),('devanagari',0x900,0x97f),('tamil',0xb80,0xbff)]:
  if any(a<=ord(c)<=b for c in text): result.append(name)
 if any(c.isalpha() and 'LATIN' in unicodedata.name(c,'') for c in text): result.append('latin')
 return result or ['unknown']

def extract_missing(items, *, taxonomy='compact'):
 """Compact missing names are open strings; FastBrain has a closed enum."""
 if not isinstance(items,list): raise ValueError('missing_information must be an array')
 result=[]
 for pos,item in enumerate(items):
  if isinstance(item,dict):
   if 'name' not in item: raise ValueError(f'missing_information[{pos}]: structured object needs name')
   name=item['name']; desc=item.get('description','')
  elif isinstance(item,str): name=item; desc=''
  else: raise ValueError(f'missing_information[{pos}]: expected string or name/description object')
  if not isinstance(name,str) or not name.strip() or not isinstance(desc,str): raise ValueError(f'missing_information[{pos}]: invalid name/description')
  if taxonomy=='fast_brain' and name not in {v.value for v in FamilyMissing}: raise ValueError(f'missing_information[{pos}]: unknown FastBrain enum {name!r}; compact has no such closed enum')
  if taxonomy not in ('compact','fast_brain'): raise ValueError('unknown taxonomy')
  result.append({'name':name,'description':desc})
 return result

CAPABILITY_FAMILIES={
 'location':'location','weather':'weather','open_app':'application','close_app':'application',
 'list_running_apps':'application','find_file':'file','search_files':'file','open_file':'file',
 'open_folder':'file','recent_files':'file','read_file':'file','list_directory':'file',
 'grep_code':'project','run_project_tests':'project','open_url':'browser',
 'google_search':'browser','youtube_search':'browser','read_page':'browser','get_page_title':'browser'}
def capability_family(name):
 if name is None: return 'none'
 if name not in CAPABILITY_FAMILIES: raise ValueError(f'Unmapped registered capability {name!r}')
 family=CAPABILITY_FAMILIES[name]
 assert family in {x.value for x in CapabilityFamily}
 return family

def registry_specs(path=ROOT/'assistant/tools/registry.py'):
 """Static Registry extraction avoids importing logger or tool handlers."""
 specs={}
 for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
  if not (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='ToolSpec'): continue
  kw={x.arg:x.value for x in node.keywords}
  name=ast.literal_eval(kw['name']); params=ast.literal_eval(kw['parameters'])
  specs[name]={'name':name,'description':ast.literal_eval(kw['description']),'required_arguments':params.get('required',[]),'parameter_schema':params}
 return specs

def aligned_capabilities(caps,specs):
 """Only derived required-argument metadata changes; preserve source descriptors."""
 result=[]; changes=[]
 for original in caps:
  c=copy.deepcopy(original); name=c.get('name')
  if name not in specs: raise ValueError(f'Capability {name!r} absent from current Registry')
  req=specs[name]['required_arguments']
  if c.get('required_arguments')!=req:
   changes.append({'name':name,'field':'required_arguments','original':c.get('required_arguments'),'derived':req,'reason':'Current Registry required parameters; semantic prerequisites need separate review'})
   c['required_arguments']=copy.deepcopy(req)
  result.append(c)
 return result,changes

def compact_targets(record):
 e=record['expected']; refs=e['context_references']; missing=e['missing_information']
 out={'execution_path':e['execution_path'],'needs_clarification':int(e['needs_clarification']),'needs_escalation':int(e['needs_escalation']),
 'missing_information':missing[0]['name'] if missing else '__none__','reference_kind':refs[0]['kind'] if refs else '__none__',
 'reference_resolved':int(bool(refs and refs[0].get('resolved_value') is not None))}
 for mode in ('conversational','memory','research','action'): out['request_modes:'+mode]=int(mode in e['request_modes'])
 out['capability_matcher']=[int(c['name']==e['selected_capability']) for c in record['capabilities']]
 return out

def validate_candidate(r):
 ex=_parse_example(r)
 if ex.expected.execution_path not in PATHS: raise ValueError('unknown compact semantic execution_path')
 names=[c.name for c in ex.capabilities]
 if len(set(names))!=len(names): raise ValueError('duplicate capability names within one request')
 e=r['expected']; ClassifierDecision.from_mapping({**e,'confidence':0.0,'evidence':[]})
 extract_missing(e['missing_information'])
 if e['selected_capability'] is not None and e['selected_capability'] not in names: raise ValueError('selected capability not advertised')
 if e['needs_clarification'] and e['needs_escalation']: raise ValueError('competing uncertainty flags')
 if e['execution_path']=='clarify' and not e['needs_clarification']: raise ValueError('clarify path without clarification')
 if e['execution_path']=='escalate' and not e['needs_escalation']: raise ValueError('escalate path without escalation')
 if e['execution_path'] in ('single_action','research','multi_step_task') and e['selected_capability'] is None: raise ValueError('action path without selected capability')
 if len(e['context_references'])>1 or len(e['missing_information'])>1: raise ValueError('current heads only retain first reference/missing item')
 return ex

def review_input_hash(r):
 return fingerprint({'utterance':r['utterance'],'context':r['context'],'capabilities':r['capabilities']})
def verification_hash(r): return fingerprint({'input':review_input_hash(r),'expected':r['expected'],'schema':'ClassifierDecision','policy':'compact-two-reviewer-v1'})
def gold_gate(r):
 """Requires two separate qualified exact-contract approvals, not an old accept."""
 validate_candidate(r)
 if r.get('dataset_state')!='verified_gold': raise ValueError('only verified_gold can enter supervised fitting')
 if r.get('protected_source') or r.get('original_test_lineage'): raise ValueError('original sealed/test lineage excluded from fitting')
 v=r.get('verification',{})
 if v.get('contract')!='ClassifierDecision' or v.get('policy')!='compact-two-reviewer-v1': raise ValueError('exact contract/policy approval missing')
 if v.get('record_sha256')!=verification_hash(r): raise ValueError('reviewed input/target hash mismatch')
 reviewers=v.get('reviewers',[])
 if len(reviewers)<2 or len({x.get('reviewer_id') for x in reviewers})!=len(reviewers): raise ValueError('two distinct reviewers required')
 expected_hash=fingerprint(r['expected'])
 languages=set(r.get('language_metadata',{}).get('required_languages',[])); scripts=set(observed_scripts(r['utterance']))-{'unknown'}
 for rv in reviewers:
  if not rv.get('reviewer_id') or rv.get('reviewer_type')!='human' or rv.get('review_kind')!='independent' or rv.get('prediction_blinded') is not True: raise ValueError('independent blinded human approval required')
  if rv.get('decision')!='approve' or rv.get('target_sha256')!=expected_hash: raise ValueError('reviewer target approval missing or disputed')
  if not languages or not languages.issubset(set(rv.get('qualification',{}).get('languages',[]))) or not scripts.issubset(set(rv.get('qualification',{}).get('scripts',[]))): raise ValueError('language/script qualifications missing')
  if r.get('language_metadata',{}).get('transliteration_expertise_required') and rv.get('qualification',{}).get('transliteration') is not True: raise ValueError('transliteration qualification missing')
  try: ts=datetime.datetime.fromisoformat(rv.get('timestamp','').replace('Z','+00:00'))
  except ValueError: raise ValueError('review timestamp missing')
  if ts.tzinfo is None: raise ValueError('review timestamp must include timezone')
  rationale=rv.get('field_rationale',{})
  if set(rationale)!=set(r['expected']) or not all(isinstance(x,str) and x.strip() for x in rationale.values()): raise ValueError('full-field rationale required')
 if v.get('unresolved_disagreement'): raise ValueError('unresolved disagreement')
 return True

def load_verified_training(path):
 """Read only the dedicated train file, reject pending before encoder creation."""
 records=[json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]
 if not records: raise ValueError('No verified training records; training blocked')
 ids=set()
 for r in records:
  if r.get('split')!='train': raise ValueError('training file contains non-train record')
  if r['id'] in ids: raise ValueError('duplicate training ID')
  ids.add(r['id']); gold_gate(r)
 failures=fit_coverage_diagnostics(records)
 if failures: raise ValueError('Insufficient verified head coverage: '+'; '.join(failures))
 return load_dataset(path,split='train')

def fit_coverage_diagnostics(records):
 """LR needs at least two target classes; mathematical fitability is not quality."""
 classes={}
 for r in records:
  for field,values in compact_targets(r).items():
   values=values if isinstance(values,list) else [values]
   classes.setdefault(field,set()).update(values)
 return [field+' has fewer than two verified target classes' for field,values in classes.items() if len(values)<2] or ([] if records else ['no verified records'])

def exact_pair_features(examples,request_vectors,encoder):
 """Preserves per-example descriptor; structural parity with compact inference."""
 import numpy as np
 texts=[]; pairs=[]; targets=[]
 for i,e in enumerate(examples):
  for c in e.capabilities:
   texts.append(serialize_capability(c.name,c.description,c.required_arguments)); pairs.append(i); targets.append(int(c.name==e.expected.selected_capability))
 if not texts: raise ValueError('no capability pairs available')
 vectors=encoder.encode(texts,batch_size=16)
 features=np.stack([_pair_features(request_vectors[i],vec) for i,vec in zip(pairs,vectors)])
 return features,np.asarray(targets),texts

def explicit_reference_resolution(context,kind):
 """Proposal: use an explicit inference-time reference map; never gold targets.

Absent/ambiguous references stay unresolved instead of taking arbitrary scalars.
The real adapter is unchanged and gold resolved_value never enters input context.
"""
 entries=context.get('resolved_references',[])
 if not isinstance(entries,list): return None
 values={x.get('resolved_value') for x in entries if isinstance(x,dict) and x.get('kind')==kind and isinstance(x.get('resolved_value'),str) and x['resolved_value'].strip()}
 return next(iter(values)) if len(values)==1 else None

def mapping_proposals(label):
 """Partial semantic mappings only; deliberately never returns full gold."""
 proposals={}; notes=[]
 path=label.get('execution_path')
 safe={'respond':'direct_response','memory':'memory_recall','clarify':'clarify'}
 if path in safe: proposals['execution_path']=safe[path]; notes.append('workflow path semantic equivalence; requires exact-schema review')
 elif path=='research': notes.append('research workflow may be research or multi_step_task; no automatic mapping')
 elif path=='escalate': notes.append('new escalation includes cloud/current-info policy; old escalate means unavailable capability; review reason')
 else: notes.append('fast_path/qwen_agent cannot determine the old semantic execution path')
 if 'clarification' in label:
  proposals['needs_clarification']=label['clarification']['required']
  proposals['missing_information_names']=list(label['clarification']['missing'])
  notes.append('boolean/names only; descriptions and exact context references not supplied by new schema')
 notes.append('capability family is not selected capability; uncertainty is not compact confidence; modes/intent/ref targets need fresh review')
 return proposals,notes

class UnionFind:
 def __init__(self,ids): self.parent={x:x for x in ids}
 def root(self,x):
  if self.parent[x]!=x: self.parent[x]=self.root(self.parent[x])
  return self.parent[x]
 def union(self,a,b):
  a,b=self.root(a),self.root(b)
  if a!=b: self.parent[max(a,b)]=min(a,b)
