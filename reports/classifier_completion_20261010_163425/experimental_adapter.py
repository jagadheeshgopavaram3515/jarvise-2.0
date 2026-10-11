"""Compact contract v2 experiment; no production import/integration.

Rendering heads learn only training targets. Typed resolution uses actual
inference context; no gold values, example IDs or utterance-specific exceptions.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import re,time
import numpy as np
from assistant.orchestration.compact_adapter import CompactMultilingualAdapter,_pair_features,_positive_probability,_compose_intent
from assistant.orchestration.decision import ClassifierDecision

def pair_features(left,right,style='full'):
 if style=='full':return _pair_features(left,right)
 if style=='interaction':return np.concatenate([np.abs(left-right),left*right])
 raise ValueError('unsupported pair feature style')

def typed_resolve(context,kind,user_text):
 """Resolve a unique typed object/decision; never arbitrary dict traversal."""
 candidates=[];text=user_text.casefold()
 if kind=='prior_conversation':
  for key in ('recent_topics','projects'):
   for item in context.get(key,[]) if isinstance(context.get(key,[]),list) else []:
    if not isinstance(item,dict) or not isinstance(item.get('decision'),str):continue
    names=[item.get('topic',''),item.get('name',''),*item.get('aliases',[])]
    matched=any(isinstance(n,str) and n and re.search(r'(?<!\w)'+re.escape(n.casefold())+r'(?!\w)',text) for n in names)
    candidates.append((matched,item['decision']))
 elif kind=='project':
  for item in context.get('projects',[]) if isinstance(context.get('projects',[]),list) else []:
   if not isinstance(item,dict):continue
   names=[item.get('name',''),*item.get('aliases',[])]
   if isinstance(item.get('name'),str) and any(isinstance(n,str) and n and re.search(r'(?<!\w)'+re.escape(n.casefold())+r'(?!\w)',text) for n in names):candidates.append((True,item['name']))
 for item in context.get('resolved_references',[]) if isinstance(context.get('resolved_references',[]),list) else []:
  if isinstance(item,dict) and item.get('kind')==kind and isinstance(item.get('resolved_value'),str):candidates.append((True,item['resolved_value']))
 matched={v for m,v in candidates if m};values=matched or {v for _,v in candidates}
 return next(iter(values)) if len(values)==1 else None

def project_reference(context,text):
 """Recognize an explicitly mentioned project alias without inventing context."""
 hits=[]
 for item in context.get('projects',[]) if isinstance(context.get('projects',[]),list) else []:
  if not isinstance(item,dict) or item.get('decision') is not None:continue
  for alias in [item.get('name',''),*item.get('aliases',[])]:
   if isinstance(alias,str) and alias:
    match=re.search(r'\b(?:(?:our|my|the|this)\s+)?'+re.escape(alias)+r'\b',text,re.IGNORECASE)
    if match:hits.append(match.group(0))
 return hits[0] if len(set(hits))==1 else None

def safe_fallback(reason,confidence=0.):
 raw={'intent':'unsupported_request','execution_path':'escalate','request_modes':['action'],'selected_capability':None,'context_references':[],'missing_information':[],'confidence':float(confidence),'needs_clarification':False,'needs_escalation':True,'evidence':[],'abstained':True,'abstention_reason':reason}
 ClassifierDecision.from_mapping(raw);return raw

class ExperimentalCompactAdapter(CompactMultilingualAdapter):
 def predict(self,request):
  self.supports=[];self.last_diagnostics={}
  try:
   raw=dict(super().predict(request));self.last_diagnostics['postprocessed_path']=raw['execution_path']
   if raw['needs_clarification']:
    vector=self.last_request_vector
    name=str(self.artifact['missing_information'].predict(vector)[0])
    if name=='__none__':return safe_fallback('missing_information_head_has_no_supported_name')
    model=self.artifact['missing_description'];desc=str(model.predict(vector)[0]);self.supports.append(float(np.max(model.predict_proba(vector))))
    if not desc.strip():return safe_fallback('empty_missing_description')
    raw['missing_information']=[{'name':name,'description':desc}]
   if raw['selected_capability'] is not None:self.supports.append(max(self.last_diagnostics.get('capability_scores',[0.])))
   self.supports.append(raw['confidence']);raw['confidence']=min(self.supports)
   raw['abstained']=False;ClassifierDecision.from_mapping(raw);return raw
  except (ValueError,KeyError,TypeError,IndexError) as error:
   self.last_diagnostics['error']=str(error);return safe_fallback('invalid_or_unrepresentable_output:'+str(error))

 def _select_capability(self,request,request_vector,capability_vectors):
  if not request.capabilities:return None,0.
  model=self.artifact['capability_matcher']
  features=np.stack([pair_features(request_vector,v,self.artifact.get('pair_feature_style','full')) for v in capability_vectors])
  if 1 not in list(model.classes_):raise ValueError('capability matcher lacks positive class')
  scores=model.predict_proba(features)[:,list(model.classes_).index(1)];best=int(np.argmax(scores));score=float(scores[best]);self.last_diagnostics['capability_scores']=scores.tolist()
  return (request.capabilities[best].name if score>=self.artifact['capability_threshold'] else None),score

 def _predict_references(self,request,vector):
  self.last_request_vector=vector
  model=self.artifact['reference_kind'];kind=str(model.predict(vector)[0]);self.supports.append(float(np.max(model.predict_proba(vector))))
  alias=project_reference(request.context,request.user_text)
  if alias:kind='project'
  if kind=='__none__':return []
  span_model=self.artifact['reference_span'];span=str(span_model.predict(vector)[0]);self.supports.append(float(np.max(span_model.predict_proba(vector))))
  if alias:span=alias
  elif span=='__none__':raise ValueError('reference kind predicted but reference span head is unresolved')
  elif span=='__utterance__':span=request.user_text
  value=typed_resolve(request.context,kind,request.user_text) if _positive_probability(self.artifact['reference_resolved'],vector)>=.5 else None
  return [{'reference':span,'kind':kind,'resolved_value':value}]

 def predict_with_abstention(self,request,threshold):
  if not 0<=threshold<=1:raise ValueError('threshold must be within [0,1]')
  raw=self.predict(request)
  return safe_fallback('low_joint_field_support',raw['confidence']) if raw['confidence']<threshold else raw
