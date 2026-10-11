"""Synthetic unit fixtures only; never eligible real human training labels."""
import sys
sys.dont_write_bytecode=True
import copy,json,unittest
from pathlib import Path
import numpy as np
from dataset_tools import *
from assistant.orchestration.decision import Capability
from assistant.orchestration.compact_adapter import _pair_features,serialize_request

def example():
 return {'schema_version':'3.0','id':'unit-test-only','split':'train','utterance':'Open Notepad.',
 'context':{},'capabilities':[{'name':'open_app','description':'Open an application.','required_arguments':['app']}],
 'expected':{'intent':'use:open_app','execution_path':'single_action','request_modes':['action'],'selected_capability':'open_app','context_references':[],'missing_information':[],'needs_clarification':False,'needs_escalation':False},
 'language':'en','categories':['unit_test_fixture'],'annotation_status':'reviewed','review_status':'verified_gold','provenance':'unit_test_fixture_not_real_gold','source':'unit_fixture','notes':'Synthetic unit fixture, never real human labels.',
 'group_id':'unit-fixture','dataset_state':'verified_gold','language_metadata':{'required_languages':['english'],'transliteration_expertise_required':False}}
def fixture_approvals(r):
 rationale={k:'Synthetic test rationale, not a real human approval.' for k in r['expected']}
 r['verification']={'contract':'ClassifierDecision','policy':'compact-two-reviewer-v1','record_sha256':verification_hash(r),'reviewers':[{'reviewer_id':who,'reviewer_type':'human','review_kind':'independent','prediction_blinded':True,'decision':'approve','target_sha256':fingerprint(r['expected']),'qualification':{'languages':['english'],'scripts':['latin'],'transliteration':False},'timestamp':'2026-10-10T15:00:00+05:30','field_rationale':rationale} for who in ('UNIT_FIXTURE_A','UNIT_FIXTURE_B')]}
 return r

class BuilderTests(unittest.TestCase):
 def test_structured_missing_preserved(self):
  x=[{'name':'target','description':'Unresolved requested target'}]
  self.assertEqual(extract_missing(x),x); self.assertEqual(extract_missing(x,taxonomy='fast_brain'),x)
 def test_prior_request_not_silently_remapped(self):
  x=[{'name':'prior_request','description':'Prior utterance unavailable'}]
  self.assertEqual(extract_missing(x)[0]['name'],'prior_request')
  with self.assertRaisesRegex(ValueError,'unknown FastBrain enum'): extract_missing(x,taxonomy='fast_brain')
 def test_unknown_structured_fields_rejected(self):
  with self.assertRaisesRegex(ValueError,'needs name'): extract_missing([{'kind':'target'}])
 def test_invalid_missing_type_rejected(self):
  with self.assertRaisesRegex(ValueError,'expected string'): extract_missing([17])
 def test_all_registered_families_mapped(self):
  for name in registry_specs(): self.assertIn(capability_family(name),{x.value for x in CapabilityFamily})
 def test_omitted_actions_fixed(self):
  for name,family in [('open_folder','file'),('recent_files','file'),('grep_code','project'),('run_project_tests','project')]: self.assertEqual(capability_family(name),family)
 def test_unknown_capability_fails(self):
  with self.assertRaisesRegex(ValueError,'Unmapped'): capability_family('unsupported_unit_fixture')
 def test_registry_required_arguments(self):
  specs=registry_specs()
  for name,req in [('weather',[]),('search_files',['keyword']),('list_directory',[]),('run_project_tests',[])]: self.assertEqual(specs[name]['required_arguments'],req)
 def test_argument_alignment_preserves_original(self):
  caps=[{'name':'search_files','description':'Original description','required_arguments':['query']}]; before=copy.deepcopy(caps)
  result,changes=aligned_capabilities(caps,registry_specs())
  self.assertEqual(caps,before); self.assertEqual(result[0]['required_arguments'],['keyword']); self.assertEqual(result[0]['description'],'Original description'); self.assertEqual(len(changes),1)
 def test_per_example_descriptor_feature_parity(self):
  a=example(); b=example(); b['id']='unit-second'; b['capabilities'][0]['description']='Different descriptor for the same registered name.'
  examples=[validate_candidate(a),validate_candidate(b)]
  class FakeEncoder:
   def encode(self,texts,**kw):
    self.texts=texts; return np.array([[len(t),sum(map(ord,t))%997] for t in texts],dtype=float)
  fake=FakeEncoder(); request=np.array([[1.,2.],[3.,4.]])
  features,targets,texts=exact_pair_features(examples,request,fake)
  self.assertNotEqual(texts[0],texts[1]); self.assertEqual(features.shape,(2,8)); self.assertEqual(targets.tolist(),[1,1])
  for i,e in enumerate(examples):
   c=e.capabilities[0]; infer=fake.encode([serialize_capability(c.name,c.description,c.required_arguments)])[0]
   np.testing.assert_array_equal(features[i],_pair_features(request[i],infer))
 def test_original_compact_schema(self): validate_candidate(example())
 def test_new_path_not_forced_into_old_contract(self):
  r=example(); r['expected']['execution_path']='qwen_agent'
  with self.assertRaisesRegex(ValueError,'unknown compact'): validate_candidate(r)
 def test_newer_family_not_tool_name(self):
  proposed,notes=mapping_proposals({'execution_path':'qwen_agent','capability_family':'file'})
  self.assertNotIn('selected_capability',proposed); self.assertNotIn('execution_path',proposed)
 def test_clarification_mapping_is_partial_only(self):
  proposed,notes=mapping_proposals({'execution_path':'clarify','clarification':{'required':True,'missing':['target']}})
  self.assertEqual(proposed['execution_path'],'clarify'); self.assertEqual(proposed['missing_information_names'],['target']); self.assertNotIn('intent',proposed)
 def test_single_reviewer_not_gold(self):
  r=fixture_approvals(example()); r['verification']['reviewers']=r['verification']['reviewers'][:1]
  with self.assertRaisesRegex(ValueError,'two distinct'): gold_gate(r)
 def test_gold_shape_alone_does_not_make_heads_fittable(self):
  self.assertTrue(fit_coverage_diagnostics([fixture_approvals(example())]))
 def test_synthetic_state_not_gold(self):
  r=fixture_approvals(example()); r['dataset_state']='synthetic_unverified'
  with self.assertRaisesRegex(ValueError,'only verified_gold'): gold_gate(r)
 def test_verified_hash_is_bound_to_targets_and_context(self):
  r=fixture_approvals(example()); r['context']={'active_task':'changed'}
  with self.assertRaisesRegex(ValueError,'hash mismatch'): gold_gate(r)
 def test_skipped_or_disputed_not_gold(self):
  r=fixture_approvals(example()); r['verification']['reviewers'][0]['decision']='skip'
  with self.assertRaisesRegex(ValueError,'approval missing'): gold_gate(r)
 def test_duplicate_reviewer_not_gold(self):
  r=fixture_approvals(example()); r['verification']['reviewers'][1]['reviewer_id']='UNIT_FIXTURE_A'
  with self.assertRaisesRegex(ValueError,'distinct'): gold_gate(r)
 def test_protected_test_never_fits(self):
  r=fixture_approvals(example()); r['original_test_lineage']=True
  with self.assertRaisesRegex(ValueError,'sealed/test lineage'): gold_gate(r)
 def test_exact_schema_approval_gate_positive_unit_fixture(self): self.assertTrue(gold_gate(fixture_approvals(example())))
 def test_no_arbitrary_context_first_scalar(self):
  self.assertIsNone(explicit_reference_resolution({'recent_topics':[{'topic':'meeting','decision':'ship API'}]},'prior_conversation'))
 def test_explicit_reference_map_only(self):
  c={'unrelated':'incorrect first scalar','resolved_references':[{'kind':'item','resolved_value':'document'}]}
  self.assertEqual(explicit_reference_resolution(c,'item'),'document')
 def test_ambiguous_reference_map_returns_none(self):
  c={'resolved_references':[{'kind':'item','resolved_value':'a'},{'kind':'item','resolved_value':'b'}]}
  self.assertIsNone(explicit_reference_resolution(c,'item'))
 def test_gold_targets_never_input(self):
  r=example(); e=validate_candidate(r); request=DecisionRequest(e.utterance,e.context,e.capabilities)
  before=serialize_request(request); r['expected']['intent']='UNIT_GOLD_SECRET'; self.assertEqual(before,serialize_request(request)); self.assertNotIn('UNIT_GOLD_SECRET',before)
 def test_normalization_no_translation_or_script_collapse(self):
  self.assertNotEqual(normalized('తెలుగు'),normalized('telugu')); self.assertEqual(normalized(' Open   FILE '),'open file')
 def test_transitive_group_union(self):
  u=UnionFind(['a','b','c']); u.union('a','b'); u.union('b','c'); self.assertEqual(u.root('a'),u.root('c'))

class BuiltDatasetTests(unittest.TestCase):
 def test_all_candidate_files_parse_actual_loader(self):
  root=Path(__file__).resolve().parent
  for name in ('pending_review.jsonl','synthetic_unverified.jsonl'):
   raw=[json.loads(l) for l in (root/name).read_text(encoding='utf-8').splitlines() if l.strip()]
   for r in raw: validate_candidate(r); self.assertFalse(r['original_test_lineage']); self.assertEqual(r['context'],r['original_context'])
   if raw: self.assertEqual(len(load_dataset(root/name).examples),len(raw))
 def test_supervised_empty_and_guard_blocks_before_encoder(self):
  root=Path(__file__).resolve().parent
  for name in ('train.jsonl','validation.jsonl','sealed_test.jsonl'): self.assertEqual((root/name).read_bytes(),b'')
  with self.assertRaisesRegex(ValueError,'training blocked'): load_verified_training(root/'train.jsonl')
 def test_candidates_rejected_by_training_gate(self):
  root=Path(__file__).resolve().parent
  with self.assertRaisesRegex(ValueError,'non-train'): load_verified_training(root/'synthetic_unverified.jsonl')
 def test_downloads_test_duplicate_not_candidate(self):
  root=Path(__file__).resolve().parent; ids=set()
  for name in ('pending_review.jsonl','synthetic_unverified.jsonl'):
   ids.update(json.loads(l)['original_id'] for l in (root/name).read_text(encoding='utf-8').splitlines())
  self.assertNotIn('v4_open_downloads-en',ids); self.assertIn('train_open_downloads-en',ids)
 def test_each_scenario_has_one_candidate_group(self):
  root=Path(__file__).resolve().parent; mapping={}
  for name in ('pending_review.jsonl','synthetic_unverified.jsonl'):
   for line in (root/name).read_text(encoding='utf-8').splitlines():
    r=json.loads(line); prior=mapping.setdefault(r['original_group_id'],r['group_id']); self.assertEqual(prior,r['group_id'])

if __name__=='__main__': unittest.main(verbosity=2)
