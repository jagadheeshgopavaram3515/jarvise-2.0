"""Focused offline checks of actual experiment outputs; no encoder/training."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,collections,unittest,types
import numpy as np,joblib
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];BUILD=ROOT/'reports/multilingual_dataset_build_20261010_155129'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BUILD))
pkg=types.ModuleType('assistant.orchestration');pkg.__path__=[str(ROOT/'assistant/orchestration')];sys.modules.setdefault('assistant.orchestration',pkg)
def js(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def jl(n):return [json.loads(l) for l in (OUT/n).read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
class IntegrityTests(unittest.TestCase):
 def test_fixed_validation_denominator(self):
  n=len(jl('provisional_validation.jsonl'));self.assertEqual(n,40)
  for p in OUT.glob('experiment_*/metrics.json'):self.assertEqual(json.loads(p.read_text(encoding='utf-8'))['exact_full_record']['denominator'],n)
 def test_training_authorization_status(self):
  self.assertTrue(all(r['dataset_state']=='synthetic_unverified' for r in jl('provisional_train.jsonl')))
 def test_no_protected_records(self):
  protected=set(json.loads((BUILD/'protected_test_manifest.json').read_text())['original_protected_record_ids'])
  for r in jl('provisional_train.jsonl')+jl('provisional_validation.jsonl'):self.assertNotIn(r['original_id'],protected)
 def test_no_group_crossing(self):
  self.assertFalse({r['group_id'] for r in jl('provisional_train.jsonl')}&{r['group_id'] for r in jl('provisional_validation.jsonl')})
 def test_baseline_training_id_excluded_from_validation(self):
  ids=set(json.loads((ROOT/'benchmarks/fast_classifier/models/v3/compact_heads.json').read_text())['training_example_ids'])
  self.assertFalse(ids&{r['original_id'] for r in jl('provisional_validation.jsonl')})
 def test_feature_parity_and_actual_encoder_agree(self):
  p=js('feature_parity_report.json');self.assertEqual(p['status'],'PASS');self.assertTrue(p['actual_encoder_vs_cached_adapter_prediction_parity']);self.assertEqual(p['request_feature_dimensions'],768);self.assertEqual(p['pair_feature_dimensions'],3072)
 def test_features_finite(self):
  with np.load(OUT/'frozen_features.npz') as x:self.assertTrue(np.isfinite(x['vectors']).all());self.assertEqual(x['vectors'].shape[1],768)
 def test_all_metrics_provisional(self):
  self.assertEqual(js('baseline_metrics.json')['gold_verified_accuracy']['denominator'],0)
  self.assertEqual(js('best_candidate_metadata.json')['gold_verified_accuracy'],'NOT_MEASURED')
 def test_strict_exact_score_recomputes(self):
  p=js('best_candidate_metadata.json');m=js(p['selected_experiment']+'/metrics.json');fields=('intent','execution_path','request_modes','selected_capability','context_references','missing_information','needs_clarification','needs_escalation')
  count=0
  for r in m['predictions']:
   pred=r['prediction'];gold=r['expected'];match=pred is not None and all((sorted(pred[f])==sorted(gold[f]) if f=='request_modes' else pred[f]==gold[f]) for f in fields);self.assertEqual(match,r['exact_match']);count+=match
  self.assertEqual(count,m['exact_full_record']['agreement'])
 def test_every_model_new_and_finite(self):
  for path in OUT.glob('experiment_*/compact_heads.joblib'):
   a=joblib.load(path)
   self.assertEqual(a['capability_encoding'],'independent');self.assertEqual(a['training_examples'],len(jl('provisional_train.jsonl')))
   for h in [a[k] for k in ('execution_path','clarification','escalation','missing_information','reference_kind','reference_resolved','capability_matcher')]+list(a['request_modes'].values()):self.assertTrue(np.isfinite(h.coef_).all());self.assertGreaterEqual(len(h.classes_),2)
 def test_saved_best_hash(self):self.assertEqual(sha(OUT/'best_candidate/compact_heads.joblib'),js('best_candidate_metadata.json')['best_model_sha256'])
 def test_original_inputs_unchanged(self):
  for p,info in js('reproducibility_manifest.json')['input_hashes'].items():self.assertEqual(sha(p),info['sha256'])
 def test_bound_and_stop_condition(self):
  m=js('best_candidate_metadata.json');self.assertLessEqual(m['experiments_run'],20);self.assertIn(m['stop_reason'],('THREE_CONSECUTIVE_PRIMARY_PLATEAU_EXPERIMENTS','PROVISIONAL_TARGET_REACHED','TWENTY_EXPERIMENT_LIMIT'))
if __name__=='__main__':unittest.main(verbosity=2)
