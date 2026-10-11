"""Inspect frozen100-007 in v5."""
import json
import joblib
from pathlib import Path

ROOT = Path('D:/JARVIS')
EVAL_DIR = ROOT / 'reports/classifier_v4_frozen_100_eval_20261010_180014'
EXP4 = ROOT / 'reports/classifier_v4_experiment_20261010_175140'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
OUT_DIR = ROOT / 'reports/jarvis_brain_forensic_improvement_20261010_183000'

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP4))
sys.path.insert(0, str(OUT_DIR))

from assistant.orchestration.decision import Capability, DecisionRequest
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder
from experimental_adapter_v5 import ExperimentalV5Adapter

cache_manifest = json.loads((PRIOR / 'feature_cache_manifest.json').read_text(encoding='utf-8'))
encoder = FrozenMultilingualEncoder(cache_manifest['encoder_snapshot'], 160)
artifact = joblib.load(EXP4 / 'best_candidate/compact_heads.joblib')

adapter = ExperimentalV5Adapter.__new__(ExperimentalV5Adapter)
adapter.artifact = artifact
adapter.encoder = encoder
adapter._adapter_id = 'test-v5'
adapter.inference_count = 0
adapter.last_inference_metrics = {}

corpus = [json.loads(l) for l in (EVAL_DIR / 'unseen_evaluation_corpus_100.jsonl').read_text(encoding='utf-8').splitlines()]
r = next(x for x in corpus if x['id'] == 'frozen100-007')

req = DecisionRequest(
    r['utterance'],
    r['context'],
    tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities'])
)

pred = adapter.predict(req)
print(f"ID: {r['id']}")
print(f"Diagnostics: {adapter.last_diagnostics}")
print(f"Prediction: {pred['selected_capability']}, Path: {pred['execution_path']}")
for c, s in zip(r['capabilities'], adapter.last_diagnostics.get('capability_scores', [])):
    print(f"   {c['name']:<20}: {s:.4f}")

