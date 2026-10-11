"""Evaluate v5 on the 24-record unseen corpus to verify independent generalization."""
import json
import joblib
import numpy as np
from pathlib import Path

ROOT = Path('D:/JARVIS')
EXP_DIR = ROOT / 'reports/jarvis_brain_forensic_improvement_20261010_183000'
EXP4 = ROOT / 'reports/classifier_v4_experiment_20261010_175140'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))

from assistant.orchestration.decision import Capability, DecisionRequest, ClassifierDecision
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder
from experimental_adapter_v5 import ExperimentalV5Adapter

FIELDS = (
    'intent', 'execution_path', 'request_modes', 'selected_capability',
    'needs_clarification', 'needs_escalation', 'missing_information', 'context_references'
)

cache_manifest = json.loads((PRIOR / 'feature_cache_manifest.json').read_text(encoding='utf-8'))
encoder = FrozenMultilingualEncoder(cache_manifest['encoder_snapshot'], 160)
artifact = joblib.load(EXP4 / 'best_candidate/compact_heads.joblib')

adapter = ExperimentalV5Adapter.__new__(ExperimentalV5Adapter)
adapter.artifact = artifact
adapter.encoder = encoder
adapter._adapter_id = 'test-v5-24'
adapter.inference_count = 0
adapter.last_inference_metrics = {}

corpus_24 = [json.loads(l) for l in (EXP4 / 'unseen_evaluation_corpus_v4.jsonl').read_text(encoding='utf-8').splitlines()]

exact_matches = 0
cap_matches = 0

for r in corpus_24:
    req = DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))
    pred = dict(adapter.predict(req))
    ClassifierDecision.from_mapping(pred)
    exp = r['expected']
    is_exact = all((sorted(pred[k]) == sorted(exp[k]) if k == 'request_modes' else pred[k] == exp[k]) for k in FIELDS)
    is_cap = pred['selected_capability'] == exp['selected_capability']
    exact_matches += int(is_exact)
    cap_matches += int(is_cap)

results = {
    'denominator': len(corpus_24),
    'v5_exact': exact_matches,
    'v5_exact_pct': exact_matches / len(corpus_24) * 100.0,
    'v5_capability': cap_matches,
    'v5_capability_pct': cap_matches / len(corpus_24) * 100.0,
}

(EXP_DIR / 'independent_24_results_v5.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print("24-Record Unseen Corpus Results:")
print(f"  v4 baseline exact: 12 / 24 (50.0%) | v4 capability: 19 / 24 (79.2%)")
print(f"  v5 improved exact: {exact_matches} / 24 ({exact_matches/len(corpus_24)*100:.1f}%) | v5 capability: {cap_matches} / 24 ({cap_matches/len(corpus_24)*100:.1f}%)")

