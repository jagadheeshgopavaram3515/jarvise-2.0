"""Check super().predict on frozen100-096."""
import json
import joblib
from pathlib import Path
ROOT = Path('D:/JARVIS')
EVAL_DIR = ROOT / 'reports/classifier_v4_frozen_100_eval_20261010_180014'
EXP4 = ROOT / 'reports/classifier_v4_experiment_20261010_175140'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP4))
from assistant.orchestration.decision import DecisionRequest, Capability
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder, CompactMultilingualAdapter

cache_manifest = json.loads((PRIOR / 'feature_cache_manifest.json').read_text(encoding='utf-8'))
encoder = FrozenMultilingualEncoder(cache_manifest['encoder_snapshot'], 160)
v4_artifact = joblib.load(EXP4 / 'best_candidate/compact_heads.joblib')

corpus = [json.loads(l) for l in (EVAL_DIR / 'unseen_evaluation_corpus_100.jsonl').read_text(encoding='utf-8').splitlines()]
r = next(x for x in corpus if x['id'] == 'frozen100-096')

req = DecisionRequest(
    r['utterance'],
    r['context'],
    tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities'])
)

adapter = CompactMultilingualAdapter.__new__(CompactMultilingualAdapter)
adapter.artifact = v4_artifact
adapter.encoder = encoder
adapter.inference_count = 0
adapter.last_inference_metrics = {}

raw = adapter.predict(req)
print("Base CompactMultilingualAdapter.predict on frozen100-096:")
print(json.dumps(dict(raw), indent=2))

