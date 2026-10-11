import sys, json, joblib, hashlib, numpy as np
from pathlib import Path

ROOT = Path('D:/JARVIS')
COMP = ROOT / 'reports/classifier_completion_20261010_163425'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
BUILD = ROOT / 'reports/multilingual_dataset_build_20261010_155129'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUILD))

from assistant.orchestration.decision import DecisionRequest, Capability
from assistant.orchestration.compact_adapter import serialize_request, serialize_capability

heads = joblib.load(COMP / 'best_candidate/compact_heads.joblib')
cache = json.loads((PRIOR / 'feature_cache_manifest.json').read_text())
vectors = np.load(PRIOR / 'frozen_features.npz')['vectors']
index = {x['text_sha256']: x['index'] for x in cache['texts']}
def encoded(text): return vectors[index[hashlib.sha256(text.encode()).hexdigest()]]

VAL = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]
for r in VAL:
    if r['original_id'] == 'ctx-hologram-001':
        req = DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))
        vec = encoded(serialize_request(req)).reshape(1, -1)
        for m, h in heads['request_modes'].items():
            prob = h.predict_proba(vec)[0]
            print(f"Mode {m}: classes={h.classes_}, prob={prob}")
