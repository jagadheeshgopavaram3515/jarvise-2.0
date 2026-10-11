import joblib
import numpy as np
import json
import hashlib
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'D:/JARVIS')
sys.path.insert(0, 'D:/JARVIS/reports/classifier_completion_20261010_163425')

from assistant.orchestration.compact_adapter import serialize_request, serialize_capability
from assistant.orchestration.decision import DecisionRequest, Capability
from experimental_adapter import pair_features

ROOT = Path('D:/JARVIS')
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
OUT = ROOT / 'reports/classifier_completion_20261010_163425'

cache = json.loads((PRIOR / 'feature_cache_manifest.json').read_text(encoding='utf-8'))
with np.load(PRIOR / 'frozen_features.npz') as a:
    vectors = a['vectors']
index = {x['text_sha256']: x['index'] for x in cache['texts']}
def encoded(text):
    return vectors[index[hashlib.sha256(text.encode()).hexdigest()]]

model_art = joblib.load(OUT / 'best_candidate/compact_heads.joblib')
matcher = model_art['capability_matcher']
style = model_art.get('pair_feature_style', 'interaction')
val = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

for i, r in enumerate(val[:10]):
    if 'archive' in r.get('original_id', ''):
        req = DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))
        q_vec = encoded(serialize_request(req))
        cap_vecs = [encoded(serialize_capability(c.name, c.description, c.required_arguments)) for c in req.capabilities]
        feats = np.stack([pair_features(q_vec, cv, style) for cv in cap_vecs])
        scores = matcher.predict_proba(feats)[:, list(matcher.classes_).index(1)]
        print(f"{r['original_id']} ({r['utterance']}):")
        for c, s in zip(req.capabilities, scores):
            print(f"   {c.name}: {s:.4f}")

