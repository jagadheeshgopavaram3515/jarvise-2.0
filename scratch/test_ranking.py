import json
import hashlib
import numpy as np
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path('D:/JARVIS')
sys.path.insert(0, str(ROOT))
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'

from assistant.orchestration.compact_adapter import serialize_request, serialize_capability
from assistant.orchestration.decision import DecisionRequest, Capability

cache = json.loads((PRIOR / 'feature_cache_manifest.json').read_text(encoding='utf-8'))
with np.load(PRIOR / 'frozen_features.npz') as a:
    vectors = a['vectors']
index = {x['text_sha256']: x['index'] for x in cache['texts']}
def encoded(text):
    return vectors[index[hashlib.sha256(text.encode()).hexdigest()]]

val = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

top1_correct = 0
top3_correct = 0
total_with_cap = 0
action_path_count = 0

print("Testing Candidate Ranking on 40 Validation Records:")
for i, r in enumerate(val):
    req = DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))
    q_vec = encoded(serialize_request(req))
    expected_cap = r['expected']['selected_capability']
    path = r['expected']['execution_path']
    
    if req.capabilities:
        sims = []
        for c in req.capabilities:
            c_text = serialize_capability(c.name, c.description, c.required_arguments)
            c_vec = encoded(c_text)
            sim = float(np.dot(q_vec, c_vec))
            sims.append((c.name, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        top1 = sims[0][0]
        top3 = [x[0] for x in sims[:3]]
    else:
        top1 = None
        top3 = []

    if expected_cap is not None:
        total_with_cap += 1
        is_top1 = (top1 == expected_cap)
        is_top3 = (expected_cap in top3)
        if is_top1: top1_correct += 1
        if is_top3: top3_correct += 1
        print(f"[{i:02d}] {r['original_id']} | path={path} | exp={expected_cap} | top1={top1} ({is_top1}) | top3={top3} | scores={[(n, round(s, 3)) for n, s in sims]}")
    else:
        print(f"[{i:02d}] {r['original_id']} | path={path} | exp=None | top1={top1} | top_score={sims[0][1] if req.capabilities else 'no-caps'}")

print("\n" + "="*80)
print(f"Top-1 Accuracy on records with expected capability: {top1_correct}/{total_with_cap}")
print(f"Top-3 Accuracy on records with expected capability: {top3_correct}/{total_with_cap}")
