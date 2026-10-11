import json
import hashlib
import numpy as np
import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path('D:/JARVIS')
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
sys.path.insert(0, str(ROOT))

from assistant.orchestration.compact_adapter import serialize_request, serialize_capability
from assistant.orchestration.decision import DecisionRequest, Capability

cache = json.loads((PRIOR / 'feature_cache_manifest.json').read_text(encoding='utf-8'))
with np.load(PRIOR / 'frozen_features.npz') as a:
    vectors = a['vectors']
index = {x['text_sha256']: x['index'] for x in cache['texts']}
def encoded(text):
    return vectors[index[hashlib.sha256(text.encode()).hexdigest()]]

train = [json.loads(l) for l in (PRIOR / 'provisional_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
val = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

def evaluate_ranker(score_fn, split_data, split_name):
    correct = 0
    total = 0
    archive_correct = 0
    for r in split_data:
        exp_cap = r['expected']['selected_capability']
        path = r['expected']['execution_path']
        # Only evaluate on records where execution path is an action path or has expected cap
        if not r['capabilities'] or exp_cap is None:
            continue
        total += 1
        req = DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))
        q_vec = encoded(serialize_request(req))
        scores = []
        for c in req.capabilities:
            c_vec = encoded(serialize_capability(c.name, c.description, c.required_arguments))
            s = score_fn(r['utterance'], q_vec, c, c_vec)
            scores.append((c.name, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        is_correct = (scores[0][0] == exp_cap)
        if is_correct:
            correct += 1
            if 'archive' in r.get('original_id', ''):
                archive_correct += 1
    acc = correct / total if total else 0
    print(f"[{split_name}] Top-1 Accuracy: {correct}/{total} ({acc*100:.1f}%) | archive: {archive_correct}/20")
    return correct, total

# Base score function: purely cosine similarity of serialized capability
print("1. Pure Cosine Similarity:")
evaluate_ranker(lambda u, qv, c, cv: float(np.dot(qv, cv)), train, "TRAIN")
evaluate_ranker(lambda u, qv, c, cv: float(np.dot(qv, cv)), val, "VAL")

