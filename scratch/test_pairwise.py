import json
import hashlib
import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path('D:/JARVIS')
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'reports/classifier_completion_20261010_163425'))

from assistant.orchestration.compact_adapter import serialize_request, serialize_capability
from assistant.orchestration.decision import DecisionRequest, Capability
from experimental_adapter import pair_features

cache = json.loads((PRIOR / 'feature_cache_manifest.json').read_text(encoding='utf-8'))
with np.load(PRIOR / 'frozen_features.npz') as a:
    vectors = a['vectors']
index = {x['text_sha256']: x['index'] for x in cache['texts']}
def encoded(text):
    return vectors[index[hashlib.sha256(text.encode()).hexdigest()]]

train = [json.loads(l) for l in (PRIOR / 'provisional_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
val = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

def get_cap_vec(c):
    return encoded(serialize_capability(c['name'], c['description'], c['required_arguments']))

# Let's test pairwise ranking data:
# For each training example with a selected capability:
# For each negative candidate in that example:
# feature = pair_feat(q, c_pos) - pair_feat(q, c_neg) with label +1
# and -(pair_feat(q, c_pos) - pair_feat(q, c_neg)) with label 0

for style in ['interaction', 'full']:
    X_rank = []
    y_rank = []
    for r in train:
        pos_cap = r['expected']['selected_capability']
        if not pos_cap or not r['capabilities']:
            continue
        req = DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))
        q_vec = encoded(serialize_request(req))
        
        pos_obj = next((c for c in r['capabilities'] if c['name'] == pos_cap), None)
        if not pos_obj:
            continue
        pos_vec = get_cap_vec(pos_obj)
        pos_feat = pair_features(q_vec, pos_vec, style)
        
        for c in r['capabilities']:
            if c['name'] != pos_cap:
                neg_vec = get_cap_vec(c)
                neg_feat = pair_features(q_vec, neg_vec, style)
                diff = pos_feat - neg_feat
                X_rank.append(diff)
                y_rank.append(1)
                X_rank.append(-diff)
                y_rank.append(0)

    X_rank = np.array(X_rank)
    y_rank = np.array(y_rank)
    print(f"Style: {style}, Pairwise training samples: {len(X_rank)}")
    
    for C in [0.01, 0.1, 1.0, 4.0, 10.0]:
        # Fit ranker without intercept since it's symmetric
        ranker = LogisticRegression(C=C, fit_intercept=False, max_iter=2000, random_state=42)
        ranker.fit(X_rank, y_rank)
        w = ranker.coef_[0]
        
        # Test on validation
        correct = 0
        total = 0
        archive_correct = 0
        for r in val:
            exp_cap = r['expected']['selected_capability']
            if not exp_cap or not r['capabilities']:
                continue
            total += 1
            req = DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))
            q_vec = encoded(serialize_request(req))
            scores = []
            for c in r['capabilities']:
                c_vec = get_cap_vec(c)
                feat = pair_features(q_vec, c_vec, style)
                score = float(np.dot(w, feat))
                scores.append((c['name'], score))
            scores.sort(key=lambda x: x[1], reverse=True)
            if scores[0][0] == exp_cap:
                correct += 1
                if 'archive' in r.get('original_id', ''):
                    archive_correct += 1
        print(f"  C={C}: Val Top-1 Accuracy: {correct}/{total} (archive: {archive_correct}/20)")
