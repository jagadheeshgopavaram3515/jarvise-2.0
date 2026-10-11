import sys, json, joblib, hashlib, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

ROOT = Path('D:/JARVIS')
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
BUILD = ROOT / 'reports/multilingual_dataset_build_20261010_155129'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUILD))

from assistant.orchestration.decision import DecisionRequest, Capability
from assistant.orchestration.compact_adapter import serialize_request

cache = json.loads((PRIOR / 'feature_cache_manifest.json').read_text())
vectors = np.load(PRIOR / 'frozen_features.npz')['vectors']
index = {x['text_sha256']: x['index'] for x in cache['texts']}
def encoded(text): return vectors[index[hashlib.sha256(text.encode()).hexdigest()]]

train = [json.loads(l) for l in (PRIOR / 'provisional_train.jsonl').read_text(encoding='utf-8').splitlines()]
val = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]

def req(r):
    return DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))

X_train = np.stack([encoded(serialize_request(req(r))) for r in train])
X_val = np.stack([encoded(serialize_request(req(r))) for r in val])

for target_name in ['needs_clarification', 'needs_escalation']:
    y_train = [int(r['expected'][target_name]) for r in train]
    y_val = [int(r['expected'][target_name]) for r in val]
    print(f"\n--- {target_name} ---")
    for C in [1.0, 2.0, 4.0]:
        for weight in [None, 'balanced']:
            clf = LogisticRegression(C=C, class_weight=weight, max_iter=3000, random_state=42).fit(X_train, y_train)
            val_acc = clf.score(X_val, y_val)
            print(f"C={C:<4} weight={str(weight):<8} | train={clf.score(X_train, y_train):.3f} | val={val_acc:.3f} ({int(round(val_acc*40))}/40)")

