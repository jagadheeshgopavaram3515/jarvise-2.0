import json
import hashlib
import numpy as np
import sys
import re
from pathlib import Path
from sklearn.linear_model import LogisticRegression

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

def extract_candidate_features(utterance, q_vec, cap, cap_vec):
    # 1. Dense BERT similarity
    sim = float(np.dot(q_vec, cap_vec))
    
    # 2. Tokenize utterance
    tokens = set(re.findall(r'\w+', utterance.lower()))
    
    # 3. Name tokens
    name_tokens = set(cap.name.lower().split('_'))
    name_overlap = len(tokens & name_tokens)
    
    # 4. Description tokens
    desc_tokens = set(re.findall(r'\w+', cap.description.lower())) - {'a', 'an', 'the', 'in', 'for', 'with', 'by', 'its'}
    desc_overlap = len(tokens & desc_tokens)
    
    # 5. Arg tokens
    arg_tokens = set(cap.required_arguments)
    arg_overlap = len(tokens & arg_tokens)
    
    # 6. Entity type semantic matching:
    # Does request mention folder/directory and capability operates on folder?
    has_folder_word = bool(tokens & {'folder', 'directory', 'directories', 'folders'})
    is_folder_cap = ('folder' in cap.name or 'folder' in cap.description)
    folder_match = 1.0 if (has_folder_word and is_folder_cap) else (-1.0 if (has_folder_word and not is_folder_cap) else 0.0)
    
    # Does request mention file and capability operates on file?
    has_file_word = bool(tokens & {'file', 'files', 'spreadsheet', 'workbook', 'document', 'pdf', 'notes'}) and not has_folder_word
    is_file_cap = ('file' in cap.name or 'file' in cap.description) and not is_folder_cap
    file_match = 1.0 if (has_file_word and is_file_cap) else 0.0

    # Required argument feasibility: e.g. URL required but no URL present
    has_url = bool(re.search(r'https?://|www\.|\.org|\.com|\.io|\burl\b', utterance.lower()))
    url_feasible = 0.0
    if 'url' in cap.required_arguments and not has_url:
        url_feasible = -1.0
    
    return np.array([sim, name_overlap, desc_overlap, arg_overlap, folder_match, file_match, url_feasible], dtype=np.float32)

# Train a ranker on TRAIN pairwise differences
X_train = []
y_train = []

for r in train:
    exp_cap = r['expected']['selected_capability']
    if not exp_cap or not r['capabilities']:
        continue
    req = DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))
    q_vec = encoded(serialize_request(req))
    
    pos_cap = next((c for c in req.capabilities if c.name == exp_cap), None)
    if not pos_cap:
        continue
    pos_vec = encoded(serialize_capability(pos_cap.name, pos_cap.description, pos_cap.required_arguments))
    pos_feats = extract_candidate_features(r['utterance'], q_vec, pos_cap, pos_vec)
    
    for c in req.capabilities:
        if c.name != exp_cap:
            neg_vec = encoded(serialize_capability(c.name, c.description, c.required_arguments))
            neg_feats = extract_candidate_features(r['utterance'], q_vec, c, neg_vec)
            diff = pos_feats - neg_feats
            X_train.append(diff)
            y_train.append(1)
            X_train.append(-diff)
            y_train.append(0)

X_train = np.array(X_train)
y_train = np.array(y_train)
print(f"Training pairwise ranker on {len(X_train)} samples, feature dim = {X_train.shape[1]}")

ranker = LogisticRegression(C=1.0, fit_intercept=False, random_state=42)
ranker.fit(X_train, y_train)
w = ranker.coef_[0]
print("Learned ranking weights:", [round(x, 4) for x in w])

# Evaluate on VAL
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
    for c in req.capabilities:
        c_vec = encoded(serialize_capability(c.name, c.description, c.required_arguments))
        feats = extract_candidate_features(r['utterance'], q_vec, c, c_vec)
        score = float(np.dot(w, feats))
        scores.append((c.name, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    is_correct = (scores[0][0] == exp_cap)
    if is_correct:
        correct += 1
        if 'archive' in r.get('original_id', ''):
            archive_correct += 1
    else:
        print(f"FAILED: {r['original_id']} | exp: {exp_cap} | pred: {scores[0][0]} | scores: {scores}")

print(f"\nVAL Top-1 Accuracy: {correct}/{total} ({correct/total*100:.1f}%) | archive: {archive_correct}/20")
