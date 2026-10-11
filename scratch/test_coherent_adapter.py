import sys, json, joblib, hashlib, re, copy
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path('D:/JARVIS')
COMP = ROOT / 'reports/classifier_completion_20261010_163425'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
BUILD = ROOT / 'reports/multilingual_dataset_build_20261010_155129'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(COMP))

from assistant.orchestration.decision import DecisionRequest, Capability, ClassifierDecision
from assistant.orchestration.compact_adapter import CompactMultilingualAdapter, serialize_request, serialize_capability, _compose_intent
from experimental_adapter import typed_resolve, project_reference, safe_fallback

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

# Train learned heads on train set
path_clf = LogisticRegression(C=4.0, class_weight=None, max_iter=3000, random_state=42).fit(X_train, [r['expected']['execution_path'] for r in train])
clar_clf = LogisticRegression(C=4.0, class_weight='balanced', max_iter=3000, random_state=42).fit(X_train, [int(r['expected']['needs_clarification']) for r in train])
esc_clf = LogisticRegression(C=4.0, class_weight=None, max_iter=3000, random_state=42).fit(X_train, [int(r['expected']['needs_escalation']) for r in train])
ref_kind_clf = LogisticRegression(C=4.0, class_weight=None, max_iter=3000, random_state=42).fit(X_train, [('__none__' if not r['expected']['context_references'] else r['expected']['context_references'][0]['kind']) for r in train])
missing_info_clf = LogisticRegression(C=4.0, class_weight=None, max_iter=3000, random_state=42).fit(X_train, [('__none__' if not r['expected']['missing_information'] else r['expected']['missing_information'][0]['name']) for r in train])

# Mode classifiers
mode_clfs = {}
for m in ['action', 'conversational', 'memory', 'research']:
    mode_clfs[m] = LogisticRegression(C=4.0, class_weight='balanced', max_iter=3000, random_state=42).fit(X_train, [int(m in r['expected']['request_modes']) for r in train])

def rank_capabilities(request, request_vector, capability_vectors):
    if not request.capabilities:
        return None, 0.0
    
    q_norm = request_vector / (np.linalg.norm(request_vector) + 1e-9)
    cap_norms = capability_vectors / (np.linalg.norm(capability_vectors, axis=1, keepdims=True) + 1e-9)
    dense_sims = np.dot(cap_norms, q_norm.T).flatten()
    
    scores = []
    q_tokens = set(re.findall(r'\w+', request.user_text.lower()))
    
    for i, cap in enumerate(request.capabilities):
        base_sim = float(dense_sims[i])
        score = base_sim
        cap_name = cap.name.lower()
        name_tokens = set(re.findall(r'\w+', cap_name))
        overlap = len(q_tokens & name_tokens)
        if overlap > 0:
            score += 0.05 * overlap
            
        folder_terms = {'folder', 'directory', 'archive', 'tar', 'zip', 'extract', 'backup', 'folders'}
        file_terms = {'file', 'files', 'document', 'recent', 'code'}
        if any(t in q_tokens for t in folder_terms):
            if 'folder' in cap_name or 'directory' in cap_name:
                score += 0.20
            elif 'file' in cap_name:
                score -= 0.15
        elif any(t in q_tokens for t in file_terms):
            if 'file' in cap_name:
                score += 0.15
            elif 'folder' in cap_name:
                score -= 0.15
                
        search_terms = {'search', 'google', 'find', 'lookup', 'guide', 'steps', 'how'}
        if any(t in q_tokens for t in search_terms):
            if 'search' in cap_name:
                score += 0.10
                
        if cap.required_arguments:
            for req_arg in cap.required_arguments:
                if req_arg == 'url':
                    has_url = bool(re.search(r'https?://\S+|www\.\S+', request.user_text))
                    if not has_url:
                        score -= 0.50
        scores.append(score)
        
    scores = np.array(scores)
    best_idx = int(np.argmax(scores))
    best_cap = request.capabilities[best_idx]
    best_score = float(scores[best_idx])
    
    if best_score < 0.40:
        return None, best_score
    return best_cap.name, best_score

FIELDS = ('intent', 'execution_path', 'request_modes', 'selected_capability', 'needs_clarification', 'needs_escalation', 'missing_information', 'context_references')

def predict_v3(request):
    req_vec = encoded(serialize_request(request)).reshape(1, -1)
    
    # 1. Learned categorical predictions
    path = path_clf.predict(req_vec)[0]
    needs_clar = bool(clar_clf.predict(req_vec)[0])
    needs_esc = bool(esc_clf.predict(req_vec)[0])
    
    # Modes
    modes = []
    for m, mclf in mode_clfs.items():
        if mclf.predict(req_vec)[0] == 1:
            modes.append(m)
    if not modes:
        modes = ['action']
        
    # 2. Capability selection
    cap_vecs = np.stack([encoded(serialize_capability(c.name, c.description, c.required_arguments)) for c in request.capabilities]) if request.capabilities else np.zeros((0, 768))
    selected_cap, cap_score = rank_capabilities(request, req_vec[0], cap_vecs)
    
    # Coherent reconciliation between learned path and candidate ranker
    if selected_cap is not None:
        # We found a strongly matching capability
        if path not in ('single_action', 'multi_step_task'):
            path = 'single_action'
        needs_clar = False
        needs_esc = False
    else:
        # No capability selected
        if path in ('single_action', 'multi_step_task'):
            if needs_esc:
                path = 'escalate'
            elif needs_clar:
                path = 'clarify'
            else:
                path = 'clarify'
                needs_clar = True
                
    if path not in ('single_action', 'multi_step_task'):
        selected_cap = None
        
    # 3. Deterministic Missing Information rendering
    missing_info = []
    if needs_clar:
        missing_name = missing_info_clf.predict(req_vec)[0]
        if missing_name == '__none__':
            missing_name = 'target'
        if missing_name == 'target':
            desc = "The requested target is not identifiable"
        else:
            desc = "Required information is not available in the supplied request or context"
        missing_info = [{'name': missing_name, 'description': desc}]
        
    # 4. Context reference resolution
    refs = []
    alias = project_reference(request.context, request.user_text)
    ref_kind = ref_kind_clf.predict(req_vec)[0]
    if alias:
        ref_kind = 'project'
        
    if ref_kind != '__none__':
        # Extract span / resolve
        text_lower = request.user_text.lower()
        pronouns = {'it', 'that', 'this', 'atha', 'athai', 'ithu', 'athu', 'adi', 'idi', 'use', 'isse', 'yeh', 'woh'}
        tokens = set(re.findall(r'\w+', text_lower))
        
        if alias:
            span = alias
        elif tokens & pronouns and ref_kind == 'item':
            span = 'it'
        elif ref_kind == 'prior_conversation' and 'prior discussion' in [r['expected']['context_references'][0]['reference'] for r in train if r['expected']['context_references']]:
            # If training data indicates prior discussion
            span = 'prior discussion'
        else:
            span = request.user_text
            
        val_resolved = typed_resolve(request.context, ref_kind, request.user_text)
        refs = [{'reference': span, 'kind': ref_kind, 'resolved_value': val_resolved}]
        
    intent = _compose_intent(path, selected_cap)
    
    out = {
        'intent': intent,
        'execution_path': path,
        'request_modes': sorted(modes),
        'selected_capability': selected_cap,
        'needs_clarification': needs_clar,
        'needs_escalation': needs_esc,
        'missing_information': missing_info,
        'context_references': refs,
        'confidence': 0.85,
        'evidence': [{'kind': 'input_span', 'reference': request.user_text[:160]}],
        'abstained': False
    }
    ClassifierDecision.from_mapping(out)
    return out

def exact(r, p):
    return all((sorted(p[k]) == sorted(r['expected'][k]) if k == 'request_modes' else p[k] == r['expected'][k]) for k in FIELDS)

exact_count = 0
for i, r in enumerate(val):
    decision_req = req(r)
    p = predict_v3(decision_req)
    ok = exact(r, p)
    if ok:
        exact_count += 1
    else:
        diffs = [k for k in FIELDS if (sorted(p[k]) != sorted(r['expected'][k]) if k == 'request_modes' else p[k] != r['expected'][k])]
        print(f"Diff {r['original_id']:<35}: {diffs}")

print(f"\nStrict Exact Agreement: {exact_count} / {len(val)} ({exact_count / len(val) * 100:.1f}%)")

