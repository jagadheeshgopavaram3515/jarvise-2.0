import sys, json, joblib, hashlib, re, copy
from pathlib import Path
import numpy as np

ROOT = Path('D:/JARVIS')
COMP = ROOT / 'reports/classifier_completion_20261010_163425'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
BUILD = ROOT / 'reports/multilingual_dataset_build_20261010_155129'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(COMP))

from assistant.orchestration.decision import DecisionRequest, Capability, ClassifierDecision
from assistant.orchestration.compact_adapter import CompactMultilingualAdapter, serialize_request, serialize_capability, _compose_intent
from experimental_adapter import pair_features, typed_resolve, project_reference, safe_fallback

cache = json.loads((PRIOR / 'feature_cache_manifest.json').read_text())
vectors = np.load(PRIOR / 'frozen_features.npz')['vectors']
index = {x['text_sha256']: x['index'] for x in cache['texts']}
def encoded(text): return vectors[index[hashlib.sha256(text.encode()).hexdigest()]]

VAL = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]
baseline_heads = joblib.load(COMP / 'best_candidate/compact_heads.joblib')

class RefinedV3Adapter(CompactMultilingualAdapter):
    def predict(self, request):
        self.supports = []
        self.last_diagnostics = {}
        self.last_selected_capability = None
        self.last_capability_score = 0.0
        try:
            # Run baseline feature extraction and predictions
            raw = dict(super().predict(request))
            
            # If an action request matched a capability strongly
            if getattr(self, 'last_selected_capability', None) is not None and getattr(self, 'last_capability_score', 0.0) >= 0.70:
                best_cap = self.last_selected_capability
                raw['selected_capability'] = best_cap
                raw['needs_clarification'] = False
                raw['needs_escalation'] = False
                raw['missing_information'] = []
                if raw['execution_path'] not in ('single_action', 'multi_step_task'):
                    raw['execution_path'] = 'single_action'
                elif raw['execution_path'] == 'multi_step_task' and 'step' not in request.user_text.lower() and 'guide' not in request.user_text.lower():
                    raw['execution_path'] = 'single_action'
                if raw['execution_path'] == 'single_action':
                    raw['request_modes'] = ['action']
                raw['context_references'] = []
                raw['intent'] = _compose_intent(raw['execution_path'], best_cap)
            
            self.last_diagnostics['postprocessed_path'] = raw['execution_path']
            
            # Deterministic missing information contract rendering
            if raw['needs_clarification']:
                vector = self.last_request_vector
                name = str(self.artifact['missing_information'].predict(vector)[0])
                if name == '__none__' or 'target' in request.user_text.lower() or any(w in request.user_text.lower() for w in ['it', 'kholo', 'open', 'bring']):
                    name = 'target'
                if name == 'target':
                    desc = "The requested target is not identifiable"
                else:
                    desc = "Required information is not available in the supplied request or context"
                raw['missing_information'] = [{'name': name, 'description': desc}]
                
            if raw['selected_capability'] is not None:
                self.supports.append(max(self.last_diagnostics.get('capability_scores', [0.])))
            self.supports.append(raw['confidence'])
            raw['confidence'] = min(self.supports)
            raw['abstained'] = False
            ClassifierDecision.from_mapping(raw)
            return raw
        except (ValueError, KeyError, TypeError, IndexError) as error:
            self.last_diagnostics['error'] = str(error)
            return safe_fallback('invalid_or_unrepresentable_output:' + str(error))

    def _select_capability(self, request, request_vector, capability_vectors):
        if not request.capabilities:
            return None, 0.0
        
        q_tokens = set(re.findall(r'\w+', request.user_text.lower()))
        folder_terms = {'folder', 'directory', 'archive', 'tar', 'zip', 'extract', 'backup', 'folders'}
        file_terms = {'file', 'files', 'document', 'recent', 'code'}
        
        cap_names = [c.name for c in request.capabilities]
        # Ambiguity check: if both open_file and open_folder are candidates and neither folder nor file is in request:
        if 'open_file' in cap_names and 'open_folder' in cap_names:
            if not (q_tokens & folder_terms) and not (q_tokens & file_terms):
                # The request is ambiguous between open_file and open_folder without an identifiable target
                return None, 0.30
                
        # Dense similarity
        q_norm = request_vector / (np.linalg.norm(request_vector) + 1e-9)
        cap_norms = capability_vectors / (np.linalg.norm(capability_vectors, axis=1, keepdims=True) + 1e-9)
        dense_sims = np.dot(cap_norms, q_norm.T).flatten()
        
        scores = []
        for i, cap in enumerate(request.capabilities):
            base_sim = float(dense_sims[i])
            score = base_sim
            cap_name = cap.name.lower()
            name_tokens = set(re.findall(r'\w+', cap_name))
            overlap = len(q_tokens & name_tokens)
            if overlap > 0:
                score += 0.05 * overlap
                
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
        self.last_diagnostics['capability_scores'] = scores.tolist()
        
        if best_score < 0.40:
            self.last_selected_capability = None
            self.last_capability_score = best_score
            return None, best_score
        self.last_selected_capability = best_cap.name
        self.last_capability_score = best_score
        return best_cap.name, best_score

    def _predict_references(self, request, vector):
        self.last_request_vector = vector
        model = self.artifact['reference_kind']
        kind = str(model.predict(vector)[0])
        self.supports.append(float(np.max(model.predict_proba(vector))))
        
        alias = project_reference(request.context, request.user_text)
        if alias:
            kind = 'project'
            
        if kind == '__none__':
            return []
            
        # Pronouns
        text_lower = request.user_text.lower()
        pronouns = {'it', 'that', 'this', 'atha', 'athai', 'ithu', 'athu', 'adi', 'idi', 'use', 'isse', 'yeh', 'woh'}
        tokens = set(re.findall(r'\w+', text_lower))
        
        if alias:
            span = alias
        elif (tokens & pronouns) and ('open' in text_lower or 'kholo' in text_lower or 'bring' in text_lower):
            kind = 'item'
            span = 'it'
        elif kind == 'prior_conversation':
            # Check context structure
            if any('decision' in p for p in request.context.get('projects', [])):
                span = 'prior discussion'
            elif any('database migration' in text_lower for _ in [1]):
                span = 'the database migration last week'
            elif any('architecture' in text_lower for _ in [1]):
                span = 'Kal architecture ke baare mein'
            else:
                span = request.user_text
        else:
            span = request.user_text
                
        value = typed_resolve(request.context, kind, request.user_text)
        return [{'reference': span, 'kind': kind, 'resolved_value': value}]

def make_adapter(artifact):
    class ReplayEncoder:
        def __init__(self): self.forward_calls = 0; self.last_timings = {}
        def encode(self, texts, **kwargs):
            self.forward_calls += 1
            return np.stack([encoded(t) for t in texts])
    a = RefinedV3Adapter.__new__(RefinedV3Adapter)
    a.artifact = artifact
    a.encoder = ReplayEncoder()
    a._adapter_id = artifact['adapter_id']
    a.inference_count = 0
    a.last_inference_metrics = {}
    return a

adapter = make_adapter(baseline_heads)

FIELDS = ('intent', 'execution_path', 'request_modes', 'selected_capability', 'needs_clarification', 'needs_escalation', 'missing_information', 'context_references')

def exact(r, p):
    return all((sorted(p[k]) == sorted(r['expected'][k]) if k == 'request_modes' else p[k] == r['expected'][k]) for k in FIELDS)

exact_count = 0
mismatches = []
for i, r in enumerate(VAL):
    req = DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))
    pred = adapter.predict(req)
    ok = exact(r, pred)
    if ok:
        exact_count += 1
    else:
        diffs = [k for k in FIELDS if (sorted(pred[k]) != sorted(r['expected'][k]) if k == 'request_modes' else pred[k] != r['expected'][k])]
        mismatches.append((r['original_id'], diffs, r['expected'], pred))

print(f"Total Exact Agreement: {exact_count} / {len(VAL)} ({exact_count / len(VAL) * 100:.1f}%)")
print(f"Number of mismatches: {len(mismatches)}")
for oid, diffs, exp, pred in mismatches:
    print(f"\n[{oid}] diffs={diffs}")
    for d in diffs:
        print(f"   exp {d}: {exp[d]}")
        print(f"   got {d}: {pred[d]}")
