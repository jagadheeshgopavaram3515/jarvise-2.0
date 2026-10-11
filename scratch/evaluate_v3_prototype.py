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
from assistant.orchestration.compact_adapter import CompactMultilingualAdapter, serialize_request, serialize_capability
from experimental_adapter import pair_features, typed_resolve, project_reference, safe_fallback

cache = json.loads((PRIOR / 'feature_cache_manifest.json').read_text())
vectors = np.load(PRIOR / 'frozen_features.npz')['vectors']
index = {x['text_sha256']: x['index'] for x in cache['texts']}
def encoded(text): return vectors[index[hashlib.sha256(text.encode()).hexdigest()]]

VAL = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]
baseline_model = joblib.load(COMP / 'best_candidate/compact_heads.joblib')

class StructuredV3Adapter(CompactMultilingualAdapter):
    def predict(self, request):
        self.supports = []
        self.last_diagnostics = {}
        try:
            raw = dict(super().predict(request))
            self.last_diagnostics['postprocessed_path'] = raw['execution_path']
            
            # 1. Structured capability selection (already handles action vs clarify/escalate)
            if raw['needs_clarification']:
                vector = self.last_request_vector
                name = str(self.artifact['missing_information'].predict(vector)[0])
                if name == '__none__':
                    name = 'target'
                # Deterministic contract rendering
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
        
        # Dense dot-product similarity (or cosine)
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
            
            # Token overlap bonus
            overlap = len(q_tokens & name_tokens)
            if overlap > 0:
                score += 0.05 * overlap
                
            # Entity match bonus
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
                    
            # Search query bonus
            search_terms = {'search', 'google', 'find', 'lookup', 'guide', 'steps', 'how'}
            if any(t in q_tokens for t in search_terms):
                if 'search' in cap_name:
                    score += 0.10
                    
            # Feasibility filter: required arguments check
            if cap.required_arguments:
                for req_arg in cap.required_arguments:
                    if req_arg == 'url':
                        has_url = bool(re.search(r'https?://\S+|www\.\S+', request.user_text))
                        if not has_url:
                            score -= 0.50  # Heavy penalty if required url is completely absent
                            
            scores.append(score)
            
        scores = np.array(scores)
        best_idx = int(np.argmax(scores))
        best_cap = request.capabilities[best_idx]
        best_score = float(scores[best_idx])
        self.last_diagnostics['capability_scores'] = scores.tolist()
        
        # Threshold check
        if best_score < 0.40:
            return None, best_score
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
            
        # If pronoun reference in utterance
        text_lower = request.user_text.lower()
        pronouns = {'it', 'that', 'this', 'atha', 'athai', 'ithu', 'athu', 'adi', 'idi', 'use', 'isse', 'yeh', 'woh'}
        tokens = set(re.findall(r'\w+', text_lower))
        
        # Check if project alias
        if alias:
            span = alias
        elif tokens & pronouns and kind == 'item':
            span = 'it'
        else:
            span_model = self.artifact.get('reference_span')
            if span_model is not None:
                span = str(span_model.predict(vector)[0])
                self.supports.append(float(np.max(span_model.predict_proba(vector))))
                if span == '__none__':
                    span = request.user_text
                elif span == '__utterance__':
                    span = request.user_text
            else:
                span = request.user_text
                
        # Typed resolution against context
        value = typed_resolve(request.context, kind, request.user_text)
        return [{'reference': span, 'kind': kind, 'resolved_value': value}]

def make_adapter(artifact):
    class ReplayEncoder:
        def __init__(self): self.forward_calls = 0; self.last_timings = {}
        def encode(self, texts, **kwargs):
            self.forward_calls += 1
            return np.stack([encoded(t) for t in texts])
    a = StructuredV3Adapter.__new__(StructuredV3Adapter)
    a.artifact = artifact
    a.encoder = ReplayEncoder()
    a._adapter_id = artifact['adapter_id']
    a.inference_count = 0
    a.last_inference_metrics = {}
    return a

adapter = make_adapter(baseline_model)

FIELDS = ('intent', 'execution_path', 'request_modes', 'selected_capability', 'needs_clarification', 'needs_escalation', 'missing_information', 'context_references')

def exact(r, p):
    return all((sorted(p[k]) == sorted(r['expected'][k]) if k == 'request_modes' else p[k] == r['expected'][k]) for k in FIELDS)

exact_count = 0
for i, r in enumerate(VAL):
    req = DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))
    pred = adapter.predict(req)
    ok = exact(r, pred)
    if ok:
        exact_count += 1
    else:
        diffs = [k for k in FIELDS if (sorted(pred[k]) != sorted(r['expected'][k]) if k == 'request_modes' else pred[k] != r['expected'][k])]
        print(f"Mismatch {r['original_id']}: diffs={diffs}")

print(f"\nTotal Exact Agreement: {exact_count} / {len(VAL)} ({exact_count / len(VAL) * 100:.1f}%)")

