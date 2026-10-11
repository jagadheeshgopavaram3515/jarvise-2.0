import sys, json, joblib, hashlib
from pathlib import Path
import numpy as np

ROOT = Path('D:/JARVIS')
EXP4 = ROOT / 'reports/classifier_v4_experiment_20261010_175140'
EXP3 = ROOT / 'reports/classifier_next_experiment_20261010_171102'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
BUILD = ROOT / 'reports/multilingual_dataset_build_20261010_155129'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(EXP4))

from assistant.orchestration.decision import DecisionRequest, Capability, ClassifierDecision
from experimental_adapter_v4 import ExperimentalV4Adapter

val = [json.loads(line) for line in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]
cache = json.loads((PRIOR / 'feature_cache_manifest.json').read_text(encoding='utf-8'))
vectors = np.load(PRIOR / 'frozen_features.npz')['vectors']
index = {x['text_sha256']: x['index'] for x in cache['texts']}
def encoded(text): return vectors[index[hashlib.sha256(text.encode()).hexdigest()]]

class ReplayEncoder:
    def __init__(self): self.forward_calls = 0; self.last_timings = {}
    def encode(self, texts, **kwargs):
        self.forward_calls += 1
        return np.stack([encoded(t) for t in texts])

artifact = joblib.load(EXP3 / 'best_candidate/compact_heads.joblib')
adapter = ExperimentalV4Adapter.__new__(ExperimentalV4Adapter)
adapter.artifact = artifact
adapter.encoder = ReplayEncoder()
adapter._adapter_id = 'experimental-v4'
adapter.inference_count = 0
adapter.last_inference_metrics = {}

FIELDS = ('intent', 'execution_path', 'request_modes', 'selected_capability', 'needs_clarification', 'needs_escalation', 'missing_information', 'context_references')

def exact(r, p):
    return all((sorted(p[k]) == sorted(r['expected'][k]) if k == 'request_modes' else p[k] == r['expected'][k]) for k in FIELDS)

exact_count = 0
field_counts = {f: 0 for f in FIELDS}
mismatches = []

for r in val:
    req = DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))
    p = adapter.predict(req)
    ok = exact(r, p)
    if ok:
        exact_count += 1
    else:
        diffs = [k for k in FIELDS if (sorted(p[k]) != sorted(r['expected'][k]) if k == 'request_modes' else p[k] != r['expected'][k])]
        mismatches.append((r['original_id'], diffs, r['expected'], p))
    for f in FIELDS:
        if (sorted(p[f]) == sorted(r['expected'][f]) if f == 'request_modes' else p[f] == r['expected'][f]):
            field_counts[f] += 1

print(f"=== Clean V4 on Original 40 Validation Records ===")
print(f"Strict Exact Agreement: {exact_count} / {len(val)} ({exact_count / len(val) * 100:.1f}%)")
print("\nPer-field Breakdown:")
for f in FIELDS:
    print(f"  {f:<22}: {field_counts[f]} / {len(val)}")

print(f"\nMismatches ({len(mismatches)}):")
for oid, diffs, exp, pred in mismatches:
    print(f"\n[{oid}] diffs: {diffs}")
    for d in diffs:
        print(f"   exp {d}: {exp[d]}")
        print(f"   got {d}: {pred[d]}")

