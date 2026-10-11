import sys, json, joblib, hashlib, numpy as np
from pathlib import Path

ROOT = Path('D:/JARVIS')
COMP = ROOT / 'reports/classifier_completion_20261010_163425'
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

heads = joblib.load(COMP / 'best_candidate/compact_heads.joblib')
VAL = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]

print("Inspecting archive folder records where path/clarify was triggered:")
for r in VAL:
    if 'synth_open_archive_folder' in r['original_id']:
        req = DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))
        vec = encoded(serialize_request(req)).reshape(1, -1)
        path_pred = heads['execution_path'].predict(vec)[0]
        clar_pred = heads['clarification'].predict(vec)[0]
        esc_pred = heads['escalation'].predict(vec)[0]
        clar_prob = heads['clarification'].predict_proba(vec)[0][1]
        esc_prob = heads['escalation'].predict_proba(vec)[0][1]
        print(f"{r['original_id']:<38} | path_head={path_pred:<14} | clar={clar_pred} ({clar_prob:.3f}) | esc={esc_pred} ({esc_prob:.3f})")

