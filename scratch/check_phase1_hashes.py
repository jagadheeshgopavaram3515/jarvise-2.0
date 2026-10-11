import hashlib, json
from pathlib import Path

ROOT = Path('D:/JARVIS')
EXP4 = ROOT / 'reports/classifier_v4_experiment_20261010_175140'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
OUT = ROOT / 'reports/classifier_v4_frozen_100_eval_20261010_180014'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    'v4_adapter_sha256': sha(EXP4 / 'experimental_adapter_v4.py'),
    'v4_model_artifact_sha256': sha(EXP4 / 'best_candidate/compact_heads.joblib'),
    'provisional_train_sha256': sha(PRIOR / 'provisional_train.jsonl'),
    'provisional_validation_sha256': sha(PRIOR / 'provisional_validation.jsonl'),
    'unseen_corpus_24_sha256': sha(EXP4 / 'unseen_evaluation_corpus_v4.jsonl'),
    'frozen_features_sha256': sha(PRIOR / 'frozen_features.npz'),
}

print(json.dumps(manifest, indent=2))
(OUT / 'phase1_input_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

