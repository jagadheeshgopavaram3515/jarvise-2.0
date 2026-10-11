import hashlib, json
from pathlib import Path

ROOT = Path('D:/JARVIS')
EXP = ROOT / 'reports/classifier_next_experiment_20261010_171102'
COMP = ROOT / 'reports/classifier_completion_20261010_163425'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
AUDIT = ROOT / 'reports/classifier_generalization_audit_20261010_173628'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

hashes = {
    'baseline_model_v3_sha256': sha(ROOT / 'benchmarks/fast_classifier/models/v3/compact_heads.joblib'),
    'v2_model_candidate_sha256': sha(COMP / 'best_candidate/compact_heads.joblib'),
    'v3_model_candidate_sha256': sha(EXP / 'best_candidate/compact_heads.joblib'),
    'provisional_train_sha256': sha(PRIOR / 'provisional_train.jsonl'),
    'provisional_validation_sha256': sha(PRIOR / 'provisional_validation.jsonl'),
    'frozen_features_sha256': sha(PRIOR / 'frozen_features.npz'),
    'adapter_v3_code_sha256': sha(EXP / 'experimental_adapter_v3.py'),
    'runner_v3_code_sha256': sha(EXP / 'run_next_experiment.py'),
    'test_v3_code_sha256': sha(EXP / 'test_next_experiment.py'),
    'best_metrics_v3_sha256': sha(EXP / 'best_metrics.json'),
}

print(json.dumps(hashes, indent=2))
(AUDIT / 'sha256_manifest.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')

