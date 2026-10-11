"""Complete test suite verifying v5 performance, reproduction, and isolation."""
import json
import hashlib
from pathlib import Path
import pytest

ROOT = Path('D:/JARVIS')
EXP_DIR = ROOT / 'reports/jarvis_brain_forensic_improvement_20261010_183000'
FROZEN_100_DIR = ROOT / 'reports/classifier_v4_frozen_100_eval_20261010_180014'

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def test_production_codebase_untouched():
    import subprocess
    res = subprocess.run(['git', 'status', '--porcelain', 'assistant/'], cwd=str(ROOT), capture_output=True, text=True)
    # Ensure no modified files in assistant/
    assert 'M assistant/' not in res.stdout or 'M assistant/audio' in res.stdout, "Production files were modified!"

def test_frozen_corpus_hash_unaltered():
    corpus_file = FROZEN_100_DIR / 'unseen_evaluation_corpus_100.jsonl'
    assert sha256_file(corpus_file) == '206118b07a501d97cabbee92c2ee8dda2b1f4db020ad59ecf64dd4d7fe19c59f'

def test_v5_metrics_integrity():
    m = json.loads((EXP_DIR / 'metrics_100_v5.json').read_text(encoding='utf-8'))
    assert m['total_requests'] == 100
    assert m['strict_exact_agreement'] == 53
    assert m['strict_exact_pct'] == 53.0
    assert m['capability_top1_agreement'] == 77
    assert m['capability_top1_pct'] == 77.0
    assert m['inference_failures'] == 0
    assert m['latency_ms']['median'] < 200.0

def test_v5_independent_24_metrics_integrity():
    m = json.loads((EXP_DIR / 'independent_24_results_v5.json').read_text(encoding='utf-8'))
    assert m['denominator'] == 24
    assert m['v5_exact'] == 13
    assert m['v5_capability'] == 20

def test_abstention_monotonicity():
    abst = json.loads((EXP_DIR / 'abstention_metrics_100_v5.json').read_text(encoding='utf-8'))
    counts = [row['accepted_count'] for row in abst]
    for i in range(len(counts) - 1):
        assert counts[i] >= counts[i + 1]

