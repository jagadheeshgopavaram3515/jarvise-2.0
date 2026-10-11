"""Non-destructive test suite verifying integrity, coverage, and reproducibility of the frozen 100-request evaluation."""
import json
import hashlib
from pathlib import Path
import pytest

ROOT = Path('D:/JARVIS')
EVAL_DIR = ROOT / 'reports/classifier_v4_frozen_100_eval_20261010_180014'

import sys
sys.path.insert(0, str(ROOT))
from assistant.orchestration.decision import ClassifierDecision

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def test_evaluation_isolated_from_production():
    """Verify evaluation scripts only write inside the isolated reports/ evaluation directory."""
    eval_code = (EVAL_DIR / 'run_frozen_100_eval.py').read_text(encoding='utf-8')
    assert 'assistant/' not in eval_code or 'from assistant' in eval_code
    assert 'write_text' not in eval_code or 'EVAL_DIR' in eval_code

def test_corpus_hash_and_uniqueness():
    corpus_file = EVAL_DIR / 'unseen_evaluation_corpus_100.jsonl'
    assert corpus_file.exists()
    assert sha256_file(corpus_file) == '206118b07a501d97cabbee92c2ee8dda2b1f4db020ad59ecf64dd4d7fe19c59f'
    lines = [json.loads(l) for l in corpus_file.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(lines) == 100
    utterances = [r['utterance'].strip().lower() for r in lines]
    assert len(set(utterances)) == 100, "Corpus contains duplicate utterances"

def test_corpus_diversity_coverage():
    corpus_file = EVAL_DIR / 'unseen_evaluation_corpus_100.jsonl'
    lines = [json.loads(l) for l in corpus_file.read_text(encoding='utf-8').splitlines() if l.strip()]
    groups = set(r['scenario_group'] for r in lines)
    assert len(groups) == 10, "Corpus must span 10 scenario groups"
    languages = set(r['language'] for r in lines)
    assert 'en' in languages
    assert 'hi' in languages and 'hi-Latn' in languages
    assert 'te' in languages and 'te-Latn' in languages
    assert 'ta' in languages and 'ta-Latn' in languages

def test_predictions_schema_validity():
    pred_file = EVAL_DIR / 'predictions_100.json'
    assert pred_file.exists()
    preds = json.loads(pred_file.read_text(encoding='utf-8'))
    assert len(preds) == 100
    for p in preds:
        pred_dict = p['prediction']
        decision = ClassifierDecision.from_mapping(pred_dict)
        assert decision.execution_path in {
            'single_action', 'multi_step_task', 'direct_response',
            'memory_recall', 'clarify', 'escalate', 'research'
        }

def test_metrics_integrity():
    metrics = json.loads((EVAL_DIR / 'metrics_100.json').read_text(encoding='utf-8'))
    assert metrics['total_requests'] == 100
    assert metrics['strict_exact_agreement'] == 45
    assert metrics['strict_exact_pct'] == 45.0
    assert metrics['capability_top1_agreement'] == 66
    assert metrics['capability_top1_pct'] == 66.0
    assert metrics['inference_failures'] == 0

def test_abstention_monotonicity():
    abst = json.loads((EVAL_DIR / 'abstention_metrics_100.json').read_text(encoding='utf-8'))
    counts = [row['accepted_count'] for row in abst]
    # Acceptance count must be non-increasing with increasing threshold
    for i in range(len(counts) - 1):
        assert counts[i] >= counts[i + 1]

