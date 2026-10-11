"""Evaluation of Experimental v5 Adapter on the unchanged frozen 100-request corpus.
Compares performance directly against baseline v4.
"""
import csv
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path('D:/JARVIS')
EXP_DIR = ROOT / 'reports/jarvis_brain_forensic_improvement_20261010_183000'
FROZEN_100_DIR = ROOT / 'reports/classifier_v4_frozen_100_eval_20261010_180014'
EXP4 = ROOT / 'reports/classifier_v4_experiment_20261010_175140'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))

from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    DecisionRequest,
)
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder
from experimental_adapter_v5 import ExperimentalV5Adapter

FIELDS = (
    'intent',
    'execution_path',
    'request_modes',
    'selected_capability',
    'needs_clarification',
    'needs_escalation',
    'missing_information',
    'context_references',
)

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def dump_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def build_decision_request(r):
    return DecisionRequest(
        r['utterance'],
        r['context'],
        tuple(
            Capability(c['name'], c['description'], tuple(c['required_arguments']))
            for c in r['capabilities']
        ),
    )

def exact_match(exp, pred):
    return all(
        (sorted(pred[k]) == sorted(exp[k]) if k == 'request_modes' else pred[k] == exp[k])
        for k in FIELDS
    )

def main():
    print("=" * 60)
    print("JARVIS Brain v5: Frozen 100-Request Benchmark")
    print("=" * 60)

    corpus_path = FROZEN_100_DIR / 'unseen_evaluation_corpus_100.jsonl'
    v5_adapter_path = EXP_DIR / 'experimental_adapter_v5.py'
    model_path = EXP4 / 'best_candidate/compact_heads.joblib'

    corpus_sha = sha256_file(corpus_path)
    v5_adapter_sha = sha256_file(v5_adapter_path)
    model_sha = sha256_file(model_path)

    print(f"Corpus SHA-256:       {corpus_sha}")
    print(f"v5 Adapter SHA-256:   {v5_adapter_sha}")
    print(f"Model Artifact SHA:   {model_sha}")

    # Load unchanged frozen 100 corpus
    corpus = [json.loads(l) for l in corpus_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(corpus) == 100, f"Expected 100 records, got {len(corpus)}"

    # Load real multilingual BERT encoder and v5 adapter
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cache_manifest = json.loads((PRIOR / 'feature_cache_manifest.json').read_text(encoding='utf-8'))
    encoder = FrozenMultilingualEncoder(cache_manifest['encoder_snapshot'], 160)
    for p in encoder.model.parameters():
        p.requires_grad_(False)
    artifact = joblib.load(model_path)

    adapter = ExperimentalV5Adapter.__new__(ExperimentalV5Adapter)
    adapter.artifact = artifact
    adapter.encoder = encoder
    adapter._adapter_id = 'experimental-v5-forensic'
    adapter.inference_count = 0
    adapter.last_inference_metrics = {}

    print(f"Loaded {len(corpus)} records. Running evaluation...")
    predictions = []
    latencies = []
    errors = []

    for r in corpus:
        req = build_decision_request(r)
        start = time.perf_counter()
        try:
            pred = dict(adapter.predict(req))
            ClassifierDecision.from_mapping(pred)
            err = None
        except Exception as ex:
            pred = None
            err = str(ex)
        elapsed = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed)
        predictions.append(pred)
        errors.append(err)

    exact_flags = [p is not None and exact_match(r['expected'], p) for r, p in zip(corpus, predictions)]
    strict_exact_count = sum(exact_flags)
    cap_top1_count = sum(p is not None and p['selected_capability'] == r['expected']['selected_capability'] for r, p in zip(corpus, predictions))

    # Field counts
    field_counts = {}
    for f in FIELDS:
        field_counts[f] = sum(
            p is not None and (
                sorted(p[f]) == sorted(r['expected'][f])
                if f == 'request_modes'
                else p[f] == r['expected'][f]
            )
            for r, p in zip(corpus, predictions)
        )

    # Discrepancies and group stats
    group_stats = {}
    lang_stats = {}
    discrepancies = []

    for r, p, is_exact, err in zip(corpus, predictions, exact_flags, errors):
        grp = r['scenario_group']
        lang = r['language']

        g = group_stats.setdefault(grp, {'count': 0, 'exact': 0, 'cap_top1': 0})
        g['count'] += 1
        g['exact'] += int(is_exact)
        g['cap_top1'] += int(p is not None and p['selected_capability'] == r['expected']['selected_capability'])

        l = lang_stats.setdefault(lang, {'count': 0, 'exact': 0, 'cap_top1': 0})
        l['count'] += 1
        l['exact'] += int(is_exact)
        l['cap_top1'] += int(p is not None and p['selected_capability'] == r['expected']['selected_capability'])

        if not is_exact:
            diffs = {}
            if p is not None:
                for f in FIELDS:
                    match = (sorted(p[f]) == sorted(r['expected'][f])) if f == 'request_modes' else (p[f] == r['expected'][f])
                    if not match:
                        diffs[f] = {'expected': r['expected'][f], 'predicted': p[f]}
            discrepancies.append({
                'id': r['id'],
                'scenario_group': grp,
                'language': lang,
                'utterance': r['utterance'],
                'mismatched_fields': diffs,
                'prediction': p,
                'expected': r['expected']
            })

    # Abstention curve
    abstention_results = []
    for thresh in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        accepted = []
        accepted_exact = []
        for r, p in zip(corpus, predictions):
            if p is not None and not p.get('abstained') and p.get('confidence', 0.0) >= thresh:
                accepted.append(r)
                if exact_match(r['expected'], p):
                    accepted_exact.append(r)
        acc_len = len(accepted)
        acc_exact_len = len(accepted_exact)
        acc_precision = (acc_exact_len / acc_len * 100.0) if acc_len > 0 else 0.0
        abstention_results.append({
            'threshold': thresh,
            'accepted_count': acc_len,
            'accepted_exact': acc_exact_len,
            'accepted_precision_pct': acc_precision,
            'coverage_pct': acc_len / len(corpus) * 100.0,
            'overall_exact_count': acc_exact_len,
            'total': len(corpus)
        })

    # Save metrics
    metrics = {
        'total_requests': len(corpus),
        'strict_exact_agreement': strict_exact_count,
        'strict_exact_pct': strict_exact_count / len(corpus) * 100.0,
        'capability_top1_agreement': cap_top1_count,
        'capability_top1_pct': cap_top1_count / len(corpus) * 100.0,
        'inference_failures': sum(err is not None for err in errors),
        'latency_ms': {
            'median': float(np.median(latencies)),
            'p95': float(np.percentile(latencies, 95)),
            'mean': float(np.mean(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
        },
        'field_agreements': {
            f: {
                'count': field_counts[f],
                'pct': field_counts[f] / len(corpus) * 100.0
            } for f in FIELDS
        }
    }

    dump_json(EXP_DIR / 'metrics_100_v5.json', metrics)
    dump_json(EXP_DIR / 'scenario_group_metrics_v5.json', group_stats)
    dump_json(EXP_DIR / 'language_metrics_v5.json', lang_stats)
    dump_json(EXP_DIR / 'abstention_metrics_100_v5.json', abstention_results)
    dump_json(EXP_DIR / 'discrepancies_v5.json', discrepancies)

    pred_records = [
        {
            'id': r['id'],
            'scenario_group': r['scenario_group'],
            'language': r['language'],
            'utterance': r['utterance'],
            'expected': r['expected'],
            'prediction': p,
            'exact_match': is_exact,
            'error': err,
            'latency_ms': lat
        }
        for r, p, is_exact, err, lat in zip(corpus, predictions, exact_flags, errors, latencies)
    ]
    dump_json(EXP_DIR / 'predictions_100_v5.json', pred_records)

    # Confusion matrix for capabilities
    cm_dir = EXP_DIR / 'confusion_matrices'
    cm_dir.mkdir(parents=True, exist_ok=True)
    exp_caps = [str(r['expected']['selected_capability']) for r in corpus]
    pred_caps = [str(p['selected_capability']) if p is not None else 'None' for p in predictions]
    all_caps = sorted(list(set(exp_caps + pred_caps)))
    cap_cm = confusion_matrix(exp_caps, pred_caps, labels=all_caps)
    with (cm_dir / 'capability_confusion_v5.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Expected \\ Predicted'] + all_caps)
        for name, row in zip(all_caps, cap_cm):
            writer.writerow([name] + list(row))

    print("\n" + "=" * 60)
    print("V5 EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Strict Exact Agreement:       {strict_exact_count} / {len(corpus)} ({strict_exact_count / len(corpus) * 100.0:.1f}%)")
    print(f"Capability Top-1 Accuracy:    {cap_top1_count} / {len(corpus)} ({cap_top1_count / len(corpus) * 100.0:.1f}%)")
    print(f"Inference Failures:           {sum(err is not None for err in errors)} / {len(corpus)}")
    print(f"Latency: Median {metrics['latency_ms']['median']:.2f}ms | p95 {metrics['latency_ms']['p95']:.2f}ms | Mean {metrics['latency_ms']['mean']:.2f}ms")
    print("\nPer-Field Agreement:")
    for f in FIELDS:
        fc = field_counts[f]
        print(f"  {f:<22}: {fc:>3} / {len(corpus)} ({fc/len(corpus)*100.0:.1f}%)")

    print("\nPer-Language / Script Summary:")
    for lang, st in sorted(lang_stats.items()):
        print(f"  {lang:<14}: Exact {st['exact']:>2}/{st['count']:>2} ({st['exact']/st['count']*100:.1f}%) | Cap {st['cap_top1']:>2}/{st['count']:>2} ({st['cap_top1']/st['count']*100:.1f}%)")

    print("\nPer-Scenario-Group Summary:")
    for grp, st in sorted(group_stats.items()):
        print(f"  {grp:<28}: Exact {st['exact']:>2}/{st['count']:>2} ({st['exact']/st['count']*100:.1f}%) | Cap {st['cap_top1']:>2}/{st['count']:>2} ({st['cap_top1']/st['count']*100:.1f}%)")

    # Load v4 metrics for direct comparison
    v4_metrics = json.loads((FROZEN_100_DIR / 'metrics_100.json').read_text(encoding='utf-8'))
    print("\n" + "=" * 60)
    print("BEFORE (v4) vs AFTER (v5) COMPARISON")
    print("=" * 60)
    print(f"Strict Exact Agreement:  v4: {v4_metrics['strict_exact_agreement']}/100 ({v4_metrics['strict_exact_pct']:.1f}%)  -->  v5: {strict_exact_count}/100 ({strict_exact_count/len(corpus)*100.0:.1f}%) [Delta: +{strict_exact_count - v4_metrics['strict_exact_agreement']}]")
    print(f"Capability Top-1 Acc:    v4: {v4_metrics['capability_top1_agreement']}/100 ({v4_metrics['capability_top1_pct']:.1f}%)  -->  v5: {cap_top1_count}/100 ({cap_top1_count/len(corpus)*100.0:.1f}%) [Delta: +{cap_top1_count - v4_metrics['capability_top1_agreement']}]")
    print("\nField Differentiators:")
    for f in FIELDS:
        c4 = v4_metrics['field_agreements'][f]['count']
        c5 = field_counts[f]
        diff = c5 - c4
        sign = f"+{diff}" if diff >= 0 else str(diff)
        print(f"  {f:<22}: v4: {c4:>3}/100 ({c4}%) --> v5: {c5:>3}/100 ({c5}%) [{sign}]")

if __name__ == '__main__':
    main()

