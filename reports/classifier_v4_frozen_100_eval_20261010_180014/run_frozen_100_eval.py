"""Evaluation of Frozen JARVIS Compact BERT v4 on 100-request unseen corpus.
Preserves production assistant codebase unchanged.
Evaluates the clean v4 model artifact and adapter without retraining or tuning.
"""
import collections
import copy
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
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path('D:/JARVIS')
EVAL_DIR = Path(__file__).resolve().parent
EXP4 = ROOT / 'reports/classifier_v4_experiment_20261010_175140'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
BUILD = ROOT / 'reports/multilingual_dataset_build_20261010_155129'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(EXP4))

from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    DecisionRequest,
)
from assistant.orchestration.compact_adapter import (
    CompactMultilingualAdapter,
    FrozenMultilingualEncoder,
    serialize_capability,
    serialize_request,
)
from experimental_adapter_v4 import ExperimentalV4Adapter

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

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

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
    print("JARVIS Compact BERT v4: Frozen 100-Request Generalization Evaluation")
    print("=" * 60)

    corpus_path = EVAL_DIR / 'unseen_evaluation_corpus_100.jsonl'
    v4_adapter_path = EXP4 / 'experimental_adapter_v4.py'
    v4_model_path = EXP4 / 'best_candidate/compact_heads.joblib'
    encoder_path = ROOT / 'benchmarks/fast_classifier/models/v3/encoder'

    corpus_sha = sha256_file(corpus_path)
    v4_adapter_sha = sha256_file(v4_adapter_path)
    v4_model_sha = sha256_file(v4_model_path)

    print(f"Corpus SHA-256:       {corpus_sha}")
    print(f"v4 Adapter SHA-256:   {v4_adapter_sha}")
    print(f"v4 Model SHA-256:     {v4_model_sha}")

    # Load 100-request corpus
    raw_lines = [l.strip() for l in corpus_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    corpus = [json.loads(l) for l in raw_lines]
    print(f"Loaded {len(corpus)} records.")

    # Initialize live real encoder and v4 adapter
    print("\nLoading FrozenMultilingualEncoder...")
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cache_manifest = load_json(PRIOR / 'feature_cache_manifest.json')
    encoder = FrozenMultilingualEncoder(cache_manifest['encoder_snapshot'], 160)
    for p in encoder.model.parameters():
        p.requires_grad_(False)
    v4_artifact = joblib.load(v4_model_path)

    adapter = ExperimentalV4Adapter.__new__(ExperimentalV4Adapter)
    adapter.artifact = v4_artifact
    adapter.encoder = encoder
    adapter._adapter_id = 'experimental-v4-clean'
    adapter.inference_count = 0
    adapter.last_inference_metrics = {}

    print("Running frozen evaluation on 100 requests...")
    predictions = []
    latencies = []
    errors = []

    for i, r in enumerate(corpus):
        req = build_decision_request(r)
        start = time.perf_counter()
        try:
            pred = dict(adapter.predict(req))
            # Validate against schema
            ClassifierDecision.from_mapping(pred)
            err = None
        except Exception as ex:
            pred = None
            err = str(ex)
        elapsed = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed)
        predictions.append(pred)
        errors.append(err)

    # Compute agreements
    exact_flags = [p is not None and exact_match(r['expected'], p) for r, p in zip(corpus, predictions)]
    strict_exact_count = sum(exact_flags)
    cap_top1_count = sum(p is not None and p['selected_capability'] == r['expected']['selected_capability'] for r, p in zip(corpus, predictions))

    # Field-level accuracy
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

    # Group breakdowns
    group_stats = {}
    lang_stats = {}
    discrepancies = []

    for idx, (r, p, is_exact, err) in enumerate(zip(corpus, predictions, exact_flags, errors)):
        grp = r.get('scenario_group', 'unknown')
        lang = r.get('language', 'unknown')

        # Group
        g = group_stats.setdefault(grp, {'count': 0, 'exact': 0, 'cap_top1': 0, 'field_agreements': {f: 0 for f in FIELDS}})
        g['count'] += 1
        g['exact'] += int(is_exact)
        g['cap_top1'] += int(p is not None and p['selected_capability'] == r['expected']['selected_capability'])
        for f in FIELDS:
            if p is not None and (sorted(p[f]) == sorted(r['expected'][f]) if f == 'request_modes' else p[f] == r['expected'][f]):
                g['field_agreements'][f] += 1

        # Language
        l = lang_stats.setdefault(lang, {'count': 0, 'exact': 0, 'cap_top1': 0, 'field_agreements': {f: 0 for f in FIELDS}})
        l['count'] += 1
        l['exact'] += int(is_exact)
        l['cap_top1'] += int(p is not None and p['selected_capability'] == r['expected']['selected_capability'])
        for f in FIELDS:
            if p is not None and (sorted(p[f]) == sorted(r['expected'][f]) if f == 'request_modes' else p[f] == r['expected'][f]):
                l['field_agreements'][f] += 1

        # Record discrepancy if not exact
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
                'error': err,
                'mismatched_fields': diffs if p is not None else 'INFERENCE_FAILURE',
                'full_prediction': p,
                'expected': r['expected']
            })

    # Abstention curve
    print("\nComputing abstention curve on 100 requests...")
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

    # Confusion matrices
    cm_dir = EVAL_DIR / 'confusion_matrices'
    cm_dir.mkdir(parents=True, exist_ok=True)

    # Capability confusion
    exp_caps = [str(r['expected']['selected_capability']) for r in corpus]
    pred_caps = [str(p['selected_capability']) if p is not None else 'None' for p in predictions]
    all_caps = sorted(list(set(exp_caps + pred_caps)))
    cap_cm = confusion_matrix(exp_caps, pred_caps, labels=all_caps)
    with (cm_dir / 'capability_confusion.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Expected \\ Predicted'] + all_caps)
        for name, row in zip(all_caps, cap_cm):
            writer.writerow([name] + list(row))

    # Intent confusion
    exp_intents = [str(r['expected']['intent']) for r in corpus]
    pred_intents = [str(p['intent']) if p is not None else 'None' for p in predictions]
    all_intents = sorted(list(set(exp_intents + pred_intents)))
    intent_cm = confusion_matrix(exp_intents, pred_intents, labels=all_intents)
    with (cm_dir / 'intent_confusion.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Expected \\ Predicted'] + all_intents)
        for name, row in zip(all_intents, intent_cm):
            writer.writerow([name] + list(row))

    # Execution path confusion
    exp_paths = [str(r['expected']['execution_path']) for r in corpus]
    pred_paths = [str(p['execution_path']) if p is not None else 'None' for p in predictions]
    all_paths = sorted(list(set(exp_paths + pred_paths)))
    path_cm = confusion_matrix(exp_paths, pred_paths, labels=all_paths)
    with (cm_dir / 'execution_path_confusion.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Expected \\ Predicted'] + all_paths)
        for name, row in zip(all_paths, path_cm):
            writer.writerow([name] + list(row))

    # Metrics payload
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

    # Save output artifacts
    dump_json(EVAL_DIR / 'metrics_100.json', metrics)
    dump_json(EVAL_DIR / 'scenario_group_metrics.json', group_stats)
    dump_json(EVAL_DIR / 'language_metrics.json', lang_stats)
    dump_json(EVAL_DIR / 'abstention_metrics_100.json', abstention_results)
    dump_json(EVAL_DIR / 'discrepancies_and_failure_modes.json', discrepancies)

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
    dump_json(EVAL_DIR / 'predictions_100.json', pred_records)

    eval_manifest = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'corpus_file': 'unseen_evaluation_corpus_100.jsonl',
        'corpus_sha256': corpus_sha,
        'v4_adapter_sha256': v4_adapter_sha,
        'v4_model_sha256': v4_model_sha,
        'summary': {
            'total': len(corpus),
            'strict_exact': strict_exact_count,
            'strict_exact_pct': strict_exact_count / len(corpus) * 100.0,
            'capability_top1': cap_top1_count,
            'capability_top1_pct': cap_top1_count / len(corpus) * 100.0,
            'median_latency_ms': float(np.median(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
        }
    }
    dump_json(EVAL_DIR / 'eval_manifest.json', eval_manifest)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
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

    print("\nAbstention Operating Curve:")
    for ab in abstention_results:
        print(f"  Threshold {ab['threshold']:.1f}: Accepted {ab['accepted_count']:>3}/100, Precision {ab['accepted_precision_pct']:.1f}%, Full Set Exact {ab['overall_exact_count']}")

if __name__ == '__main__':
    main()
