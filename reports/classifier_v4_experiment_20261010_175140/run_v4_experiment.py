"""Execution and evaluation of JARVIS Compact BERT v4.
Evaluates clean v4 adapter on original 40-record validation set and new 24-record unseen corpus.
Preserves production assistant codebase unchanged.
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
OUT = Path(__file__).resolve().parent
EXP3 = ROOT / 'reports/classifier_next_experiment_20261010_171102'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
BUILD = ROOT / 'reports/multilingual_dataset_build_20261010_155129'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(EXP3))

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

INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def dump(filename, data):
    p = OUT / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def req(r):
    return DecisionRequest(
        r['utterance'],
        r['context'],
        tuple(
            Capability(c['name'], c['description'], tuple(c['required_arguments']))
            for c in r['capabilities']
        ),
    )


def exact(exp, pred):
    return all(
        (sorted(pred[k]) == sorted(exp[k]) if k == 'request_modes' else pred[k] == exp[k])
        for k in FIELDS
    )


def evaluate_dataset(adapter, dataset):
    pred = []
    latencies = []
    errors = []

    for r in dataset:
        start = time.perf_counter()
        try:
            p = dict(adapter.predict(req(r)))
            ClassifierDecision.from_mapping(p)
            err = None
        except Exception as ex:
            p = None
            err = str(ex)
        latencies.append((time.perf_counter() - start) * 1000.0)
        pred.append(p)
        errors.append(err)

    agreement = [p is not None and exact(r['expected'], p) for r, p in zip(dataset, pred)]

    fields_score = {
        f: sum(
            p is not None
            and (
                sorted(p[f]) == sorted(r['expected'][f])
                if f == 'request_modes'
                else p[f] == r['expected'][f]
            )
            for r, p in zip(dataset, pred)
        )
        for f in FIELDS
    }

    # Top-1 capability accuracy
    cap_agreements = sum(
        p is not None and p['selected_capability'] == r['expected']['selected_capability']
        for r, p in zip(dataset, pred)
    )

    return {
        'exact_agreement': sum(agreement),
        'denominator': len(dataset),
        'capability_top1_agreement': cap_agreements,
        'fields': fields_score,
        'latency_ms': {
            'median': float(np.median(latencies)),
            'p95': float(np.percentile(latencies, 95)),
            'mean': float(np.mean(latencies)),
        },
        'inference_failures': sum(x is not None for x in errors),
        'predictions': [
            {
                'id': r.get('id', r.get('original_id')),
                'language': r.get('language', 'en'),
                'utterance': r['utterance'],
                'expected': r['expected'],
                'prediction': p,
                'exact': ok,
                'error': er,
            }
            for r, p, ok, er in zip(dataset, pred, agreement, errors)
        ],
    }


def main():
    print("=== JARVIS Classifier v4 Experiment ===")

    # 1. Verify inputs
    baseline_path = ROOT / 'benchmarks/fast_classifier/models/v3/compact_heads.joblib'
    v3_path = EXP3 / 'best_candidate/compact_heads.joblib'
    cache = js(PRIOR / 'feature_cache_manifest.json')

    INPUTS[str(baseline_path.resolve())] = sha(baseline_path)
    INPUTS[str(v3_path.resolve())] = sha(v3_path)
    INPUTS[str((PRIOR / 'provisional_train.jsonl').resolve())] = sha(PRIOR / 'provisional_train.jsonl')
    INPUTS[str((PRIOR / 'provisional_validation.jsonl').resolve())] = sha(PRIOR / 'provisional_validation.jsonl')
    INPUTS[str((PRIOR / 'frozen_features.npz').resolve())] = sha(PRIOR / 'frozen_features.npz')

    # Load frozen feature cache for validation replay
    vectors = np.load(PRIOR / 'frozen_features.npz')['vectors']
    index = {x['text_sha256']: x['index'] for x in cache['texts']}

    def encoded(text):
        return vectors[index[hashlib.sha256(text.encode()).hexdigest()]]

    class ReplayEncoder:
        def __init__(self):
            self.forward_calls = 0
            self.last_timings = {'preprocessing_ms': 0.1, 'model_inference_ms': 0.2}

        def encode(self, texts, **kwargs):
            self.forward_calls += 1
            return np.stack([encoded(t) for t in texts])

    # 2. Build Candidate v4 Artifact
    v4_artifact = joblib.load(v3_path)
    v4_artifact['adapter_id'] = 'experimental-v4-clean'
    v4_artifact['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    v4_artifact['ranker_version'] = 'clean-generalized-dual-ranker-v4'
    joblib.dump(v4_artifact, OUT / 'best_candidate/compact_heads.joblib')

    # 3. Evaluate v4 on Original 40-Record Validation Set
    val = jl(PRIOR / 'provisional_validation.jsonl')
    adapter_val = ExperimentalV4Adapter.__new__(ExperimentalV4Adapter)
    adapter_val.artifact = v4_artifact
    adapter_val.encoder = ReplayEncoder()
    adapter_val._adapter_id = 'experimental-v4-clean'
    adapter_val.inference_count = 0
    adapter_val.last_inference_metrics = {}

    v4_val_results = evaluate_dataset(adapter_val, val)
    dump('original_validation_metrics_v4.json', v4_val_results)

    print(f"\n1. Original Validation (40 records):")
    print(f"   Strict Exact: {v4_val_results['exact_agreement']} / {v4_val_results['denominator']} ({v4_val_results['exact_agreement']/v4_val_results['denominator']*100:.1f}%)")
    print(f"   Capability Top-1: {v4_val_results['capability_top1_agreement']} / {v4_val_results['denominator']} ({v4_val_results['capability_top1_agreement']/v4_val_results['denominator']*100:.1f}%)")
    print("   Per-field:")
    for f, count in v4_val_results['fields'].items():
        print(f"     {f:<22}: {count} / {v4_val_results['denominator']}")

    # 4. Evaluate v4 on New 24-Record Unseen Evaluation Corpus with Real Local Encoder
    print("\n2. Evaluating on New Unseen Evaluation Corpus with Real Local Encoder...")
    unseen_corpus = jl(OUT / 'unseen_evaluation_corpus_v4.jsonl')

    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    real_encoder = FrozenMultilingualEncoder(cache['encoder_snapshot'], 160)
    for p in real_encoder.model.parameters():
        p.requires_grad_(False)

    adapter_unseen = ExperimentalV4Adapter.__new__(ExperimentalV4Adapter)
    adapter_unseen.artifact = v4_artifact
    adapter_unseen.encoder = real_encoder
    adapter_unseen._adapter_id = 'experimental-v4-clean'
    adapter_unseen.inference_count = 0
    adapter_unseen.last_inference_metrics = {}

    unseen_results = evaluate_dataset(adapter_unseen, unseen_corpus)
    dump('unseen_corpus_metrics_v4.json', unseen_results)

    print(f"   Unseen Corpus Strict Exact: {unseen_results['exact_agreement']} / {unseen_results['denominator']} ({unseen_results['exact_agreement']/unseen_results['denominator']*100:.1f}%)")
    print(f"   Unseen Capability Top-1: {unseen_results['capability_top1_agreement']} / {unseen_results['denominator']} ({unseen_results['capability_top1_agreement']/unseen_results['denominator']*100:.1f}%)")
    print(f"   Real Encoder Latency: Median {unseen_results['latency_ms']['median']:.1f}ms | p95 {unseen_results['latency_ms']['p95']:.1f}ms")
    print("   Per-field:")
    for f, count in unseen_results['fields'].items():
        print(f"     {f:<22}: {count} / {unseen_results['denominator']}")

    # Language breakdown on unseen corpus
    unseen_languages = {}
    for r, p_info in zip(unseen_corpus, unseen_results['predictions']):
        lang = r['language']
        item = unseen_languages.setdefault(lang, {'denominator': 0, 'exact': 0, 'capability_top1': 0})
        item['denominator'] += 1
        item['exact'] += int(p_info['exact'])
        item['capability_top1'] += int(p_info['prediction']['selected_capability'] == r['expected']['selected_capability'])

    dump('unseen_languages_v4.json', unseen_languages)

    # 5. Abstention Curve on Unseen Corpus
    abstention_rows = []
    for thresh in [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]:
        accepted = []
        accepted_exact = []
        for r in unseen_corpus:
            p = adapter_unseen.predict_with_abstention(req(r), thresh)
            if not p.get('abstained'):
                accepted.append(r)
                if exact(r['expected'], p):
                    accepted_exact.append(r)
        abstention_rows.append({
            'threshold': thresh,
            'accepted': len(accepted),
            'accepted_exact': len(accepted_exact),
            'full_set_exact': len(accepted_exact),
            'denominator': len(unseen_corpus),
        })
    dump('abstention_metrics_v4.json', abstention_rows)

    # 6. Run Manifest
    manifest = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'inputs': INPUTS,
        'original_validation': {
            'denominator': 40,
            'v3_exact': 38,
            'v4_exact': v4_val_results['exact_agreement'],
            'v4_capability_top1': v4_val_results['capability_top1_agreement'],
        },
        'unseen_corpus': {
            'denominator': 24,
            'v4_exact': unseen_results['exact_agreement'],
            'v4_capability_top1': unseen_results['capability_top1_agreement'],
        },
        'artifacts': {
            'compact_heads': sha(OUT / 'best_candidate/compact_heads.joblib'),
            'adapter_v4': sha(OUT / 'experimental_adapter_v4.py'),
            'unseen_corpus': sha(OUT / 'unseen_evaluation_corpus_v4.jsonl'),
        },
    }
    dump('run_manifest.json', manifest)

    print("\nRun completed successfully.")


if __name__ == '__main__':
    main()

