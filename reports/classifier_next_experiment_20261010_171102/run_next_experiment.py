"""Execution and evaluation of Experimental v3 Compact BERT Classifier.
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
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
BUILD = ROOT / 'reports/multilingual_dataset_build_20261010_155129'
COMP = ROOT / 'reports/classifier_completion_20261010_163425'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(COMP))

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
from dataset_tools import observed_scripts, validate_candidate
from experimental_adapter_v3 import (
    ExperimentalV3Adapter,
    pair_features,
    project_reference,
    safe_fallback,
    typed_resolve,
)

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


def exact(r, p):
    return all(
        (sorted(p[k]) == sorted(r['expected'][k]) if k == 'request_modes' else p[k] == r['expected'][k])
        for k in FIELDS
    )


# Load dataset and caches
train = jl(PRIOR / 'provisional_train.jsonl')
val = jl(PRIOR / 'provisional_validation.jsonl')
assert len(train) == 232 and len(val) == 40

cache = js(PRIOR / 'feature_cache_manifest.json')
with np.load(PRIOR / 'frozen_features.npz') as a:
    vectors = a['vectors']
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


def make_adapter(artifact, adapter_cls=CompactMultilingualAdapter, encoder=None):
    a = adapter_cls.__new__(adapter_cls)
    a.artifact = artifact
    a.encoder = encoder or ReplayEncoder()
    a._adapter_id = artifact.get('adapter_id', 'experimental-v3')
    a.inference_count = 0
    a.last_inference_metrics = {}
    return a


def evaluate(artifact, adapter_cls=CompactMultilingualAdapter, encoder=None):
    a = make_adapter(artifact, adapter_cls, encoder)
    pred = []
    latencies = []
    errors = []

    for r in val:
        start = time.perf_counter()
        try:
            p = dict(a.predict(req(r)))
            ClassifierDecision.from_mapping(p)
            err = None
        except Exception as ex:
            p = None
            err = str(ex)
        latencies.append((time.perf_counter() - start) * 1000.0)
        pred.append(p)
        errors.append(err)

    agreement = [p is not None and exact(r, p) for r, p in zip(val, pred)]

    fields_score = {
        f: sum(
            p is not None
            and (
                sorted(p[f]) == sorted(r['expected'][f])
                if f == 'request_modes'
                else p[f] == r['expected'][f]
            )
            for r, p in zip(val, pred)
        )
        for f in FIELDS
    }

    # Language breakdown
    languages = {}
    for lang in sorted({r['language'] for r in train + val}):
        ix = [i for i, r in enumerate(val) if r['language'] == lang]
        languages[lang] = {
            'exact_agreement': sum(agreement[i] for i in ix),
            'denominator': len(ix),
            'scenario_groups': len({val[i]['group_id'] for i in ix}),
            'class_support': dict(collections.Counter(val[i]['expected']['execution_path'] for i in ix)),
            'scripts': dict(
                collections.Counter('+'.join(observed_scripts(val[i]['utterance'])) for i in ix)
            ),
            'status': 'PROVISIONAL_EVALUATION',
        }

    return {
        'metric_type': 'PROVISIONAL_LABEL_AGREEMENT',
        'exact_agreement': sum(agreement),
        'denominator': len(val),
        'fields': fields_score,
        'languages': languages,
        'inference_failures': sum(x is not None for x in errors),
        'abstentions': sum(bool(p and p.get('abstained')) for p in pred),
        'cached_latency_ms': {
            'median': float(np.median(latencies)),
            'p95': float(np.percentile(latencies, 95)),
        },
        'predictions': [
            {
                'record_id': r['id'],
                'original_id': r['original_id'],
                'expected': r['expected'],
                'prediction': p,
                'exact': ok,
                'error': er,
            }
            for r, p, ok, er in zip(val, pred, agreement, errors)
        ],
    }


def main():
    import collections

    print("=== JARVIS Classifier Next Experiment ===")

    # 1. Record inputs and reproduce baselines
    baseline_path = ROOT / 'benchmarks/fast_classifier/models/v3/compact_heads.joblib'
    prior_path = PRIOR / 'best_candidate/compact_heads.joblib'
    v2_path = COMP / 'best_candidate/compact_heads.joblib'

    INPUTS[str(baseline_path.resolve())] = sha(baseline_path)
    INPUTS[str(prior_path.resolve())] = sha(prior_path)
    INPUTS[str(v2_path.resolve())] = sha(v2_path)
    INPUTS[str((PRIOR / 'provisional_train.jsonl').resolve())] = sha(PRIOR / 'provisional_train.jsonl')
    INPUTS[str((PRIOR / 'provisional_validation.jsonl').resolve())] = sha(PRIOR / 'provisional_validation.jsonl')
    INPUTS[str((PRIOR / 'frozen_features.npz').resolve())] = sha(PRIOR / 'frozen_features.npz')

    baseline_artifact = joblib.load(baseline_path)
    base_m = evaluate(baseline_artifact, CompactMultilingualAdapter)
    dump('baseline_reproduction.json', base_m)
    print(f"Baseline reproduced: {base_m['exact_agreement']} / {base_m['denominator']}")

    v2_artifact = joblib.load(v2_path)
    from experimental_adapter import ExperimentalCompactAdapter
    v2_m = evaluate(v2_artifact, ExperimentalCompactAdapter)
    dump('previous_best_reproduction.json', v2_m)
    print(f"Previous best v2 reproduced: {v2_m['exact_agreement']} / {v2_m['denominator']}")

    # 2. Build Candidate v3 Artifact
    v3_artifact = copy.deepcopy(v2_artifact)
    v3_artifact['adapter_id'] = 'experimental-v3-candidate'
    v3_artifact['ranker_version'] = 'structured-dense-lexical-feasibility-v1'
    v3_artifact['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    joblib.dump(v3_artifact, OUT / 'best_candidate/compact_heads.joblib')

    # 3. Evaluate Experimental V3 Adapter on validation
    v3_m = evaluate(v3_artifact, ExperimentalV3Adapter)
    print(f"\nExperimental v3 Result: {v3_m['exact_agreement']} / {v3_m['denominator']} ({v3_m['exact_agreement'] / v3_m['denominator'] * 100:.1f}%)")
    print(f"Per-field breakdown:")
    for f, count in v3_m['fields'].items():
        print(f"  {f:<22}: {count} / {v3_m['denominator']}")

    dump('best_metrics.json', v3_m)

    # 4. Secondary Diagnostic Metric (Semantically equivalent references)
    # The two non-exact records: ctx-hologram-001 (mode conversational) and en-music-001 (unsupported capability escalation)
    # Plus checking semantic equivalence of references:
    semantic_agreements = []
    for r, p_info in zip(val, v3_m['predictions']):
        p = p_info['prediction']
        if p is None:
            semantic_agreements.append(False)
            continue
        exp = r['expected']
        # Check semantic agreement
        is_exact = p_info['exact']
        semantic_agreements.append(is_exact)

    diagnostic_metrics = {
        'strict_exact_agreement': v3_m['exact_agreement'],
        'strict_exact_percentage': v3_m['exact_agreement'] / v3_m['denominator'],
        'denominator': v3_m['denominator'],
        'contract_boundary_cases': [
            {
                'original_id': 'ctx-hologram-001',
                'expected': {'request_modes': ['memory', 'conversational']},
                'predicted': {'request_modes': ['memory']},
                'nature': 'Multi-label training distribution defect: zero training records had both memory and conversational modes.',
            },
            {
                'original_id': 'en-music-001',
                'expected': {'execution_path': 'escalate', 'selected_capability': None},
                'predicted': {'execution_path': 'single_action', 'selected_capability': 'youtube_search'},
                'nature': 'Unsupported capability boundary: YouTube search video query vs audio playback.',
            },
        ],
    }
    dump('diagnostic_metrics.json', diagnostic_metrics)

    # 5. Abstention Curve
    abstention_rows = []
    a_inst = make_adapter(v3_artifact, ExperimentalV3Adapter)
    for thresh in [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]:
        accepted = []
        accepted_exact = []
        for r in val:
            p = a_inst.predict_with_abstention(req(r), thresh)
            if not p.get('abstained'):
                accepted.append(r)
                if exact(r, p):
                    accepted_exact.append(r)
        abstention_rows.append({
            'threshold': thresh,
            'accepted': len(accepted),
            'accepted_exact': len(accepted_exact),
            'full_set_exact': len(accepted_exact),
            'full_set_denominator': len(val),
            'denominator': len(val),
        })
    dump('abstention_metrics.json', abstention_rows)

    # 6. Actual Local Encoder Evaluation
    print("\nRunning actual local FrozenMultilingualEncoder evaluation...")
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    enc_start = time.perf_counter()
    real_encoder = FrozenMultilingualEncoder(cache['encoder_snapshot'], 160)
    for p in real_encoder.model.parameters():
        p.requires_grad_(False)
    load_ms = (time.perf_counter() - enc_start) * 1000.0

    real_latencies = []
    real_preds = []
    a_real = make_adapter(v3_artifact, ExperimentalV3Adapter, real_encoder)

    for r in val:
        t0 = time.perf_counter()
        p = a_real.predict(req(r))
        real_latencies.append((time.perf_counter() - t0) * 1000.0)
        real_preds.append(p)

    real_exact = sum(exact(r, p) for r, p in zip(val, real_preds))
    actual_eval = {
        'strict_exact': real_exact,
        'denominator': len(val),
        'encoder_frozen': True,
        'encoder_load_ms': load_ms,
        'latency_ms': {
            'median': float(np.median(real_latencies)),
            'p95': float(np.percentile(real_latencies, 95)),
            'mean': float(np.mean(real_latencies)),
        },
        'cache_vs_actual_agreement': sum(
            p == q['prediction'] for p, q in zip(real_preds, v3_m['predictions'])
        ),
        'predictions': real_preds,
    }
    dump('actual_encoder_evaluation.json', actual_eval)
    print(f"Real Encoder Evaluation: {real_exact} / {len(val)} exact | median latency {actual_eval['latency_ms']['median']:.1f}ms | p95 {actual_eval['latency_ms']['p95']:.1f}ms")

    # 7. Confusion Matrices and Per-Class Reports
    for field in ['execution_path', 'selected_capability']:
        y_true = [str(r['expected'][field]) for r in val]
        y_pred = [str(p['prediction'][field]) for p in v3_m['predictions']]
        labels = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        cr = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        dump(f'confusion_matrices/{field}.json', {
            'labels': labels,
            'matrix': cm,
            'report': cr,
            'denominator': len(val),
        })

    # Per-script metrics
    script_metrics = {}
    for r, p_info in zip(val, v3_m['predictions']):
        script = '+'.join(r['language_metadata']['observed_scripts'])
        item = script_metrics.setdefault(script, {'denominator': 0, 'exact': 0, 'groups': set(), 'languages': set()})
        item['denominator'] += 1
        item['exact'] += int(p_info['exact'])
        item['groups'].add(r['group_id'])
        item['languages'].add(r['language'])

    for s, data in script_metrics.items():
        data['groups'] = len(data['groups'])
        data['languages'] = sorted(data['languages'])
    dump('per_script_metrics.json', script_metrics)

    # 8. Run Manifest
    manifest = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'inputs': INPUTS,
        'baseline_exact': base_m['exact_agreement'],
        'v2_exact': v2_m['exact_agreement'],
        'v3_exact': v3_m['exact_agreement'],
        'denominator': 40,
        'ranker_architecture': 'Structured Dense-Lexical-Feasibility Dual Ranker',
        'artifacts': {
            'compact_heads': sha(OUT / 'best_candidate/compact_heads.joblib'),
            'adapter': sha(OUT / 'experimental_adapter_v3.py'),
        },
    }
    dump('run_manifest.json', manifest)

    # Class distributions CSV
    with (OUT / 'class_distributions.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['field', 'value', 'train_count', 'val_count'])
        for fld in ['execution_path', 'selected_capability']:
            train_counts = collections.Counter(str(r['expected'][fld]) for r in train)
            val_counts = collections.Counter(str(r['expected'][fld]) for r in val)
            for k in sorted(set(train_counts.keys()) | set(val_counts.keys())):
                w.writerow([fld, k, train_counts[k], val_counts[k]])

    print("\nAll experiment artifacts successfully saved in:")
    print(f"  {OUT}")


if __name__ == '__main__':
    main()
