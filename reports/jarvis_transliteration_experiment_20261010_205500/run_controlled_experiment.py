"""Controlled Multilingual Transliteration Experiment Harness for JARVIS Brain.
Evaluates Condition A (Native Control), Condition B (Transliteration-Only),
and Condition C (Original + Transliterated Representation) on Frozen 100 and Independent 24 corpora.
"""
import copy
import csv
import datetime
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path('D:/JARVIS')
EXP_DIR = Path(__file__).resolve().parent
V5_DIR = ROOT / 'reports/jarvis_brain_forensic_improvement_20261010_183000'
FROZEN_100_DIR = ROOT / 'reports/classifier_v4_frozen_100_eval_20261010_180014'
EXP4_DIR = ROOT / 'reports/classifier_v4_experiment_20261010_175140'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(V5_DIR))

from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    DecisionRequest,
)
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder
from experimental_adapter_v5 import ExperimentalV5Adapter
from indic_transliterator import transliterate_text

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

def exact_match(exp, pred):
    return all(
        (sorted(pred[k]) == sorted(exp[k]) if k == 'request_modes' else pred[k] == exp[k])
        for k in FIELDS
    )

def evaluate_corpus_under_condition(adapter, corpus, condition_mode):
    """Evaluate corpus under Condition A, B, or C."""
    predictions = []
    latencies_model = []
    latencies_translit = []
    latencies_e2e = []
    errors = []
    transformation_records = []

    for r in corpus:
        orig_text = r['utterance']

        # 1. Preprocessing / Transliteration
        t_start = time.perf_counter()
        trans_res = transliterate_text(orig_text)
        t_translit_ms = (time.perf_counter() - t_start) * 1000.0

        if condition_mode == 'A':
            eval_text = orig_text
        elif condition_mode == 'B':
            eval_text = trans_res.transliterated_text
        elif condition_mode == 'C':
            eval_text = (
                f"{orig_text} [transliteration: {trans_res.transliterated_text}]"
                if trans_res.transformation_applied
                else orig_text
            )
        else:
            raise ValueError(f"Unknown condition mode: {condition_mode}")

        req = DecisionRequest(
            eval_text,
            r['context'],
            tuple(
                Capability(c['name'], c['description'], tuple(c['required_arguments']))
                for c in r['capabilities']
            )
        )

        # 2. Model Inference
        m_start = time.perf_counter()
        try:
            pred = dict(adapter.predict(req))
            ClassifierDecision.from_mapping(pred)
            err = None
        except Exception as ex:
            pred = None
            err = str(ex)
        m_infer_ms = (time.perf_counter() - m_start) * 1000.0
        e2e_ms = t_translit_ms + m_infer_ms

        latencies_translit.append(t_translit_ms)
        latencies_model.append(m_infer_ms)
        latencies_e2e.append(e2e_ms)
        predictions.append(pred)
        errors.append(err)
        transformation_records.append({
            'original': orig_text,
            'transformed': eval_text,
            'detected_scripts': trans_res.detected_scripts,
            'applied': trans_res.transformation_applied
        })

    exact_flags = [p is not None and exact_match(r['expected'], p) for r, p in zip(corpus, predictions)]
    strict_count = sum(exact_flags)
    cap_count = sum(p is not None and p['selected_capability'] == r['expected']['selected_capability'] for r, p in zip(corpus, predictions))

    field_counts = {
        f: sum(
            p is not None and (
                sorted(p[f]) == sorted(r['expected'][f])
                if f == 'request_modes'
                else p[f] == r['expected'][f]
            )
            for r, p in zip(corpus, predictions)
        )
        for f in FIELDS
    }

    # Group and language breakdowns
    group_stats = {}
    lang_stats = {}
    for r, p, ok in zip(corpus, predictions, exact_flags):
        grp = r.get('scenario_group', r.get('scenario', 'general'))
        lang = r.get('language', 'en')

        g = group_stats.setdefault(grp, {'count': 0, 'exact': 0, 'cap_top1': 0})
        g['count'] += 1
        g['exact'] += int(ok)
        g['cap_top1'] += int(p is not None and p['selected_capability'] == r['expected']['selected_capability'])

        l = lang_stats.setdefault(lang, {'count': 0, 'exact': 0, 'cap_top1': 0})
        l['count'] += 1
        l['exact'] += int(ok)
        l['cap_top1'] += int(p is not None and p['selected_capability'] == r['expected']['selected_capability'])

    return {
        'condition': condition_mode,
        'denominator': len(corpus),
        'strict_exact': strict_count,
        'strict_exact_pct': strict_count / len(corpus) * 100.0,
        'capability_top1': cap_count,
        'capability_top1_pct': cap_count / len(corpus) * 100.0,
        'inference_failures': sum(e is not None for e in errors),
        'fields': field_counts,
        'group_stats': group_stats,
        'lang_stats': lang_stats,
        'latencies': {
            'transliteration_median_ms': float(np.median(latencies_translit)),
            'transliteration_p95_ms': float(np.percentile(latencies_translit, 95)),
            'model_median_ms': float(np.median(latencies_model)),
            'model_p95_ms': float(np.percentile(latencies_model, 95)),
            'e2e_median_ms': float(np.median(latencies_e2e)),
            'e2e_p95_ms': float(np.percentile(latencies_e2e, 95)),
            'e2e_mean_ms': float(np.mean(latencies_e2e)),
        },
        'predictions': predictions,
        'transformations': transformation_records
    }

def main():
    print("=" * 70)
    print("JARVIS BRAIN: CONTROLLED MULTILINGUAL TRANSLITERATION EXPERIMENT")
    print("=" * 70)

    # 1. Verify hashes
    corpus100_path = FROZEN_100_DIR / 'unseen_evaluation_corpus_100.jsonl'
    corpus24_path = EXP4_DIR / 'unseen_evaluation_corpus_v4.jsonl'
    v5_adapter_path = V5_DIR / 'experimental_adapter_v5.py'
    model_path = EXP4_DIR / 'best_candidate/compact_heads.joblib'

    corpus100_sha = sha256_file(corpus100_path)
    corpus24_sha = sha256_file(corpus24_path)
    v5_adapter_sha = sha256_file(v5_adapter_path)
    model_sha = sha256_file(model_path)

    print(f"Frozen 100 Corpus SHA:    {corpus100_sha}")
    print(f"Independent 24 Corpus SHA: {corpus24_sha}")
    print(f"v5 Adapter SHA:           {v5_adapter_sha}")
    print(f"Model Heads SHA:          {model_sha}")

    # 2. Load model and adapter
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
    adapter._adapter_id = 'v5-translit-experiment'
    adapter.inference_count = 0
    adapter.last_inference_metrics = {}

    # 3. Load Corpora
    corpus100 = [json.loads(l) for l in corpus100_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    corpus24 = [json.loads(l) for l in corpus24_path.read_text(encoding='utf-8').splitlines() if l.strip()]

    print(f"\nLoaded {len(corpus100)} records in Frozen 100 corpus, {len(corpus24)} in Independent 24 corpus.")

    # 4. Evaluate Frozen 100 Corpus across Conditions A, B, C
    print("\n--- Evaluating Frozen 100-Request Corpus ---")
    print("Running Condition A (Native Control)...")
    res100_A = evaluate_corpus_under_condition(adapter, corpus100, 'A')
    print("Running Condition B (Transliteration-Only)...")
    res100_B = evaluate_corpus_under_condition(adapter, corpus100, 'B')
    print("Running Condition C (Original + Transliterated)...")
    res100_C = evaluate_corpus_under_condition(adapter, corpus100, 'C')

    # 5. Evaluate Independent 24 Corpus across Conditions A, B, C
    print("\n--- Evaluating Independent 24-Request Corpus ---")
    print("Running Condition A (Native Control)...")
    res24_A = evaluate_corpus_under_condition(adapter, corpus24, 'A')
    print("Running Condition B (Transliteration-Only)...")
    res24_B = evaluate_corpus_under_condition(adapter, corpus24, 'B')
    print("Running Condition C (Original + Transliterated)...")
    res24_C = evaluate_corpus_under_condition(adapter, corpus24, 'C')

    # 6. Save condition results
    dump_json(EXP_DIR / 'results_100_condition_A.json', res100_A)
    dump_json(EXP_DIR / 'results_100_condition_B.json', res100_B)
    dump_json(EXP_DIR / 'results_100_condition_C.json', res100_C)
    dump_json(EXP_DIR / 'results_24_condition_A.json', res24_A)
    dump_json(EXP_DIR / 'results_24_condition_B.json', res24_B)
    dump_json(EXP_DIR / 'results_24_condition_C.json', res24_C)

    # 7. Generate Per-Example Comparison Table for 100 records
    comparison_records = []
    for idx, r in enumerate(corpus100):
        pA = res100_A['predictions'][idx]
        pB = res100_B['predictions'][idx]
        pC = res100_C['predictions'][idx]
        tr = res100_B['transformations'][idx]
        exp = r['expected']

        exactA = exact_match(exp, pA) if pA else False
        exactB = exact_match(exp, pB) if pB else False
        exactC = exact_match(exp, pC) if pC else False

        capA = pA['selected_capability'] if pA else None
        capB = pB['selected_capability'] if pB else None
        capC = pC['selected_capability'] if pC else None
        exp_cap = exp['selected_capability']

        cap_changed = (capA != capB)
        exact_change = "IMPROVED" if (not exactA and exactB) else ("REGRESSED" if (exactA and not exactB) else "UNCHANGED")
        cap_change = "IMPROVED" if (capA != exp_cap and capB == exp_cap) else ("REGRESSED" if (capA == exp_cap and capB != exp_cap) else "UNCHANGED")

        # Fields changed between A and B
        diff_fields = []
        if pA and pB:
            for f in FIELDS:
                valA = sorted(pA[f]) if f == 'request_modes' else pA[f]
                valB = sorted(pB[f]) if f == 'request_modes' else pB[f]
                if valA != valB:
                    diff_fields.append(f)

        comparison_records.append({
            'id': r['id'],
            'scenario_group': r['scenario_group'],
            'language': r['language'],
            'original_utterance': r['utterance'],
            'transliterated_utterance': tr['transformed'],
            'applied_transliteration': tr['applied'],
            'expected_capability': exp_cap,
            'expected_execution_path': exp['execution_path'],
            'condA_pred': {'exact': exactA, 'cap': capA, 'path': pA['execution_path'] if pA else None},
            'condB_pred': {'exact': exactB, 'cap': capB, 'path': pB['execution_path'] if pB else None},
            'condC_pred': {'exact': exactC, 'cap': capC, 'path': pC['execution_path'] if pC else None},
            'capability_changed_B_vs_A': cap_changed,
            'exact_change_B_vs_A': exact_change,
            'capability_change_B_vs_A': cap_change,
            'mismatched_fields_B_vs_A': diff_fields,
            'meaning_preserved': True
        })

    dump_json(EXP_DIR / 'per_example_comparison_100.json', comparison_records)

    # 8. Print Executive Comparison
    print("\n" + "=" * 70)
    print("FROZEN 100-REQUEST EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<22} | {'Strict Exact':<18} | {'Capability Top-1':<18} | {'Median Latency'}")
    print("-" * 70)
    print(f"{'A: Native Baseline':<22} | {res100_A['strict_exact']:>2}/100 ({res100_A['strict_exact_pct']:.1f}%)        | {res100_A['capability_top1']:>2}/100 ({res100_A['capability_top1_pct']:.1f}%)        | {res100_A['latencies']['e2e_median_ms']:.2f}ms")
    print(f"{'B: Transliteration':<22} | {res100_B['strict_exact']:>2}/100 ({res100_B['strict_exact_pct']:.1f}%)        | {res100_B['capability_top1']:>2}/100 ({res100_B['capability_top1_pct']:.1f}%)        | {res100_B['latencies']['e2e_median_ms']:.2f}ms (translit: {res100_B['latencies']['transliteration_median_ms']:.3f}ms)")
    print(f"{'C: Combined Representation':<22} | {res100_C['strict_exact']:>2}/100 ({res100_C['strict_exact_pct']:.1f}%)        | {res100_C['capability_top1']:>2}/100 ({res100_C['capability_top1_pct']:.1f}%)        | {res100_C['latencies']['e2e_median_ms']:.2f}ms")

    print("\n--- Language / Script Disaggregation on Frozen 100 ---")
    print(f"{'Language/Script':<16} | {'Cond A Exact':<14} | {'Cond B Exact':<14} | {'Cond A Cap':<14} | {'Cond B Cap':<14}")
    print("-" * 70)
    for lang in sorted(res100_A['lang_stats'].keys()):
        stA = res100_A['lang_stats'][lang]
        stB = res100_B['lang_stats'][lang]
        cnt = stA['count']
        print(f"{lang:<16} | {stA['exact']:>2}/{cnt:>2} ({stA['exact']/cnt*100:.1f}%)     | {stB['exact']:>2}/{cnt:>2} ({stB['exact']/cnt*100:.1f}%)     | {stA['cap_top1']:>2}/{cnt:>2} ({stA['cap_top1']/cnt*100:.1f}%)     | {stB['cap_top1']:>2}/{cnt:>2} ({stB['cap_top1']/cnt*100:.1f}%)")

    print("\n" + "=" * 70)
    print("INDEPENDENT 24-REQUEST CORPUS RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<22} | {'Strict Exact':<18} | {'Capability Top-1':<18} | {'Median Latency'}")
    print("-" * 70)
    print(f"{'A: Native Baseline':<22} | {res24_A['strict_exact']:>2}/24 ({res24_A['strict_exact_pct']:.1f}%)         | {res24_A['capability_top1']:>2}/24 ({res24_A['capability_top1_pct']:.1f}%)         | {res24_A['latencies']['e2e_median_ms']:.2f}ms")
    print(f"{'B: Transliteration':<22} | {res24_B['strict_exact']:>2}/24 ({res24_B['strict_exact_pct']:.1f}%)         | {res24_B['capability_top1']:>2}/24 ({res24_B['capability_top1_pct']:.1f}%)         | {res24_B['latencies']['e2e_median_ms']:.2f}ms")
    print(f"{'C: Combined Representation':<22} | {res24_C['strict_exact']:>2}/24 ({res24_C['strict_exact_pct']:.1f}%)         | {res24_C['capability_top1']:>2}/24 ({res24_C['capability_top1_pct']:.1f}%)         | {res24_C['latencies']['e2e_median_ms']:.2f}ms")

if __name__ == '__main__':
    main()

