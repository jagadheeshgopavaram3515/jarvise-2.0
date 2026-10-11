"""Trace candidate ranker scores for specific records."""
import json
import joblib
import numpy as np
from pathlib import Path

ROOT = Path('D:/JARVIS')
EVAL_DIR = ROOT / 'reports/classifier_v4_frozen_100_eval_20261010_180014'
EXP4 = ROOT / 'reports/classifier_v4_experiment_20261010_175140'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP4))
from assistant.orchestration.decision import DecisionRequest, Capability
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder
from experimental_adapter_v4 import ExperimentalV4Adapter

cache_manifest = json.loads((PRIOR / 'feature_cache_manifest.json').read_text(encoding='utf-8'))
encoder = FrozenMultilingualEncoder(cache_manifest['encoder_snapshot'], 160)
v4_artifact = joblib.load(EXP4 / 'best_candidate/compact_heads.joblib')

adapter = ExperimentalV4Adapter.__new__(ExperimentalV4Adapter)
adapter.artifact = v4_artifact
adapter.encoder = encoder
adapter._adapter_id = 'trace'
adapter.inference_count = 0
adapter.last_inference_metrics = {}

corpus = [json.loads(l) for l in (EVAL_DIR / 'unseen_evaluation_corpus_100.jsonl').read_text(encoding='utf-8').splitlines()]

for r in corpus:
    if r['id'] in ('frozen100-025', 'frozen100-026', 'frozen100-047', 'frozen100-096'):
        req = DecisionRequest(
            r['utterance'],
            r['context'],
            tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities'])
        )
        pred = adapter.predict(req)
        print(f"ID: {r['id']} | Utterance: {r['utterance']}")
        print(f"Diagnostics: {adapter.last_diagnostics}")
        print(f"Last cap: {adapter.last_selected_capability} with score {adapter.last_capability_score}")
        for c, s in zip(r['capabilities'], adapter.last_diagnostics.get('capability_scores', [])):
            print(f"   {c['name']:<20}: {s:.4f}")
        print("=" * 60)

