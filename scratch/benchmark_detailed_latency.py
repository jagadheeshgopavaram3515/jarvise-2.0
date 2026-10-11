import sys, json, time, platform
from pathlib import Path
import numpy as np
import torch

ROOT = Path('D:/JARVIS')
EXP = ROOT / 'reports/classifier_next_experiment_20261010_171102'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
BUILD = ROOT / 'reports/multilingual_dataset_build_20261010_155129'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(EXP))

from assistant.orchestration.decision import DecisionRequest, Capability
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder
from experimental_adapter_v3 import ExperimentalV3Adapter

cache = json.loads((PRIOR / 'feature_cache_manifest.json').read_text(encoding='utf-8'))
val = [json.loads(line) for line in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]

# Measure startup latency
t_start = time.perf_counter()
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
real_encoder = FrozenMultilingualEncoder(cache['encoder_snapshot'], 160)
for p in real_encoder.model.parameters():
    p.requires_grad_(False)
load_ms = (time.perf_counter() - t_start) * 1000.0

import joblib
artifact = joblib.load(EXP / 'best_candidate/compact_heads.joblib')
adapter = ExperimentalV3Adapter.__new__(ExperimentalV3Adapter)
adapter.artifact = artifact
adapter.encoder = real_encoder
adapter._adapter_id = artifact['adapter_id']
adapter.inference_count = 0
adapter.last_inference_metrics = {}

encoder_times = []
ranking_times = []
postprocess_times = []
total_times = []

# Warm up run on 5 samples
for r in val[:5]:
    req = DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))
    adapter.predict(req)

# Benchmark across all 40 records
for r in val:
    req = DecisionRequest(r['utterance'], r['context'], tuple(Capability(c['name'], c['description'], tuple(c['required_arguments'])) for c in r['capabilities']))
    t0 = time.perf_counter()
    p = adapter.predict(req)
    total_ms = (time.perf_counter() - t0) * 1000.0
    
    enc_ms = adapter.encoder.last_timings.get('model_inference_ms', 0.0) + adapter.encoder.last_timings.get('preprocessing_ms', 0.0)
    post_ms = adapter.last_inference_metrics.get('postprocessing_ms', 0.0)
    
    # Calculate ranking time vs postprocessing
    encoder_times.append(enc_ms)
    postprocess_times.append(post_ms)
    total_times.append(total_ms)

results = {
    'hardware': {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'torch_num_threads': torch.get_num_threads(),
    },
    'model_startup_ms': load_ms,
    'encoder_inference_ms': {
        'median': float(np.median(encoder_times)),
        'p95': float(np.percentile(encoder_times, 95)),
        'mean': float(np.mean(encoder_times)),
    },
    'postprocessing_ms': {
        'median': float(np.median(postprocess_times)),
        'p95': float(np.percentile(postprocess_times, 95)),
        'mean': float(np.mean(postprocess_times)),
    },
    'total_pipeline_ms': {
        'median': float(np.median(total_times)),
        'p95': float(np.percentile(total_times, 95)),
        'mean': float(np.mean(total_times)),
    },
}

print(json.dumps(results, indent=2))
Path('D:/JARVIS/reports/classifier_generalization_audit_20261010_173628/latency_benchmark.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

