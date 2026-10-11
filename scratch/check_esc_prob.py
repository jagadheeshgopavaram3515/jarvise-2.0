"""Check escalation head on 096."""
import json
import joblib
from pathlib import Path

ROOT = Path('D:/JARVIS')
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
EXP4 = ROOT / 'reports/classifier_v4_experiment_20261010_175140'

import sys
sys.path.insert(0, str(ROOT))
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder, _positive_probability, serialize_request
from assistant.orchestration.decision import DecisionRequest

cache_manifest = json.loads((PRIOR / 'feature_cache_manifest.json').read_text(encoding='utf-8'))
encoder = FrozenMultilingualEncoder(cache_manifest['encoder_snapshot'], 160)
artifact = joblib.load(EXP4 / 'best_candidate/compact_heads.joblib')

for text in [
    "Transfer five hundred dollars from my savings account to John",
    "Uber app lo airport ki cab book cheyyi",
    "Open it up right now",
    "Delete that completely"
]:
    req = DecisionRequest(text, {}, ())
    vec = encoder.encode([serialize_request(req)])
    esc_prob = _positive_probability(artifact['escalation'], vec)
    cla_prob = _positive_probability(artifact['clarification'], vec)
    print(f"Utterance: {text}")
    print(f"   Escalation prob:   {esc_prob:.4f}")
    print(f"   Clarification prob: {cla_prob:.4f}")
    print("-" * 50)

