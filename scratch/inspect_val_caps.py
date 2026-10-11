import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path('D:/JARVIS')
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'

val = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

for i, r in enumerate(val):
    exp_cap = r['expected']['selected_capability']
    if exp_cap:
        print(f"[{i:02d}] {r['original_id']}:")
        print(f"    utterance: {r['utterance']}")
        print(f"    expected:  {exp_cap}")
        print(f"    available: {[c['name'] for c in r['capabilities']]}")

