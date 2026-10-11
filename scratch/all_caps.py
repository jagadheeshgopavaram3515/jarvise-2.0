import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path('D:/JARVIS')
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'

val = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
train = [json.loads(l) for l in (PRIOR / 'provisional_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

print("Unique capabilities across train and val:")
all_caps = {}
for r in train + val:
    for c in r['capabilities']:
        all_caps[c['name']] = c

for name, c in sorted(all_caps.items()):
    print(f"  {name}: {c['description']} | required: {c['required_arguments']}")

