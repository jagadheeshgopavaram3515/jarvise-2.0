import json
from pathlib import Path
ROOT = Path('D:/JARVIS')
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
train = [json.loads(l) for l in (PRIOR / 'provisional_train.jsonl').read_text(encoding='utf-8').splitlines()]
val = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]

for name, data in [('train', train), ('val', val)]:
    print(f'=== {name} ===')
    for r in data:
        miss = r['expected']['missing_information']
        if miss:
            print(f"{r.get('original_id', r.get('id'))}: {miss}")

