import json
from pathlib import Path

ROOT = Path('D:/JARVIS')
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
val = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]

for r in val:
    print(f"{r['original_id']:<35} | path={r['expected']['execution_path']:<15} | modes={r['expected']['request_modes']}")

