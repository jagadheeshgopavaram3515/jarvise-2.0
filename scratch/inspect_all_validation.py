import json
from pathlib import Path

ROOT = Path('D:/JARVIS')
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
VAL = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]
print(f'Total validation records: {len(VAL)}')
for i, r in enumerate(VAL):
    exp = r['expected']
    print(f"{i:02d}: {r['original_id']:<35} | path={exp['execution_path']:<13} | cap={str(exp['selected_capability']):<15} | refs={len(exp['context_references'])} | miss={len(exp['missing_information'])}")

