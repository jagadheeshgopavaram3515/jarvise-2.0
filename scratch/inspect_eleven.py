import json
from pathlib import Path

ROOT = Path('D:/JARVIS')
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
VAL = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]
LIMIT = json.loads((PRIOR / 'deterministic_output_limits.json').read_text(encoding='utf-8'))
eleven_ids = {x['record_id'] for x in LIMIT['records']}
eleven_records = [r for r in VAL if r['id'] in eleven_ids]

print(f"Eleven records count: {len(eleven_records)}")
for r in eleven_records:
    print("=" * 60)
    print(f"ID: {r['original_id']}")
    print(f"Utterance: {r['utterance']}")
    print(f"Context: {r['context']}")
    print(f"Expected path: {r['expected']['execution_path']}")
    print(f"Expected intent: {r['expected']['intent']}")
    print(f"Expected refs: {r['expected']['context_references']}")
    print(f"Expected miss: {r['expected']['missing_information']}")

