import json
from pathlib import Path
from collections import Counter

train = [json.loads(l) for l in Path('reports/autonomous_bert_training_20261010_161410/provisional_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
val = [json.loads(l) for l in Path('reports/autonomous_bert_training_20261010_161410/provisional_validation.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

print(f"Train records: {len(train)}, with capabilities: {sum(bool(r['capabilities']) for r in train)}")
print(f"Val records: {len(val)}, with capabilities: {sum(bool(r['capabilities']) for r in val)}")

train_cap_names = Counter(c['name'] for r in train for c in r['capabilities'])
print("\nTrain all advertised capabilities count:", train_cap_names)

val_cap_names = Counter(c['name'] for r in val for c in r['capabilities'])
print("\nVal all advertised capabilities count:", val_cap_names)

train_pos_caps = Counter(r['expected']['selected_capability'] for r in train if r['expected']['selected_capability'])
print("\nTrain selected (positive) capabilities:", train_pos_caps)

val_pos_caps = Counter(r['expected']['selected_capability'] for r in val if r['expected']['selected_capability'])
print("\nVal selected (positive) capabilities:", val_pos_caps)

