"""Check why memory and direct groups failed in v5."""
import json
from pathlib import Path

ROOT = Path('D:/JARVIS')
EXP_DIR = ROOT / 'reports/jarvis_brain_forensic_improvement_20261010_183000'

discs = json.loads((EXP_DIR / 'discrepancies_v5.json').read_text(encoding='utf-8'))

for d in discs:
    if d['scenario_group'] in ('corpus100-memory', 'corpus100-direct'):
        print(f"ID: {d['id']} | Group: {d['scenario_group']}")
        print(f"Utterance: {d['utterance']}")
        print(f"Expected Path: {d['expected']['execution_path']} | Pred Path: {d['prediction']['execution_path']}")
        print(f"Expected Intent: {d['expected']['intent']} | Pred Intent: {d['prediction']['intent']}")
        print(f"Diffs: {d['mismatched_fields']}")
        print("-" * 50)

