"""Inspect candidate ranker diagnostics for app launching."""
import json
from pathlib import Path

ROOT = Path('D:/JARVIS')
EVAL_DIR = ROOT / 'reports/classifier_v4_frozen_100_eval_20261010_180014'

preds = json.loads((EVAL_DIR / 'predictions_100.json').read_text(encoding='utf-8'))

for p in preds:
    if p['id'] in ('frozen100-025', 'frozen100-026', 'frozen100-047', 'frozen100-096'):
        print(f"ID: {p['id']} | Utterance: {p['utterance']}")
        print(f"Expected: {p['expected']}")
        print(f"Prediction: {p['prediction']}")
        print("=" * 60)

