"""Inspect candidate ranker failures in detail and save to utf-8 text file."""
import json
from pathlib import Path

ROOT = Path('D:/JARVIS')
OUT_DIR = ROOT / 'reports/jarvis_brain_forensic_improvement_20261010_183000'

findings = json.loads((OUT_DIR / 'forensic_findings.json').read_text(encoding='utf-8'))
ranker_failures = [f for f in findings if f['root_cause_type'] == 'capability_ranker_scoring_error']

lines = [f"Total capability ranker failures: {len(ranker_failures)}\n"]
for f in ranker_failures:
    lines.append(f"ID: {f['id']} | Group: {f['scenario_group']} | Lang: {f['language']}")
    lines.append(f"Utterance: {f['utterance']}")
    lines.append(f"Expected Cap:  {f['expected']['selected_capability']}")
    lines.append(f"Predicted Cap: {f['predicted']['selected_capability']}")
    lines.append(f"Expected Path: {f['expected']['execution_path']} | Predicted Path: {f['predicted']['execution_path']}")
    lines.append("-" * 60)

Path('scratch/ranker_failures_utf8.txt').write_text('\n'.join(lines), encoding='utf-8')
print(f"Written {len(ranker_failures)} records to scratch/ranker_failures_utf8.txt")

