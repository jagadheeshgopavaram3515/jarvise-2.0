"""Analyze transliteration regressions in detail."""
import json
from pathlib import Path

ROOT = Path('D:/JARVIS')
EXP_DIR = ROOT / 'reports/jarvis_transliteration_experiment_20261010_205500'

comps = json.loads((EXP_DIR / 'per_example_comparison_100.json').read_text(encoding='utf-8'))
regressed = [c for c in comps if c['exact_change_B_vs_A'] == 'REGRESSED' or c['capability_change_B_vs_A'] == 'REGRESSED']
improved = [c for c in comps if c['exact_change_B_vs_A'] == 'IMPROVED' or c['capability_change_B_vs_A'] == 'IMPROVED']

lines = [
    f"Total Regressed: {len(regressed)}",
    f"Total Improved: {len(improved)}\n",
    "=== REGRESSED EXAMPLES (B was worse than A) ==="
]

for c in regressed:
    lines.append(f"ID: {c['id']} | Lang: {c['language']} | Group: {c['scenario_group']}")
    lines.append(f"  Orig:       {c['original_utterance']}")
    lines.append(f"  Translit:   {c['transliterated_utterance']}")
    lines.append(f"  Expected:   Cap={c['expected_capability']}, Path={c['expected_execution_path']}")
    lines.append(f"  Cond A:     Exact={c['condA_pred']['exact']}, Cap={c['condA_pred']['cap']}, Path={c['condA_pred']['path']}")
    lines.append(f"  Cond B:     Exact={c['condB_pred']['exact']}, Cap={c['condB_pred']['cap']}, Path={c['condB_pred']['path']}")
    lines.append(f"  Diffs B vs A: {c['mismatched_fields_B_vs_A']}")
    lines.append("-" * 60)

lines.append("\n=== IMPROVED EXAMPLES (B was better than A) ===")
for c in improved:
    lines.append(f"ID: {c['id']} | Lang: {c['language']} | Group: {c['scenario_group']}")
    lines.append(f"  Orig:       {c['original_utterance']}")
    lines.append(f"  Translit:   {c['transliterated_utterance']}")
    lines.append(f"  Expected:   Cap={c['expected_capability']}, Path={c['expected_execution_path']}")
    lines.append(f"  Cond A:     Exact={c['condA_pred']['exact']}, Cap={c['condA_pred']['cap']}, Path={c['condA_pred']['path']}")
    lines.append(f"  Cond B:     Exact={c['condB_pred']['exact']}, Cap={c['condB_pred']['cap']}, Path={c['condB_pred']['path']}")
    lines.append(f"  Diffs B vs A: {c['mismatched_fields_B_vs_A']}")
    lines.append("-" * 60)

(EXP_DIR / 'transliteration_regression_analysis.txt').write_text('\n'.join(lines), encoding='utf-8')
print(f"Written regression analysis: {len(regressed)} regressed, {len(improved)} improved.")

