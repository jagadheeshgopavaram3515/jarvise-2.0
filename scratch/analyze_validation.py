import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

bm = json.loads(Path('D:/JARVIS/reports/classifier_completion_20261010_163425/best_metrics.json').read_text(encoding='utf-8'))
preds = bm['predictions']
print(f"Total predictions: {len(preds)}, Exact matches: {bm['exact_agreement']}/{bm['denominator']}")

fields = ['intent', 'execution_path', 'request_modes', 'selected_capability', 'context_references', 'missing_information', 'needs_clarification', 'needs_escalation']

failures_by_field = {f: [] for f in fields}

for i, p in enumerate(preds):
    exp = p['expected']
    pred = p['prediction']
    exact = p['exact']
    diffs = []
    for f in fields:
        match = (sorted(pred[f]) == sorted(exp[f])) if f == 'request_modes' else (pred[f] == exp[f])
        if not match:
            failures_by_field[f].append((i, p['original_id'], pred[f], exp[f]))
            diffs.append(f"{f}: pred={pred[f]!r} != exp={exp[f]!r}")
    if not exact:
        print(f"[{i:02d}] {p['original_id']} FAIL ({len(diffs)} diffs):")
        for d in diffs:
            print(f"      {d}")
    else:
        print(f"[{i:02d}] {p['original_id']} EXACT MATCH")

print("\n" + "="*80)
print("ARCHIVE RECORDS CAPABILITY ACCURACY:")
archive_matches = 0
archive_total = 0
for p in preds:
    if 'archive' in p.get('original_id', ''):
        archive_total += 1
        exp_cap = p['expected']['selected_capability']
        pred_cap = p['prediction']['selected_capability']
        ok = (exp_cap == pred_cap)
        if ok:
            archive_matches += 1
        print(f"  {p['original_id']} | exp={exp_cap!r} | pred={pred_cap!r} | match={ok}")
print(f"Archive capability accuracy: {archive_matches}/{archive_total}")

print("\n" + "="*80)
print("NON-ARCHIVE RECORDS SUMMARY:")
for p in preds:
    if 'archive' not in p.get('original_id', ''):
        ok = p['exact']
        print(f"  {p['original_id']} | exact={ok}")

