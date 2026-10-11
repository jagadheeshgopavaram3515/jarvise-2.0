import sys, json, hashlib, collections, time, re
from pathlib import Path
import numpy as np

ROOT = Path('D:/JARVIS')
EXP = ROOT / 'reports/classifier_next_experiment_20261010_171102'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
BUILD = ROOT / 'reports/multilingual_dataset_build_20261010_155129'
COMP = ROOT / 'reports/classifier_completion_20261010_163425'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(EXP))

from assistant.orchestration.decision import DecisionRequest, Capability, ClassifierDecision
from assistant.orchestration.compact_adapter import (
    CompactMultilingualAdapter, FrozenMultilingualEncoder,
    serialize_request, serialize_capability
)
from dataset_tools import observed_scripts

FIELDS = ('intent', 'execution_path', 'request_modes', 'selected_capability', 'needs_clarification', 'needs_escalation', 'missing_information', 'context_references')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

print("=== CHECK 1: REPRODUCE 38/40 EXACT AGREEMENT FROM SAVED PREDICTIONS ===")
val = [json.loads(line) for line in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]
best_m = json.loads((EXP / 'best_metrics.json').read_text(encoding='utf-8'))

def exact(exp, pred):
    return all(
        (sorted(pred[k]) == sorted(exp[k]) if k == 'request_modes' else pred[k] == exp[k])
        for k in FIELDS
    )

saved_exact_count = 0
for r, p_info in zip(val, best_m['predictions']):
    p = p_info['prediction']
    exp = r['expected']
    ok = exact(exp, p)
    if ok:
        saved_exact_count += 1
    else:
        diffs = [k for k in FIELDS if (sorted(p[k]) != sorted(exp[k]) if k == 'request_modes' else p[k] != exp[k])]
        print(f"Non-exact: {r['original_id']} -> diffs: {diffs}")

print(f"Verified Saved Predictions Exact Agreement: {saved_exact_count} / {len(val)}")

print("\n=== CHECK 5: ARCHIVE-FOLDER PERFORMANCE VS OTHER EXAMPLES ===")
archive_records = [r for r in val if 'synth_open_archive_folder' in r['original_id']]
other_records = [r for r in val if 'synth_open_archive_folder' not in r['original_id']]

archive_groups = {r['group_id'] for r in archive_records}
other_groups = {r['group_id'] for r in other_records}
total_groups = {r['group_id'] for r in val}

print(f"Total Validation Records: {len(val)} across {len(total_groups)} scenario groups: {total_groups}")
print(f"Archive Folder Records: {len(archive_records)} across {len(archive_groups)} scenario group(s): {archive_groups}")
print(f"Other Records: {len(other_records)} across {len(other_groups)} scenario group(s): {other_groups}")

pred_by_id = {p['record_id']: p['prediction'] for p in best_m['predictions']}
archive_exact = sum(exact(r['expected'], pred_by_id[r['id']]) for r in archive_records)
other_exact = sum(exact(r['expected'], pred_by_id[r['id']]) for r in other_records)

print(f"Archive-Folder Strict Exact: {archive_exact} / {len(archive_records)} ({archive_exact/len(archive_records)*100:.1f}%)")
print(f"Other Examples Strict Exact: {other_exact} / {len(other_records)} ({other_exact/len(other_records)*100:.1f}%)")

print("\n=== CHECK 3: TRAIN VS VALIDATION INDEPENDENCE ===")
train = [json.loads(line) for line in (PRIOR / 'provisional_train.jsonl').read_text(encoding='utf-8').splitlines()]
train_groups = {r['group_id'] for r in train}
val_groups = {r['group_id'] for r in val}
group_overlap = train_groups & val_groups

train_texts = {r['utterance'].strip().lower() for r in train}
val_texts = {r['utterance'].strip().lower() for r in val}
text_overlap = train_texts & val_texts

print(f"Train Records: {len(train)} across {len(train_groups)} groups")
print(f"Validation Records: {len(val)} across {len(val_groups)} groups")
print(f"Scenario Group Overlap: {len(group_overlap)} (Groups: {group_overlap})")
print(f"Normalized Utterance Exact Overlap: {len(text_overlap)} (Texts: {text_overlap})")

# Check template similarity
train_caps = collections.Counter(r['expected']['selected_capability'] for r in train)
val_caps = collections.Counter(r['expected']['selected_capability'] for r in val)
print(f"Train Capability Distribution: {dict(train_caps)}")
print(f"Val Capability Distribution: {dict(val_caps)}")

