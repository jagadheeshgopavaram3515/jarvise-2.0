import sys, json, collections
from pathlib import Path

ROOT = Path('D:/JARVIS')
EXP = ROOT / 'reports/classifier_next_experiment_20261010_171102'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP))

from experimental_adapter_v3 import ExperimentalV3Adapter
from assistant.orchestration.decision import DecisionRequest, Capability

val = [json.loads(line) for line in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]
best_m = json.loads((EXP / 'best_metrics.json').read_text(encoding='utf-8'))

FIELDS = ('intent', 'execution_path', 'request_modes', 'selected_capability', 'needs_clarification', 'needs_escalation', 'missing_information', 'context_references')

def exact(exp, pred):
    return all(
        (sorted(pred[k]) == sorted(exp[k]) if k == 'request_modes' else pred[k] == exp[k])
        for k in FIELDS
    )

groups = collections.defaultdict(list)
for r in val:
    groups[r['group_id']].append(r)

print(f"Total Scenario Groups in Validation: {len(groups)}")
pred_by_id = {p['record_id']: p['prediction'] for p in best_m['predictions']}

for gid, rlist in groups.items():
    exact_count = sum(exact(r['expected'], pred_by_id[r['id']]) for r in rlist)
    types = collections.Counter(r['expected']['execution_path'] for r in rlist)
    languages = collections.Counter(r['language'] for r in rlist)
    print(f"\nGroup: {gid} | Count: {len(rlist)} | Exact: {exact_count}/{len(rlist)} ({exact_count/len(rlist)*100:.1f}%)")
    print(f"  Sample IDs: {[r['original_id'] for r in rlist]}")
    print(f"  Paths: {dict(types)}")
    print(f"  Languages: {dict(languages)}")

