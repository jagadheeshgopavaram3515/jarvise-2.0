"""Generate deep capability metrics, confusion matrices, and safety audit for Conditions A, B, and C."""
import csv
import json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path('D:/JARVIS')
EXP_DIR = ROOT / 'reports/jarvis_transliteration_experiment_20261010_205500'
FROZEN_100_DIR = ROOT / 'reports/classifier_v4_frozen_100_eval_20261010_180014'

corpus = [json.loads(l) for l in (FROZEN_100_DIR / 'unseen_evaluation_corpus_100.jsonl').read_text(encoding='utf-8').splitlines()]
resA = json.loads((EXP_DIR / 'results_100_condition_A.json').read_text(encoding='utf-8'))
resB = json.loads((EXP_DIR / 'results_100_condition_B.json').read_text(encoding='utf-8'))
resC = json.loads((EXP_DIR / 'results_100_condition_C.json').read_text(encoding='utf-8'))

cm_dir = EXP_DIR / 'confusion_matrices'
cm_dir.mkdir(parents=True, exist_ok=True)

exp_caps = [str(r['expected']['selected_capability']) for r in corpus]
all_possible_caps = sorted(list(set(exp_caps + [
    str(p['selected_capability']) for res in (resA, resB, resC) for p in res['predictions'] if p is not None
])))

deep_audit = {}

for name, res in [('Condition_A_Native', resA), ('Condition_B_Translit', resB), ('Condition_C_Combined', resC)]:
    pred_caps = [str(p['selected_capability']) if p is not None else 'None' for p in res['predictions']]
    
    # 1. Confusion Matrix CSV
    cm = confusion_matrix(exp_caps, pred_caps, labels=all_possible_caps)
    with (cm_dir / f'capability_confusion_{name}.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Expected \\ Predicted'] + all_possible_caps)
        for cname, row in zip(all_possible_caps, cm):
            writer.writerow([cname] + list(row))
            
    # 2. Per-capability Classification Report
    report = classification_report(exp_caps, pred_caps, labels=all_possible_caps, output_dict=True, zero_division=0)
    
    # 3. Clarification precision & unnecessary clarification rate
    # Expected clarification: r['expected']['needs_clarification'] is True
    # Predicted clarification: p['needs_clarification'] is True
    exp_clar = [r['expected']['needs_clarification'] for r in corpus]
    pred_clar = [p['needs_clarification'] if p else False for p in res['predictions']]
    
    true_pos_clar = sum(e and p for e, p in zip(exp_clar, pred_clar))
    pred_pos_clar = sum(pred_clar)
    unnecessary_clar = sum((not e) and p for e, p in zip(exp_clar, pred_clar))
    clar_precision = (true_pos_clar / pred_pos_clar * 100.0) if pred_pos_clar else 0.0
    unnecessary_clar_rate = (unnecessary_clar / (len(corpus) - sum(exp_clar)) * 100.0)
    
    # 4. Out-of-domain Escalation Correctness
    # Expected escalation: r['expected']['needs_escalation'] is True
    exp_esc = [r['expected']['needs_escalation'] for r in corpus]
    pred_esc = [p['needs_escalation'] if p else False for p in res['predictions']]
    true_pos_esc = sum(e and p for e, p in zip(exp_esc, pred_esc))
    pred_pos_esc = sum(pred_esc)
    esc_correctness = (true_pos_esc / sum(exp_esc) * 100.0) if sum(exp_esc) else 0.0
    false_escalation_count = sum((not e) and p for e, p in zip(exp_esc, pred_esc))
    
    # 5. Dangerous Inappropriate Action Triggers
    # Expected None (clarify, escalate, conversational), but model selected an action capability
    inappropriate_actions = sum(
        (r['expected']['selected_capability'] is None) and (p is not None and p['selected_capability'] is not None)
        for r, p in zip(corpus, res['predictions'])
    )

    deep_audit[name] = {
        'classification_report': report,
        'clarification': {
            'expected_clarification_count': sum(exp_clar),
            'predicted_clarification_count': pred_pos_clar,
            'true_positive_clarification': true_pos_clar,
            'clarification_precision_pct': clar_precision,
            'unnecessary_clarifications': unnecessary_clar,
            'unnecessary_clarification_rate_pct': unnecessary_clar_rate
        },
        'escalation': {
            'expected_escalations': sum(exp_esc),
            'predicted_escalations': pred_pos_esc,
            'correct_escalations': true_pos_esc,
            'escalation_recall_pct': esc_correctness,
            'false_escalations': false_escalation_count
        },
        'safety': {
            'inappropriate_action_triggers': inappropriate_actions,
            'inappropriate_action_rate_pct': inappropriate_actions / len(corpus) * 100.0
        }
    }

(EXP_DIR / 'deep_capability_metrics.json').write_text(json.dumps(deep_audit, indent=2), encoding='utf-8')
print("Successfully generated deep capability metrics and confusion matrices.")
for name, data in deep_audit.items():
    print(f"\n--- {name} ---")
    print(f"Clarification Precision:       {data['clarification']['clarification_precision_pct']:.1f}% ({data['clarification']['true_positive_clarification']}/{data['clarification']['predicted_clarification_count']})")
    print(f"Unnecessary Clarification Rate: {data['clarification']['unnecessary_clarification_rate_pct']:.1f}% ({data['clarification']['unnecessary_clarifications']} false alarms)")
    print(f"Escalation Recall:              {data['escalation']['escalation_recall_pct']:.1f}% ({data['escalation']['correct_escalations']}/{data['escalation']['expected_escalations']})")
    print(f"Inappropriate Action Triggers:  {data['safety']['inappropriate_action_triggers']} / 100 ({data['safety']['inappropriate_action_rate_pct']:.1f}%)")

