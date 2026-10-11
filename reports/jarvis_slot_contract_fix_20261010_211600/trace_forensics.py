import json
from pathlib import Path
from collections import Counter

raw = json.load(open(r'D:\JARVIS\reports\jarvis_slot_contract_fix_20261010_211600\raw_mismatches.json', encoding='utf-8'))

def trace_records(records, corpus_label):
    diagnoses = []
    for r in records:
        mid = r['id']
        utt = r['utterance']
        ff = r['failed_fields']
        exp = r['expected']
        pred = r['predicted']
        caps = r['capabilities']
        req_args = r['req_args']
        scen = r.get('scenario_group')
        lang = r.get('language_code')
        
        # Determine category and root cause
        if ff == ['context_references']:
            cat = 'reference_resolution_or_annotation_artifact'
            comp = 'context_reference_extractor'
            evidence = f"Expected context_references {exp['context_references']} but predicted {pred['context_references']}. Utterance phrasing: '{utt}'."
            conf = 'HIGH'
        elif 'selected_capability' in ff and exp.get('selected_capability') is not None and pred.get('selected_capability') is not None:
            cat = 'capability_ranking_error'
            comp = 'capability_ranker'
            evidence = f"Expected capability '{exp['selected_capability']}' but ranker chose '{pred['selected_capability']}'. Available: {caps}."
            conf = 'HIGH'
        elif 'selected_capability' in ff and exp.get('selected_capability') is not None and pred.get('selected_capability') is None:
            if pred.get('execution_path') in ('clarify', 'escalate'):
                cat = 'semantic_misclassification_false_abstention'
                comp = 'execution_path_or_calibration'
                evidence = f"Expected action with '{exp['selected_capability']}', but model abstained into '{pred['execution_path']}'."
                conf = 'HIGH'
            else:
                cat = 'semantic_misclassification'
                comp = 'classifier_head'
                evidence = f"Expected capability '{exp['selected_capability']}', but predicted None under path '{pred.get('execution_path')}'."
                conf = 'HIGH'
        elif 'selected_capability' in ff and exp.get('selected_capability') is None and pred.get('selected_capability') is not None:
            cat = 'semantic_misclassification_false_action'
            comp = 'capability_ranker_or_threshold'
            evidence = f"Expected no capability (path={exp.get('execution_path')}), but ranker committed to '{pred.get('selected_capability')}'."
            conf = 'HIGH'
        elif 'needs_escalation' in ff and exp.get('needs_escalation') == False and pred.get('needs_escalation') == True:
            cat = 'false_escalation'
            comp = 'escalation_head_or_threshold'
            evidence = f"Expected path '{exp.get('execution_path')}', but model escalated to 'escalate'."
            conf = 'HIGH'
        elif 'needs_clarification' in ff and exp.get('needs_clarification') == False and pred.get('needs_clarification') == True:
            cat = 'false_clarification'
            comp = 'clarification_head_or_threshold'
            evidence = f"Expected path '{exp.get('execution_path')}', but model triggered clarification."
            conf = 'HIGH'
        elif 'intent' in ff or 'execution_path' in ff:
            cat = 'path_or_intent_misclassification'
            comp = 'execution_path_head'
            evidence = f"Expected path '{exp.get('execution_path')}' / intent '{exp.get('intent')}', got '{pred.get('execution_path')}' / '{pred.get('intent')}'."
            conf = 'HIGH'
        else:
            cat = 'other_discrepancy'
            comp = 'adapter_postprocessing'
            evidence = f"Failed fields: {ff}."
            conf = 'MEDIUM'
            
        diagnoses.append({
            'corpus': corpus_label,
            'id': mid,
            'utterance': utt,
            'scenario_group': scen,
            'language_code': lang,
            'failed_fields': ff,
            'expected_values': exp,
            'actual_values': pred,
            'responsible_component': comp,
            'category': cat,
            'evidence': evidence,
            'confidence': conf
        })
    return diagnoses

d100 = trace_records(raw['m100'], 'Frozen 100')
d24 = trace_records(raw['m24'], 'Independent 24')

all_diagnoses = d100 + d24
out_file = Path(r'D:\JARVIS\reports\jarvis_slot_contract_fix_20261010_211600\forensic_diagnoses.json')
out_file.write_text(json.dumps(all_diagnoses, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Saved {len(all_diagnoses)} detailed forensic diagnoses to {out_file}')

counts = Counter(d['category'] for d in all_diagnoses)
print('\nAggregated Category Breakdown across both corpora (58 errors total):')
for c, cnt in counts.most_common():
    print(f'  {c}: {cnt}')

