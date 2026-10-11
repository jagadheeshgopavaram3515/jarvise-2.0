"""Forensic analysis script for all 55 discrepancies in the 100-request evaluation."""
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path('D:/JARVIS')
EVAL_DIR = ROOT / 'reports/classifier_v4_frozen_100_eval_20261010_180014'

preds = json.loads((EVAL_DIR / 'predictions_100.json').read_text(encoding='utf-8'))
discs = json.loads((EVAL_DIR / 'discrepancies_and_failure_modes.json').read_text(encoding='utf-8'))

print(f"Total records: {len(preds)}, Discrepancies: {len(discs)}")

# 1. Failure counts by field
field_mismatch_counts = Counter()
for d in discs:
    for f in d['mismatched_fields']:
        field_mismatch_counts[f] += 1

print("\n--- Field Mismatch Distribution ---")
for f, count in field_mismatch_counts.most_common():
    print(f"{f:<25}: {count} / 55 ({count/len(discs)*100:.1f}%)")

# 2. Correlated field errors
combos = Counter()
for d in discs:
    combos[tuple(sorted(d['mismatched_fields'].keys()))] += 1

print("\n--- Most Common Error Combinations ---")
for combo, cnt in combos.most_common(10):
    print(f"Count {cnt:>2}: {', '.join(combo)}")

# 3. Categorize by root cause hypothesis
categories = defaultdict(list)

for d in discs:
    mf = d['mismatched_fields']
    exp = d['expected']
    pred = d['full_prediction']
    lang = d['language']
    grp = d['scenario_group']
    utt = d['utterance']

    # Category A: Native Indic Script Embedding Drift
    # If language in (hi, te, ta) and capability was None when expected was an action
    if lang in ('hi', 'te', 'ta') and exp['selected_capability'] is not None and pred['selected_capability'] is None:
        categories['Native_Indic_Embedding_Drift'].append(d)
    
    # Category B: Clarification Contract Slot Mismatch
    # If needs_clarification is True in both or predicted, but missing_information name diverges
    elif 'missing_information' in mf and exp.get('needs_clarification') and pred.get('needs_clarification'):
        categories['Clarification_Slot_Name_Divergence'].append(d)
    
    # Category C: Multi-step vs Single-Action Decomposition
    elif 'execution_path' in mf and exp['execution_path'] == 'multi_step_task' and pred['execution_path'] == 'single_action':
        categories['Multistep_Defaulted_To_SingleAction'].append(d)
    
    # Category D: Desktop App Launch / Shell Confused
    elif grp == 'corpus100-app':
        categories['Desktop_App_Command_Mismatch'].append(d)
    
    # Category E: Web Search vs Direct Q&A
    elif grp in ('corpus100-search', 'corpus100-direct') and ('selected_capability' in mf or 'execution_path' in mf):
        categories['Search_Vs_Direct_Conversational_Ambiguity'].append(d)
        
    # Category F: Webpage Ingestion / URL Feasibility
    elif grp == 'corpus100-webpage':
        categories['Webpage_URL_Argument_Handling'].append(d)
        
    # Category G: Unsupported Action Escalation
    elif grp == 'corpus100-escalate':
        categories['Escalation_Classification_Failure'].append(d)
        
    # Other
    else:
        categories['Other_Specific_Discrepancies'].append(d)

print("\n--- Forensic Categories Breakdown ---")
for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
    print(f"{cat:<45}: {len(items):>2} records")

# Print detailed samples for each category
for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
    print(f"\n=======================================================")
    print(f"CATEGORY: {cat} ({len(items)} records)")
    print(f"=======================================================")
    for item in items[:5]:
        print(f"ID: {item['id']} | Lang: {item['language']} | Group: {item['scenario_group']}")
        print(f"Utterance: {item['utterance']}")
        print(f"Expected:  {item['expected']}")
        print(f"Predicted: {item['full_prediction']}")
        print(f"Diffs:     {item['mismatched_fields']}")
        print("-" * 50)

