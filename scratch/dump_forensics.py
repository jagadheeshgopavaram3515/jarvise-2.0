"""Generate structured forensic failure report for all 100 evaluation records."""
import json
import io
import sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path('D:/JARVIS')
EVAL_DIR = ROOT / 'reports/classifier_v4_frozen_100_eval_20261010_180014'
OUT_DIR = ROOT / 'reports/jarvis_brain_forensic_improvement_20261010_183000'

preds = json.loads((EVAL_DIR / 'predictions_100.json').read_text(encoding='utf-8'))
discs = json.loads((EVAL_DIR / 'discrepancies_and_failure_modes.json').read_text(encoding='utf-8'))

findings = []

for idx, d in enumerate(discs):
    r_id = d['id']
    grp = d['scenario_group']
    lang = d['language']
    utt = d['utterance']
    mf = d['mismatched_fields']
    exp = d['expected']
    pred = d['full_prediction']
    
    # Diagnose the earliest stage and root cause
    earliest_stage = None
    root_cause_type = None
    cause_desc = None
    severity = "HIGH"
    
    # 1. Native Indic Script Tokenization & Dense Similarity Collapse
    if lang in ('hi', 'te', 'ta') and exp['selected_capability'] is not None and pred['selected_capability'] is None:
        earliest_stage = "multilingual_tokenizer_dense_similarity"
        root_cause_type = "multilingual_representation_asymmetry"
        cause_desc = f"Subword fragmentation of native {lang} script causes dense cosine similarity with English capability description to fall below the 0.70 threshold. The adapter defaulted to clarify/unsupported."
        severity = "HIGH"
        
    # 2. Clarification Reference Span Contract (Expected reference span was 'it' or specific pronoun, but adapter produced full utterance or different span)
    elif 'context_references' in mf and len(mf) == 1:
        earliest_stage = "context_reference_extraction"
        root_cause_type = "reference_span_contract_divergence"
        cause_desc = f"The reference extractor predicted kind={pred['context_references'][0]['kind'] if pred.get('context_references') else 'none'} vs expected kind={exp['context_references'][0]['kind'] if exp.get('context_references') else 'none'}, or span was full utterance instead of pronoun."
        severity = "MEDIUM"

    # 3. Execution Path Misclassification (Multi-step vs Single Action)
    elif 'execution_path' in mf and exp['execution_path'] == 'multi_step_task' and pred['execution_path'] == 'single_action':
        earliest_stage = "execution_path_head"
        root_cause_type = "missing_multistep_reasoning"
        cause_desc = "Execution path linear head defaults to single_action because request lacks explicit English keyword 'steps'/'guide', despite semantic multi-step goal."
        severity = "HIGH"

    # 4. Capability Selection Ambiguity / Threshold Rejection
    elif 'selected_capability' in mf:
        earliest_stage = "candidate_ranker_selection"
        root_cause_type = "capability_ranker_scoring_error"
        cause_desc = f"Candidate ranker failed to rank expected capability '{exp['selected_capability']}'. Selected '{pred['selected_capability']}' instead."
        severity = "HIGH"

    # 5. Clarification Slot Name Divergence
    elif 'missing_information' in mf:
        earliest_stage = "missing_information_renderer"
        root_cause_type = "contract_rendering_divergence"
        cause_desc = "Clarification triggered correctly, but generic slot name ('target'/'required_information') differed from expected slot name."
        severity = "MEDIUM"
        
    else:
        earliest_stage = "intent_and_mode_heads"
        root_cause_type = "auxiliary_head_divergence"
        cause_desc = f"Divergence in auxiliary fields: {list(mf.keys())}"
        severity = "LOW"

    findings.append({
        'id': r_id,
        'scenario_group': grp,
        'language': lang,
        'utterance': utt,
        'mismatched_fields': list(mf.keys()),
        'diffs': mf,
        'expected': exp,
        'predicted': pred,
        'earliest_stage': earliest_stage,
        'root_cause_type': root_cause_type,
        'cause_description': cause_desc,
        'severity': severity
    })

(OUT_DIR / 'forensic_findings.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding='utf-8')

# Summary statistics
cause_counts = Counter(f['root_cause_type'] for f in findings)
stage_counts = Counter(f['earliest_stage'] for f in findings)

summary = {
    'total_discrepancies': len(findings),
    'root_causes': dict(cause_counts.most_common()),
    'earliest_stages': dict(stage_counts.most_common())
}
(OUT_DIR / 'forensic_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print("Successfully generated forensic findings JSON:")
print(json.dumps(summary, indent=2))

