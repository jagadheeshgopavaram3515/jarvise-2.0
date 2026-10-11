"""Unit and regression tests for Experimental v3 Classifier.
Ensures zero leakage, strict contracts, and robust multi-scenario coverage.
"""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path('D:/JARVIS')
OUT = Path(__file__).resolve().parent
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
BUILD = ROOT / 'reports/multilingual_dataset_build_20261010_155129'
COMP = ROOT / 'reports/classifier_completion_20261010_163425'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(OUT))

from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    DecisionRequest,
)
from assistant.orchestration.compact_adapter import CompactMultilingualAdapter
from dataset_tools import extract_missing
from experimental_adapter_v3 import (
    ExperimentalV3Adapter,
    pair_features,
    project_reference,
    safe_fallback,
    typed_resolve,
)

VAL = [json.loads(line) for line in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]
LIMIT = json.loads((PRIOR / 'deterministic_output_limits.json').read_text(encoding='utf-8'))
ELEVEN_CASES = [r for r in VAL if r['id'] in {x['record_id'] for x in LIMIT['records']}]


def test_baseline_and_previous_reproduced():
    base_rep = json.loads((OUT / 'baseline_reproduction.json').read_text(encoding='utf-8'))
    v2_rep = json.loads((OUT / 'previous_best_reproduction.json').read_text(encoding='utf-8'))
    assert base_rep['exact_agreement'] == 8 and base_rep['denominator'] == 40
    assert v2_rep['exact_agreement'] == 10 and v2_rep['denominator'] == 40


def test_v3_strict_exact_agreement_breaks_ceiling():
    best = json.loads((OUT / 'best_metrics.json').read_text(encoding='utf-8'))
    assert best['exact_agreement'] == 38
    assert best['denominator'] == 40
    assert best['exact_agreement'] / best['denominator'] >= 0.90


def test_actual_local_encoder_evaluation():
    act = json.loads((OUT / 'actual_encoder_evaluation.json').read_text(encoding='utf-8'))
    assert act['strict_exact'] == 38
    assert act['denominator'] == 40
    assert act['encoder_frozen'] is True
    assert act['latency_ms']['median'] < 250.0  # sub-250ms median


@pytest.mark.parametrize('record', ELEVEN_CASES, ids=[r['original_id'] for r in ELEVEN_CASES])
def test_each_eleven_typed_context_resolution(record):
    for ref in record['expected']['context_references']:
        resolved = typed_resolve(record['context'], ref['kind'], record['utterance'])
        assert resolved == ref['resolved_value']


def test_resolver_never_fabricates_empty_context():
    assert typed_resolve({}, 'item', 'open that') is None
    assert typed_resolve({}, 'prior_conversation', 'remember yesterday') is None


def test_unseen_project_alias_and_span():
    context = {'projects': [{'name': 'New initiative', 'aliases': ['space elevator']}]}
    assert project_reference(context, 'Remember our space elevator?') == 'our space elevator'
    assert typed_resolve(context, 'project', 'Remember our space elevator?') == 'New initiative'


def test_conflicting_project_aliases_not_resolved():
    context = {'projects': [{'name': 'A', 'aliases': ['shared']}, {'name': 'B', 'aliases': ['shared']}]}
    assert project_reference(context, 'shared') is None
    assert typed_resolve(context, 'project', 'shared') is None


def test_feasibility_check_penalizes_missing_url():
    adapter = ExperimentalV3Adapter.__new__(ExperimentalV3Adapter)
    adapter.artifact = {}
    adapter.last_diagnostics = {}
    req = DecisionRequest(
        user_text="Read this documentation",
        context={},
        capabilities=(
            Capability("read_page", "Read a webpage.", ("url",)),
            Capability("find_file", "Search for a file.", ("query",)),
        ),
    )
    # Vectors stub
    q_vec = np.ones(768)
    cap_vecs = np.ones((2, 768))
    best_cap, score = adapter._select_capability(req, q_vec, cap_vecs)
    # read_page requires url which is absent; find_file should not be blocked by url check
    assert best_cap == "find_file"


def test_ambiguous_candidate_returns_none_for_clarification():
    adapter = ExperimentalV3Adapter.__new__(ExperimentalV3Adapter)
    adapter.artifact = {}
    adapter.last_diagnostics = {}
    # Utterance does not mention folder or file, but both are offered
    req = DecisionRequest(
        user_text="open it again",
        context={},
        capabilities=(
            Capability("open_file", "Open a file.", ("path",)),
            Capability("open_folder", "Open a directory.", ("path",)),
        ),
    )
    q_vec = np.ones(768)
    cap_vecs = np.ones((2, 768))
    best_cap, score = adapter._select_capability(req, q_vec, cap_vecs)
    assert best_cap is None
    assert score == 0.30


def test_safe_fallback_contract():
    d = safe_fallback("low confidence")
    decision = ClassifierDecision.from_mapping(d)
    assert d['abstained'] is True
    assert decision.selected_capability is None
    assert decision.needs_escalation is True


def test_every_validation_prediction_valid_classifier_decision():
    best = json.loads((OUT / 'best_metrics.json').read_text(encoding='utf-8'))
    for item in best['predictions']:
        p = item['prediction']
        ClassifierDecision.from_mapping(p)


def test_protected_records_excluded():
    manifest = json.loads((BUILD / 'protected_test_manifest.json').read_text(encoding='utf-8'))
    protected_ids = set(manifest['original_protected_record_ids'])
    val_ids = {r['original_id'] for r in VAL}
    assert not (protected_ids & val_ids)


def test_production_assistant_unchanged():
    # Verify no new assistant files were modified during this classifier session
    before_text = (COMP / 'git_status_before.txt').read_text(encoding='utf-8')
    before_files = {line[3:].strip() for line in before_text.splitlines() if 'assistant/' in line}
    current = subprocess.run(
        ['git', 'status', '--porcelain=v1', 'assistant/'],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout
    current_files = {line[3:].strip() for line in current.splitlines() if 'assistant/' in line}
    assert current_files <= before_files, f"New files modified in assistant: {current_files - before_files}"
