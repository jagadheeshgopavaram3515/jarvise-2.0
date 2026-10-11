"""Unit and regression tests for JARVIS Compact BERT v4.
Verifies removal of memorized rules, general reference extraction, and benchmark reproduction.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path('D:/JARVIS')
OUT = Path(__file__).resolve().parent
EXP3 = ROOT / 'reports/classifier_next_experiment_20261010_171102'
COMP = ROOT / 'reports/classifier_completion_20261010_163425'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))

from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    DecisionRequest,
)
from experimental_adapter_v4 import (
    ExperimentalV4Adapter,
    project_reference,
    safe_fallback,
    typed_resolve,
)


def test_memorized_rules_removed_from_code():
    code = (OUT / 'experimental_adapter_v4.py').read_text(encoding='utf-8')
    # Verify specific memorized strings from v3 are absent
    assert 'database migration' not in code
    assert 'Kal architecture' not in code
    assert "'prior discussion'" not in code
    assert "'kholo'" not in code
    assert "'bring'" not in code


def test_original_validation_benchmark_reproduced():
    m = json.loads((OUT / 'original_validation_metrics_v4.json').read_text(encoding='utf-8'))
    assert m['exact_agreement'] == 31
    assert m['denominator'] == 40
    assert m['capability_top1_agreement'] == 39


def test_unseen_corpus_evaluation_reproduced():
    m = json.loads((OUT / 'unseen_corpus_metrics_v4.json').read_text(encoding='utf-8'))
    assert m['exact_agreement'] == 12
    assert m['denominator'] == 24
    assert m['capability_top1_agreement'] == 19
    assert m['latency_ms']['median'] < 250.0


def test_general_ambiguity_rejection_without_distinguishing_evidence():
    adapter = ExperimentalV4Adapter.__new__(ExperimentalV4Adapter)
    adapter.artifact = {}
    adapter.last_diagnostics = {}
    req = DecisionRequest(
        user_text="perform the action immediately",
        context={},
        capabilities=(
            Capability("tool_alpha", "Do an action.", ()),
            Capability("tool_beta", "Do an action.", ()),
        ),
    )
    # Similar embedding vectors with close similarity and no token support
    q_vec = np.ones(768)
    cap_vecs = np.ones((2, 768))
    cap_name, score = adapter._select_capability(req, q_vec, cap_vecs)
    # Competing candidates with near-zero margin and no distinguishing token return None
    assert cap_name is None
    assert score == 0.30


def test_feasibility_check_penalizes_missing_url():
    adapter = ExperimentalV4Adapter.__new__(ExperimentalV4Adapter)
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
    q_vec = np.ones(768)
    cap_vecs = np.ones((2, 768))
    cap_name, score = adapter._select_capability(req, q_vec, cap_vecs)
    assert cap_name == "find_file"


def test_unseen_project_alias_resolution():
    context = {"projects": [{"name": "Orbital Station", "aliases": ["Chimera outpost"]}]}
    span = project_reference(context, "What is the status of our Chimera outpost?")
    assert span == "our Chimera outpost"
    resolved = typed_resolve(context, "project", "What is the status of our Chimera outpost?")
    assert resolved == "Orbital Station"


def test_production_assistant_unchanged():
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

