"""Targeted regression test suite verifying fixes for confirmed forensic failure modes."""
import re
import json
import joblib
import numpy as np
import pytest
from pathlib import Path

ROOT = Path('D:/JARVIS')
EVAL_DIR = ROOT / 'reports/classifier_v4_frozen_100_eval_20261010_180014'
EXP4 = ROOT / 'reports/classifier_v4_experiment_20261010_175140'
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
OUT_DIR = ROOT / 'reports/jarvis_brain_forensic_improvement_20261010_183000'

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP4))
sys.path.insert(0, str(OUT_DIR))

from assistant.orchestration.decision import Capability, DecisionRequest, ClassifierDecision
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder

@pytest.fixture(scope="module")
def encoder_and_artifact():
    cache_manifest = json.loads((PRIOR / 'feature_cache_manifest.json').read_text(encoding='utf-8'))
    encoder = FrozenMultilingualEncoder(cache_manifest['encoder_snapshot'], 160)
    for p in encoder.model.parameters():
        p.requires_grad_(False)
    artifact = joblib.load(EXP4 / 'best_candidate/compact_heads.joblib')
    return encoder, artifact

def test_app_terms_prevent_ambiguity_rejection(encoder_and_artifact):
    """Test frozen100-025: 'Open PowerShell terminal application' selects open_app, not None."""
    encoder, artifact = encoder_and_artifact
    from experimental_adapter_v5 import ExperimentalV5Adapter
    adapter = ExperimentalV5Adapter.__new__(ExperimentalV5Adapter)
    adapter.artifact = artifact
    adapter.encoder = encoder
    adapter._adapter_id = 'test-v5'
    adapter.inference_count = 0
    adapter.last_inference_metrics = {}

    req = DecisionRequest(
        user_text="Open PowerShell terminal application",
        context={},
        capabilities=(
            Capability("open_app", "Launch an installed application or program.", ("app_name",)),
            Capability("open_folder", "Open a directory or folder in file explorer.", ("path",)),
        )
    )
    res = adapter.predict(req)
    assert res['selected_capability'] == 'open_app', f"Expected open_app, got {res['selected_capability']}"
    assert res['execution_path'] == 'single_action'

def test_page_terms_and_url_prevent_ambiguity_rejection(encoder_and_artifact):
    """Test frozen100-047: URL with 'Read the article' selects read_page, not None."""
    encoder, artifact = encoder_and_artifact
    from experimental_adapter_v5 import ExperimentalV5Adapter
    adapter = ExperimentalV5Adapter.__new__(ExperimentalV5Adapter)
    adapter.artifact = artifact
    adapter.encoder = encoder
    adapter._adapter_id = 'test-v5'
    adapter.inference_count = 0
    adapter.last_inference_metrics = {}

    req = DecisionRequest(
        user_text="Read the article at https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
        context={},
        capabilities=(
            Capability("read_page", "Fetch and extract clean readable text from a web URL.", ("url",)),
            Capability("google_search", "Search the web for up to date information.", ("query",)),
        )
    )
    res = adapter.predict(req)
    assert res['selected_capability'] == 'read_page', f"Expected read_page, got {res['selected_capability']}"
    assert res['execution_path'] == 'single_action'

def test_native_indic_script_dense_scaling(encoder_and_artifact):
    """Test frozen100-007: Telugu script 'నా డౌన్‌లోడ్స్ ఫోల్డర్ ఓపెన్ చెయ్యి' selects open_folder."""
    encoder, artifact = encoder_and_artifact
    from experimental_adapter_v5 import ExperimentalV5Adapter
    adapter = ExperimentalV5Adapter.__new__(ExperimentalV5Adapter)
    adapter.artifact = artifact
    adapter.encoder = encoder
    adapter._adapter_id = 'test-v5'
    adapter.inference_count = 0
    adapter.last_inference_metrics = {}

    req = DecisionRequest(
        user_text="నా డౌన్‌లోడ్స్ ఫోల్డర్ ఓపెన్ చెయ్యి",
        context={},
        capabilities=(
            Capability("open_folder", "Open a directory or folder in file explorer.", ("path",)),
            Capability("find_file", "Search and locate files by name or extension.", ("query",)),
        )
    )
    res = adapter.predict(req)
    assert res['selected_capability'] == 'open_folder', f"Expected open_folder, got {res['selected_capability']}"
    assert res['execution_path'] == 'single_action'

def test_out_of_domain_escalation_preserved(encoder_and_artifact):
    """Test frozen100-096: Banking transfer routes to escalate with selected_capability=None."""
    encoder, artifact = encoder_and_artifact
    from experimental_adapter_v5 import ExperimentalV5Adapter
    adapter = ExperimentalV5Adapter.__new__(ExperimentalV5Adapter)
    adapter.artifact = artifact
    adapter.encoder = encoder
    adapter._adapter_id = 'test-v5'
    adapter.inference_count = 0
    adapter.last_inference_metrics = {}

    req = DecisionRequest(
        user_text="Transfer five hundred dollars from my savings account to John",
        context={},
        capabilities=(
            Capability("open_app", "Launch an installed application or program.", ("app_name",)),
        )
    )
    res = adapter.predict(req)
    assert res['selected_capability'] is None, f"Expected None, got {res['selected_capability']}"
    assert res['execution_path'] == 'escalate', f"Expected escalate, got {res['execution_path']}"
    assert res['needs_escalation'] is True

def test_pronoun_reference_span_extraction(encoder_and_artifact):
    """Test frozen100-056: 'Delete that completely' extracts reference='that' or 'it' with kind='item'."""
    encoder, artifact = encoder_and_artifact
    from experimental_adapter_v5 import ExperimentalV5Adapter
    adapter = ExperimentalV5Adapter.__new__(ExperimentalV5Adapter)
    adapter.artifact = artifact
    adapter.encoder = encoder
    adapter._adapter_id = 'test-v5'
    adapter.inference_count = 0
    adapter.last_inference_metrics = {}

    req = DecisionRequest(
        user_text="Delete that completely",
        context={},
        capabilities=(
            Capability("delete_file", "Delete a file permanently.", ("file_path",)),
        )
    )
    res = adapter.predict(req)
    assert len(res['context_references']) == 1
    ref = res['context_references'][0]
    assert ref['kind'] == 'item'
    assert ref['reference'] in ('that', 'it')

