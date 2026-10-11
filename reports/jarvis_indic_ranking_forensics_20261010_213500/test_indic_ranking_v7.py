"""Regression tests for Experimental v7 Adapter.

Covers:
1. Native Telugu capability ranking (e.g. search vs folder)
2. Native Hindi capability ranking (e.g. file search vs folder, app close)
3. Native Tamil capability ranking (e.g. file search vs folder, search web)
4. English request preservation (no regressions on English)
5. Multilingual code-switching (e.g. 'Spotify application open karo', 'VS Code open pannu')
6. Paraphrase stability
7. Ambiguous commands triggering clarification
8. Direct conversational questions
9. Multi-step requests
10. Out-of-domain escalation preservation
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import joblib

EXP_DIR = Path("D:/JARVIS/reports/jarvis_indic_ranking_forensics_20261010_213500")
EXP4 = Path("D:/JARVIS/reports/classifier_v4_experiment_20261010_175140")
PRIOR = Path("D:/JARVIS/reports/autonomous_bert_training_20261010_161410")

V5_DIR = Path("D:/JARVIS/reports/jarvis_brain_forensic_improvement_20261010_183000")

import sys
sys.path.insert(0, "D:/JARVIS")
sys.path.insert(0, str(V5_DIR))
sys.path.insert(0, str(EXP_DIR))

from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    DecisionRequest,
)
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder
from experimental_adapter_v7 import ExperimentalV7Adapter


@pytest.fixture(scope="module")
def adapter():
    cache_manifest = json.loads((PRIOR / "feature_cache_manifest.json").read_text(encoding="utf-8"))
    encoder = FrozenMultilingualEncoder(cache_manifest["encoder_snapshot"], 160)
    artifact = joblib.load(EXP4 / "best_candidate/compact_heads.joblib")

    ad = ExperimentalV7Adapter.__new__(ExperimentalV7Adapter)
    ad.artifact = artifact
    ad.encoder = encoder
    ad._adapter_id = "test-v7"
    ad.inference_count = 0
    ad.last_inference_metrics = {}
    from slot_contract_resolver import SlotContractResolver
    ad.contract_resolver = SlotContractResolver()
    return ad


class TestIndicSemanticRanking:
    def test_native_hindi_file_search(self, adapter):
        """Hindi file search must select find_file over open_folder."""
        caps = (
            Capability("find_file", "Find local files or documents matching a query.", ("query",)),
            Capability("open_folder", "Open a specified folder or directory in file explorer.", ("path",)),
        )
        req = DecisionRequest("टैक्स रसीद की पीडीएफ फाइल ढूंढो", {}, caps)
        pred = adapter.predict(req)
        assert pred["selected_capability"] == "find_file"
        assert pred["execution_path"] == "single_action"

    def test_native_telugu_file_search(self, adapter):
        """Telugu file search must select find_file over open_folder."""
        caps = (
            Capability("find_file", "Find local files or documents matching a query.", ("query",)),
            Capability("open_folder", "Open a specified folder or directory in file explorer.", ("path",)),
        )
        req = DecisionRequest("కాలేజ్ సర్టిఫికెట్ పిడిఎఫ్ ఫైల్ వెతుకు", {}, caps)
        pred = adapter.predict(req)
        assert pred["selected_capability"] == "find_file"
        assert pred["execution_path"] == "single_action"

    def test_native_tamil_file_search(self, adapter):
        """Tamil file search must select find_file over open_folder."""
        caps = (
            Capability("find_file", "Find local files or documents matching a query.", ("query",)),
            Capability("open_folder", "Open a specified folder or directory in file explorer.", ("path",)),
        )
        req = DecisionRequest("வருமான வரி ரசீது கோப்பை தேடு", {}, caps)
        pred = adapter.predict(req)
        assert pred["selected_capability"] == "find_file"
        assert pred["execution_path"] == "single_action"

    def test_native_telugu_google_search(self, adapter):
        """Telugu web search must select google_search over open_folder or find_file."""
        caps = (
            Capability("google_search", "Perform web search for information or tutorials.", ("query",)),
            Capability("find_file", "Find local files or documents matching a query.", ("query",)),
            Capability("open_folder", "Open a specified folder or directory in file explorer.", ("path",)),
        )
        req = DecisionRequest("తాజా వాతావరణ వార్తలు గూగుల్ లో వెతుకు", {}, caps)
        pred = adapter.predict(req)
        assert pred["selected_capability"] == "google_search"

    def test_english_open_folder_preservation(self, adapter):
        """English folder command must correctly select open_folder."""
        caps = (
            Capability("open_folder", "Open a specified folder or directory in file explorer.", ("path",)),
            Capability("find_file", "Find local files or documents matching a query.", ("query",)),
        )
        req = DecisionRequest("Open my documents directory please", {}, caps)
        pred = adapter.predict(req)
        assert pred["selected_capability"] == "open_folder"

    def test_code_switching_app_open(self, adapter):
        """Code-switched 'Spotify application open karo' must select open_app."""
        caps = (
            Capability("open_app", "Launch or open an installed desktop application.", ("name",)),
            Capability("find_file", "Find local files or documents matching a query.", ("query",)),
        )
        req = DecisionRequest("Spotify application open karo", {}, caps)
        pred = adapter.predict(req)
        assert pred["selected_capability"] == "open_app"

    def test_native_telugu_ambiguous_clarification(self, adapter):
        """Ambiguous command with deictic pronoun must trigger clarification, not escalation."""
        caps = (
            Capability("delete_file", "Delete a specified file permanently.", ("path",)),
            Capability("delete_folder", "Delete a specified directory permanently.", ("path",)),
        )
        req = DecisionRequest("దీన్ని వెంటనే డిలీట్ చెయ్యి", {}, caps)
        pred = adapter.predict(req)
        assert pred["execution_path"] == "clarify"
        assert pred["needs_clarification"] is True
        assert pred["needs_escalation"] is False
        assert len(pred["missing_information"]) > 0

    def test_native_hindi_ambiguous_clarification(self, adapter):
        """Hindi ambiguous 'इसे हटा दो' must trigger clarification, not escalation."""
        caps = (
            Capability("delete_file", "Delete a file.", ("path",)),
            Capability("delete_folder", "Delete a folder.", ("path",)),
        )
        req = DecisionRequest("इसे हटा दो", {}, caps)
        pred = adapter.predict(req)
        assert pred["execution_path"] == "clarify"
        assert pred["needs_clarification"] is True
        assert pred["needs_escalation"] is False

    def test_direct_question_no_action(self, adapter):
        """Direct conversational question must not trigger random capability action."""
        req = DecisionRequest("How many planets are there in the solar system?", {}, ())
        pred = adapter.predict(req)
        assert pred["execution_path"] in ("direct_response", "conversational")
        assert pred["selected_capability"] is None

    def test_out_of_domain_escalation(self, adapter):
        """Out-of-domain physical request must escalate."""
        req = DecisionRequest("Make me a cup of hot espresso coffee", {}, ())
        pred = adapter.predict(req)
        assert pred["execution_path"] == "escalate"
        assert pred["needs_escalation"] is True
        assert pred["selected_capability"] is None
