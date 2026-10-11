"""Regression test suite for Canonical Argument Contract and Slot Contract Resolver.

Covers:
1. Valid and invalid argument names
2. Multiple missing arguments
3. Optional arguments vs required arguments
4. Unknown capabilities rejection
5. Clarification consistency (execution_path <-> needs_clarification <-> missing_information)
6. Context references preservation
7. Multilingual input stability
8. Multi-step request handling
9. Out-of-domain request escalation preservation
"""
import pytest
import sys
from pathlib import Path

ROOT = Path("D:/JARVIS")
EXP_DIR = Path("D:/JARVIS/reports/jarvis_slot_contract_fix_20261010_211600")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))

from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    DecisionRequest,
    DecisionValidationError,
    MissingInformation,
    ContextReference,
)
from slot_contract_resolver import (
    CanonicalArgumentContract,
    SlotContractResolver,
    CANONICAL_CAPABILITY_SCHEMAS,
)


class TestCanonicalArgumentContract:
    def test_canonical_schemas_defined(self):
        """Ensure all standard JARVIS capabilities have schemas."""
        contract = CanonicalArgumentContract()
        for cap in ("open_app", "close_app", "open_folder", "find_file", "read_page", "delete_file"):
            schema = contract.get_schema(cap)
            assert schema is not None, f"Schema missing for {cap}"
            assert len(schema.required_arguments) >= 1

    def test_distinguishes_required_from_optional(self):
        """Ensure required vs optional arguments are properly distinguished."""
        contract = CanonicalArgumentContract()
        schema = contract.get_schema("read_file")
        assert "path" in schema.required_arguments
        assert "offset" in schema.optional_arguments
        assert "limit" in schema.optional_arguments

    def test_validate_valid_argument_names(self):
        """Valid argument names must pass validation."""
        contract = CanonicalArgumentContract()
        assert contract.validate_argument("open_app", "app") is True
        assert contract.validate_argument("open_app", "name") is True  # standard alias
        assert contract.validate_argument("open_folder", "path") is True
        assert contract.validate_argument("read_page", "url") is True

    def test_validate_invalid_argument_names(self):
        """Invalid argument names must be rejected."""
        contract = CanonicalArgumentContract()
        assert contract.validate_argument("open_app", "non_existent_arg") is False
        assert contract.validate_argument("open_folder", "random_junk") is False

    def test_unknown_capability_rejection(self):
        """Unknown capabilities must return None and reject arguments."""
        contract = CanonicalArgumentContract()
        assert contract.get_schema("hack_mainframe") is None
        assert contract.validate_argument("hack_mainframe", "target") is False


class TestSlotContractResolver:
    def setup_method(self):
        self.resolver = SlotContractResolver()

    def test_clarification_consistency_with_missing_info(self):
        """needs_clarification requires at least one missing_information item."""
        raw = {
            "intent": "clarify_request",
            "execution_path": "clarify",
            "request_modes": ["action"],
            "selected_capability": None,
            "context_references": [],
            "missing_information": [],
            "confidence": 0.8,
            "needs_clarification": True,
            "needs_escalation": False,
        }
        # Resolver should ensure missing_information is populated when needs_clarification is True
        resolved = self.resolver.resolve_decision(raw, DecisionRequest("open that"))
        assert len(resolved["missing_information"]) > 0
        assert resolved["needs_clarification"] is True
        assert resolved["execution_path"] == "clarify"
        ClassifierDecision.from_mapping(resolved)

    def test_clarification_consistency_false_when_empty(self):
        """When needs_clarification is False, missing_information must be empty."""
        raw = {
            "intent": "use:open_app",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "open_app",
            "context_references": [],
            "missing_information": [{"name": "target", "description": "unneeded"}],
            "confidence": 0.9,
            "needs_clarification": False,
            "needs_escalation": False,
        }
        resolved = self.resolver.resolve_decision(raw, DecisionRequest("open chrome"))
        assert resolved["missing_information"] == []
        assert resolved["needs_clarification"] is False
        ClassifierDecision.from_mapping(resolved)

    def test_multiple_missing_arguments_representation(self):
        """Resolver handles multiple missing arguments correctly."""
        missing = self.resolver.create_missing_info([
            ("path", "The file path is required"),
            ("destination", "The destination path is required"),
        ])
        assert len(missing) == 2
        assert missing[0]["name"] == "path"
        assert missing[1]["name"] == "destination"

    def test_machine_readable_vs_human_readable_separation(self):
        """Machine identifier must be cleanly formatted and distinct from description."""
        item = self.resolver.format_missing_item("folder_path", "Please specify which folder to open")
        assert item["name"] == "folder_path"
        assert item["description"] == "Please specify which folder to open"

    def test_context_references_preserved(self):
        """Context references must never be dropped or mutated during contract resolution."""
        ctx_ref = [{"kind": "item", "reference": "that", "resolved_value": None}]
        raw = {
            "intent": "clarify_request",
            "execution_path": "clarify",
            "request_modes": ["action"],
            "selected_capability": None,
            "context_references": ctx_ref,
            "missing_information": [{"name": "target", "description": "Target is unidentifiable"}],
            "confidence": 0.7,
            "needs_clarification": True,
            "needs_escalation": False,
        }
        resolved = self.resolver.resolve_decision(raw, DecisionRequest("delete that"))
        assert resolved["context_references"] == ctx_ref

    def test_escalation_preserved(self):
        """Escalation path and flags must be strictly preserved without clarification interference."""
        raw = {
            "intent": "unsupported_request",
            "execution_path": "escalate",
            "request_modes": ["conversational"],
            "selected_capability": None,
            "context_references": [],
            "missing_information": [],
            "confidence": 0.85,
            "needs_clarification": False,
            "needs_escalation": True,
        }
        resolved = self.resolver.resolve_decision(raw, DecisionRequest("make me coffee"))
        assert resolved["execution_path"] == "escalate"
        assert resolved["needs_escalation"] is True
        assert resolved["needs_clarification"] is False
        assert resolved["missing_information"] == []

    def test_multilingual_input_stability(self):
        """Non-Latin and transliterated inputs pass through resolution without encoding degradation."""
        for text in ("దీన్ని వెంటనే డిలీట్ చెయ్యి", "इसे तुरंत हटाओ", "athai marubadiyum open pannu"):
            req = DecisionRequest(text)
            raw = {
                "intent": "clarify_request",
                "execution_path": "clarify",
                "request_modes": ["action"],
                "selected_capability": None,
                "context_references": [],
                "missing_information": [{"name": "target", "description": "The requested target is not identifiable"}],
                "confidence": 0.75,
                "needs_clarification": True,
                "needs_escalation": False,
            }
            resolved = self.resolver.resolve_decision(raw, req)
            decision = ClassifierDecision.from_mapping(resolved)
            assert decision.execution_path == "clarify"

    def test_multi_step_request_contract(self):
        """Multi-step tasks maintain multi_step_task execution path and appropriate modes."""
        raw = {
            "intent": "manage:multi_step_task",
            "execution_path": "multi_step_task",
            "request_modes": ["action", "research"],
            "selected_capability": "find_file",
            "context_references": [],
            "missing_information": [],
            "confidence": 0.8,
            "needs_clarification": False,
            "needs_escalation": False,
        }
        resolved = self.resolver.resolve_decision(raw, DecisionRequest("find my notes and read them"))
        assert resolved["execution_path"] == "multi_step_task"
        assert "action" in resolved["request_modes"]
        assert resolved["needs_clarification"] is False

    def test_safe_rejection_of_unsupported_capability(self):
        """If model selects a capability not advertised in request.capabilities, resolver corrects or flags it."""
        valid_cap = Capability("open_app", "Open app", ("name",))
        req = DecisionRequest("run malware", capabilities=(valid_cap,))
        raw = {
            "intent": "use:destroy_world",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "destroy_world",  # NOT in request.capabilities!
            "context_references": [],
            "missing_information": [],
            "confidence": 0.9,
            "needs_clarification": False,
            "needs_escalation": False,
        }
        resolved = self.resolver.resolve_decision(raw, req)
        # Should reject unknown capability and escalate or clarify safely
        assert resolved["selected_capability"] is None
        assert resolved["execution_path"] in ("escalate", "clarify")

