"""Contract tests for FastClassifier; these do not measure model accuracy."""
from __future__ import annotations

from unittest.mock import MagicMock

from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    FastClassifier,
    LegacyClassifierAdapter,
    RequestMode,
)
from assistant.tools.executor import ToolExecutor
from assistant.tools.registry import ToolRegistry


class MockDecisionAdapter:
    adapter_id = "mock-structured-model"

    def __init__(self, response):
        self.response = response
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        return self.response


def decision_output(**overrides):
    value = {
        "intent": "answer_user_request",
        "execution_path": "LOCAL",
        "request_modes": ["conversational"],
        "selected_capability": None,
        "context_references": [],
        "missing_information": [],
        "confidence": 0.91,
        "needs_clarification": False,
        "needs_escalation": False,
        "evidence": [{"kind": "input_span", "reference": "request"}],
    }
    value.update(overrides)
    return value


def test_unseen_paraphrase_is_sent_to_model_without_keyword_gate():
    adapter = MockDecisionAdapter(decision_output(intent="summarize_technical_material"))
    classifier = FastClassifier(adapter)
    text = "Give me the bones of this design in a form I can scan before lunch"

    decision = classifier.decide(text)

    assert decision.intent == "summarize_technical_material"
    assert adapter.requests[0].user_text == text


def test_language_identity_does_not_force_an_execution_path():
    adapter = MockDecisionAdapter(decision_output(execution_path="LOCAL"))
    classifier = FastClassifier(adapter)

    route, _ = classifier.classify("నమస్కారం")

    assert route == "LOCAL"
    assert adapter.requests[0].user_text == "నమస్కారం"


def test_context_references_are_generic_and_can_remain_unresolved():
    adapter = MockDecisionAdapter(
        decision_output(
            context_references=[
                {
                    "reference": "the workspace we discussed yesterday",
                    "kind": "workspace",
                    "resolved_value": None,
                }
            ],
            missing_information=[
                {"name": "workspace_identity", "description": "Which workspace is intended"}
            ],
            needs_clarification=True,
            confidence=0.42,
        )
    )

    decision = FastClassifier(adapter).decide("Check the workspace we discussed yesterday")

    assert decision.context_references[0].reference == "the workspace we discussed yesterday"
    assert decision.context_references[0].is_resolved is False
    assert decision.needs_clarification is True


def test_invalid_model_output_fails_closed_to_escalation():
    classifier = FastClassifier(MockDecisionAdapter({"intent": "incomplete"}))

    decision = classifier.decide("Do something")
    route, _ = classifier.classify("Do something")

    assert decision.intent == "classification_failed"
    assert decision.needs_escalation is True
    assert decision.confidence == 0.0
    assert route == "GEMINI"


def test_uncertainty_can_request_clarification_or_escalation():
    clarification = ClassifierDecision.from_mapping(
        decision_output(
            execution_path="clarify",
            missing_information=[{"name": "target"}],
            needs_clarification=True,
            confidence=0.3,
        )
    )
    escalation = ClassifierDecision.from_mapping(
        decision_output(
            execution_path="escalate",
            needs_escalation=True,
            confidence=0.1,
        )
    )

    assert clarification.needs_clarification is True
    assert escalation.needs_escalation is True


def test_capability_selection_cannot_bypass_tool_allow_list():
    adapter = MockDecisionAdapter(
        decision_output(
            execution_path="TASK",
            request_modes=["action"],
            selected_capability="unregistered_destructive_tool",
        )
    )
    classifier = FastClassifier(adapter)
    executor = ToolExecutor(ToolRegistry(), timeout=1)

    decision = classifier.decide(
        "Modify the system",
        capabilities=[Capability("read_file", "Read an allowed project file", ("path",))],
    )
    result = executor.execute(decision.selected_capability, {})

    assert decision.selected_capability == "unregistered_destructive_tool"
    assert result["success"] is False
    assert result["error"] == "unknown_tool"


def test_existing_classify_callers_remain_compatible():
    classifier = FastClassifier(MockDecisionAdapter(decision_output(execution_path="TASK")))

    route, latency_ms = classifier.classify("An arbitrary unseen request")

    assert route == "TASK"
    assert latency_ms >= 0.0


def test_existing_qwen_style_classifier_can_use_adapter_boundary():
    legacy = MagicMock()
    legacy.classify.return_value = ("LOCAL", 2.5)
    classifier = FastClassifier(LegacyClassifierAdapter(legacy))

    decision = classifier.decide("A fresh formulation")

    assert decision.execution_path == "LOCAL"
    assert RequestMode.CONVERSATIONAL in decision.request_modes
    legacy.classify.assert_called_once_with("A fresh formulation")
