from __future__ import annotations

from benchmarks.jarvis_fast_brain.replay.fixtures import FixtureExecutor
from benchmarks.jarvis_fast_brain.replay.schema import ReplayCase, model_input


def case():
    return ReplayCase.from_mapping({"id":"x","language":"en","user_input":"Open it",
        "expected":{"selected_capability":"open_file"},"available_capabilities":[{"name":"open_file"}]})


def test_expected_labels_never_enter_model_input():
    payload = model_input(case(), "system")
    assert "expected" not in payload
    assert "selected_capability" not in str(payload)


def test_fixture_is_deterministic_and_does_not_call_tools():
    fx = FixtureExecutor({"tool_results":{"google_search":{"success":True,"data":["fixture"]}}})
    assert fx.execute("google_search", {"query":"x"})["success"] is True
    assert fx.execute("google_search")["success"] is True
    assert len(fx.calls) == 2


def test_case_defaults_are_bounded():
    item = case()
    assert item.context == {}
    assert item.observations == []
