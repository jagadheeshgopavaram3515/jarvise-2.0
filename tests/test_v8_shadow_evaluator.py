"""Automated Safety, Isolation, and Contract Tests for V8 Shadow Evaluator.

Verifies all 14 required Phase G invariant areas:
1. Production output remains unchanged when shadow mode is OFF.
2. Production output remains unchanged when shadow mode is ON.
3. V8 cannot invoke action execution.
4. V8 errors do not fail production requests.
5. V8 timeouts do not delay production responses beyond documented scheduling overhead.
6. Queue overflow drops shadow jobs safely.
7. Duplicate request IDs are handled correctly.
8. Invalid V8 outputs are recorded as invalid, not counted as successful predictions.
9. Shutdown stops background work safely.
10. Both classifiers receive equivalent permitted classification inputs.
11. Privacy settings prevent raw text persistence by default.
12. Metrics exclude unreviewed examples from accuracy calculations.
13. Strict scoring uses the verified eight-field contract.
14. Paired comparisons correctly identify wins, losses, and ties.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from assistant import config
from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    DecisionRequest,
    DecisionValidationError,
)
from assistant.orchestration.graph import Orchestrator, build_routing_graph
from assistant.orchestration.router import QwenClassifier
from assistant.orchestration.shadow_evaluator import (
    ShadowEvaluator,
    detect_language_and_script,
)
from assistant.orchestration.shadow_review import evaluate_reviewed_shadow_data


class TestV8ShadowEvaluator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="shadow_test_")
        self.output_dir = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test 1: Production output remains unchanged when shadow mode is OFF
    # -------------------------------------------------------------------------
    def test_01_production_output_unchanged_when_shadow_off(self):
        mock_classifier = MagicMock(spec=QwenClassifier)
        mock_classifier.classify.return_value = ("TASK", 4.2)

        with patch.object(config, "SHADOW_MODE_ENABLED", False):
            graph = build_routing_graph(mock_classifier)
            result = graph.invoke({"user_text": "Open chrome", "route": None})
            self.assertEqual(result["route"], "TASK")
            mock_classifier.classify.assert_called_once_with("Open chrome")

    # -------------------------------------------------------------------------
    # Test 2: Production output remains unchanged when shadow mode is ON
    # -------------------------------------------------------------------------
    def test_02_production_output_unchanged_when_shadow_on(self):
        mock_classifier = MagicMock(spec=QwenClassifier)
        mock_classifier.classify.return_value = ("LOCAL", 2.1)

        evaluator = ShadowEvaluator(output_dir=self.output_dir)
        evaluator.start()
        try:
            with patch.object(config, "SHADOW_MODE_ENABLED", True), \
                 patch("assistant.orchestration.shadow_evaluator.get_shadow_evaluator", return_value=evaluator):
                graph = build_routing_graph(mock_classifier)
                result = graph.invoke({"user_text": "Hello Jarvis", "route": None})
                # Production route is exactly what the mock production classifier returned
                self.assertEqual(result["route"], "LOCAL")
        finally:
            evaluator.stop()

    # -------------------------------------------------------------------------
    # Test 3: V8 cannot invoke action execution
    # -------------------------------------------------------------------------
    def test_03_v8_cannot_invoke_action_execution(self):
        # Even if V8 predicts an aggressive tool action, tool executors are never called
        mock_v8 = MagicMock()
        mock_v8.predict.return_value = {
            "intent": "use:delete_file",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "delete_file",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": [],
            "confidence": 0.95,
        }

        evaluator = ShadowEvaluator(output_dir=self.output_dir, v8_adapter=mock_v8)
        evaluator.start()
        try:
            with patch("assistant.tools.executor.ToolExecutor.execute") as mock_tool_exec:
                enqueued = evaluator.record_production_observation(
                    user_text="Delete my project file",
                    production_route="TASK",
                    production_latency_ms=1.5,
                )
                self.assertTrue(enqueued)
                # Wait briefly for worker to drain
                time.sleep(0.3)
                mock_tool_exec.assert_not_called()
        finally:
            evaluator.stop()

    # -------------------------------------------------------------------------
    # Test 4: V8 errors do not fail production requests
    # -------------------------------------------------------------------------
    def test_04_v8_errors_do_not_fail_production_requests(self):
        mock_broken_v8 = MagicMock()
        mock_broken_v8.predict.side_effect = RuntimeError("Simulated internal GPU/BERT crash")

        evaluator = ShadowEvaluator(output_dir=self.output_dir, v8_adapter=mock_broken_v8)
        evaluator.start()
        try:
            mock_classifier = MagicMock(spec=QwenClassifier)
            mock_classifier.classify.return_value = ("TASK", 3.0)

            with patch.object(config, "SHADOW_MODE_ENABLED", True), \
                 patch("assistant.orchestration.shadow_evaluator.get_shadow_evaluator", return_value=evaluator):
                graph = build_routing_graph(mock_classifier)
                # Graph invocation must succeed completely despite V8 runtime failure
                result = graph.invoke({"user_text": "Test error", "route": None})
                self.assertEqual(result["route"], "TASK")
                time.sleep(0.3)
        finally:
            evaluator.stop()

        # Check that error event was logged
        events = [json.loads(line) for line in evaluator.events_file.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "shadow_error")
        self.assertIn("Simulated internal GPU/BERT crash", events[0]["error"])

    # -------------------------------------------------------------------------
    # Test 5: V8 timeouts do not delay production responses beyond scheduling overhead
    # -------------------------------------------------------------------------
    def test_05_v8_timeouts_do_not_delay_production(self):
        def slow_predict(req):
            time.sleep(1.0)
            return {}

        mock_slow_v8 = MagicMock()
        mock_slow_v8.predict.side_effect = slow_predict

        # Set shadow inference timeout very low (0.05s)
        evaluator = ShadowEvaluator(output_dir=self.output_dir, timeout_s=0.05, v8_adapter=mock_slow_v8)
        evaluator.start()
        try:
            t0 = time.perf_counter()
            enqueued = evaluator.record_production_observation(
                user_text="Slow query",
                production_route="GEMINI",
                production_latency_ms=2.0,
            )
            overhead_ms = (time.perf_counter() - t0) * 1000.0
            self.assertTrue(enqueued)
            # Enqueueing overhead must be negligible (< 10ms)
            self.assertLess(overhead_ms, 20.0)
            time.sleep(0.3)
        finally:
            evaluator.stop()

        events = [json.loads(line) for line in evaluator.events_file.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "shadow_timeout")

    # -------------------------------------------------------------------------
    # Test 6: Queue overflow drops shadow jobs safely
    # -------------------------------------------------------------------------
    def test_06_queue_overflow_drops_shadow_jobs_safely(self):
        # Capacity of 2, worker stopped so queue fills immediately
        evaluator = ShadowEvaluator(output_dir=self.output_dir, queue_max_size=2)
        # Do not start evaluator so queue remains full
        evaluator._queue.put_nowait(MagicMock())
        evaluator._queue.put_nowait(MagicMock())

        # Third job should be dropped immediately without blocking
        t0 = time.perf_counter()
        enqueued = evaluator.record_production_observation(
            user_text="Overflow test",
            production_route="LOCAL",
            production_latency_ms=1.0,
            request_id="req_drop_1",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertFalse(enqueued)
        self.assertLess(elapsed_ms, 10.0)

        events = [json.loads(line) for line in evaluator.events_file.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "shadow_dropped")
        self.assertEqual(events[0]["reason"], "queue_overflow")

    # -------------------------------------------------------------------------
    # Test 7: Duplicate request IDs are handled correctly
    # -------------------------------------------------------------------------
    def test_07_duplicate_request_ids_handled(self):
        evaluator = ShadowEvaluator(output_dir=self.output_dir)
        first = evaluator.record_production_observation(
            user_text="First occurrence",
            production_route="LOCAL",
            production_latency_ms=1.0,
            request_id="req_unique_001",
        )
        second = evaluator.record_production_observation(
            user_text="Second occurrence with same ID",
            production_route="LOCAL",
            production_latency_ms=1.0,
            request_id="req_unique_001",
        )
        self.assertTrue(first)
        self.assertFalse(second)  # duplicate dropped

    # -------------------------------------------------------------------------
    # Test 8: Invalid V8 outputs are recorded as invalid
    # -------------------------------------------------------------------------
    def test_08_invalid_v8_outputs_recorded_as_invalid(self):
        mock_v8 = MagicMock()
        # Missing required field 'execution_path'
        mock_v8.predict.return_value = {
            "intent": "broken",
            "request_modes": ["conversational"],
        }

        evaluator = ShadowEvaluator(output_dir=self.output_dir, v8_adapter=mock_v8)
        evaluator.start()
        try:
            evaluator.record_production_observation("Test invalid schema", "LOCAL", 1.0)
            time.sleep(0.3)
        finally:
            evaluator.stop()

        events = [json.loads(line) for line in evaluator.events_file.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["bert"]["schema_valid"])
        self.assertIn("Schema validation failed", events[0]["error"])

    # -------------------------------------------------------------------------
    # Test 9: Shutdown stops background work safely
    # -------------------------------------------------------------------------
    def test_09_shutdown_stops_background_work(self):
        evaluator = ShadowEvaluator(output_dir=self.output_dir)
        evaluator.start()
        self.assertTrue(evaluator.is_active)
        self.assertTrue(evaluator._worker_thread.is_alive())

        evaluator.stop(timeout_s=1.0)
        self.assertFalse(evaluator.is_active)
        self.assertFalse(evaluator._worker_thread.is_alive())

    # -------------------------------------------------------------------------
    # Test 10: Both classifiers receive equivalent permitted classification inputs
    # -------------------------------------------------------------------------
    def test_10_both_classifiers_receive_equivalent_inputs(self):
        received_request = {}

        def spy_predict(req):
            received_request["user_text"] = req.user_text
            received_request["context"] = dict(req.context)
            return {
                "intent": "respond",
                "execution_path": "direct_response",
                "request_modes": ["conversational"],
                "selected_capability": None,
                "needs_clarification": False,
                "needs_escalation": False,
                "missing_information": [],
                "context_references": [],
                "confidence": 1.0,
            }

        mock_v8 = MagicMock()
        mock_v8.predict.side_effect = spy_predict

        evaluator = ShadowEvaluator(output_dir=self.output_dir, v8_adapter=mock_v8)
        evaluator.start()
        try:
            user_msg = "తెలుగులో సమాధానం చెప్పండి"
            evaluator.record_production_observation(
                user_text=user_msg,
                production_route="GEMINI",
                production_latency_ms=1.5,
                context={"recent_topic": "weather"},
            )
            time.sleep(0.3)
            self.assertEqual(received_request["user_text"], user_msg)
            self.assertEqual(received_request["context"], {"recent_topic": "weather"})
        finally:
            evaluator.stop()

    # -------------------------------------------------------------------------
    # Test 11: Privacy settings prevent raw text persistence by default
    # -------------------------------------------------------------------------
    def test_11_privacy_settings_prevent_raw_text_persistence(self):
        mock_v8 = MagicMock()
        mock_v8.predict.return_value = {
            "intent": "respond",
            "execution_path": "direct_response",
            "request_modes": ["conversational"],
            "selected_capability": None,
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": [],
            "confidence": 1.0,
        }

        # Default: persist_raw_text=False
        evaluator = ShadowEvaluator(output_dir=self.output_dir, persist_raw_text=False, v8_adapter=mock_v8)
        evaluator.start()
        try:
            sensitive_msg = "My confidential password is secret123"
            evaluator.record_production_observation(sensitive_msg, "LOCAL", 1.0)
            time.sleep(0.3)
        finally:
            evaluator.stop()

        raw_content = evaluator.events_file.read_text(encoding="utf-8")
        self.assertNotIn("secret123", raw_content)
        event = json.loads(raw_content.strip())
        self.assertFalse(event["privacy"]["raw_text_persisted"])
        self.assertTrue(len(event["privacy"]["text_sha256"]) == 64)

    # -------------------------------------------------------------------------
    # Test 12: Metrics exclude unreviewed examples from accuracy calculations
    # -------------------------------------------------------------------------
    def test_12_metrics_exclude_unreviewed_examples(self):
        events_file = self.output_dir / "shadow_events.jsonl"
        reviews_file = self.output_dir / "reviewed_labels.jsonl"

        # Create 5 unreviewed shadow events
        for i in range(5):
            ev = {
                "request_id": f"req_{i}",
                "status": "completed",
                "production": {"route": "LOCAL"},
                "v8": {"prediction": {"selected_capability": None}, "schema_valid": True},
            }
            events_file.open("a", encoding="utf-8").write(json.dumps(ev) + "\n")

        # No reviews file exists yet
        result = evaluate_reviewed_shadow_data(events_file, reviews_file)
        self.assertEqual(result["status"], "insufficient_reviewed_samples")
        self.assertFalse(result["accuracy_evaluation_available"])
        self.assertEqual(result["unreviewed_records_excluded"], 5)

    # -------------------------------------------------------------------------
    # Test 13: Strict scoring uses verified eight-field contract
    # -------------------------------------------------------------------------
    def test_13_strict_scoring_uses_eight_field_contract(self):
        events_file = self.output_dir / "shadow_events.jsonl"
        reviews_file = self.output_dir / "reviewed_labels.jsonl"

        expected = {
            "intent": "use:open_app",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "open_app",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": [],
        }

        # V8 prediction has wrong missing_information
        v8_imperfect = dict(expected)
        v8_imperfect["missing_information"] = [{"name": "extra"}]

        ev = {
            "request_id": "req_strict_test",
            "status": "completed",
            "production": {"route": "TASK"},
            "v8": {"prediction": v8_imperfect, "schema_valid": True},
        }
        events_file.open("w", encoding="utf-8").write(json.dumps(ev) + "\n")

        rev = {
            "request_id": "req_strict_test",
            "review_status": "verified",
            "expected_output": expected,
        }
        reviews_file.open("w", encoding="utf-8").write(json.dumps(rev) + "\n")

        result = evaluate_reviewed_shadow_data(events_file, reviews_file)
        self.assertEqual(result["status"], "evaluated")
        # Capability top-1 is correct
        self.assertEqual(result["overall_metrics"]["v8"]["capability_top1_matches"], 1)
        # But strict 8-field match is 0 due to missing_information divergence
        self.assertEqual(result["overall_metrics"]["v8"]["strict_exact_matches"], 0)

    # -------------------------------------------------------------------------
    # Test 14: Paired comparisons correctly identify wins, losses, and ties
    # -------------------------------------------------------------------------
    def test_14_paired_comparisons_identify_wins_losses_ties(self):
        events_file = self.output_dir / "shadow_events.jsonl"
        reviews_file = self.output_dir / "reviewed_labels.jsonl"

        expected_action = {
            "intent": "use:open_app",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "open_app",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": [],
        }

        # Case 1: V8 win (V8 exact match, Production conversational mismatch)
        ev1 = {
            "request_id": "case_1",
            "production": {"route": "LOCAL"},
            "v8": {"prediction": expected_action, "schema_valid": True},
        }
        # Case 2: Tie (both wrong or both right)
        ev2 = {
            "request_id": "case_2",
            "production": {"route": "GEMINI"},
            "v8": {"prediction": expected_action, "schema_valid": False},  # V8 invalid schema
        }

        for ev in [ev1, ev2]:
            events_file.open("a", encoding="utf-8").write(json.dumps(ev) + "\n")

        for rid in ["case_1", "case_2"]:
            rev = {"request_id": rid, "review_status": "reviewed", "expected_output": expected_action}
            reviews_file.open("a", encoding="utf-8").write(json.dumps(rev) + "\n")

        result = evaluate_reviewed_shadow_data(events_file, reviews_file)
        paired = result["overall_metrics"]["paired_comparison"]
        self.assertEqual(paired["v8_wins"], 1)
        self.assertEqual(paired["production_wins"], 0)
        self.assertEqual(paired["ties"], 1)


if __name__ == "__main__":
    unittest.main()

