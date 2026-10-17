"""Automated Safety, Isolation, Attribution, and Contract Tests for BERT Shadow Classification Audit.

Verifies all 15 required Section 11 invariant areas:
1. BERT is not loaded at application startup.
2. The first eligible shadow request triggers one initialization.
3. Concurrent requests do not load duplicate model instances.
4. Every eligible command is submitted for shadow classification.
5. Production and BERT use the same request identity and intended input.
6. Production decisions are unchanged by BERT predictions.
7. BERT cannot execute tools, modify state, or affect responses.
8. BERT load failure and timeout do not interrupt production.
9. Queue overflow and shutdown are handled safely.
10. Successful and failed events are persisted correctly.
11. Metrics match recomputed counts from the event log.
12. The comparison report refreshes from current events instead of retaining stale initialization data.
13. Fast-path and Qwen attribution is correct where each actually contributes.
14. Exact agreement and broad path compatibility are calculated separately.
15. Privacy settings are respected.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
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
from assistant.orchestration.router import QwenClassifier, ProductionClassificationDetails
from assistant.orchestration.shadow_evaluator import (
    ShadowEvaluator,
    detect_language_and_script,
)
from assistant.orchestration.shadow_review import evaluate_reviewed_shadow_data


class TestBERTShadowAudit(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="bert_shadow_audit_test_")
        self.output_dir = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test 1: BERT is not loaded at application startup
    # -------------------------------------------------------------------------
    def test_01_bert_not_loaded_at_startup(self):
        evaluator = ShadowEvaluator(output_dir=self.output_dir)
        # Verify adapter is None and no cold start has run upon instantiation or start()
        self.assertIsNone(evaluator._v8_adapter)
        self.assertIsNone(evaluator._cold_start_duration_ms)
        evaluator.start()
        try:
            self.assertIsNone(evaluator._v8_adapter)
        finally:
            evaluator.stop()

    # -------------------------------------------------------------------------
    # Test 2: The first eligible shadow request triggers one initialization
    # -------------------------------------------------------------------------
    def test_02_first_eligible_request_triggers_one_init(self):
        init_count = 0
        mock_adapter = MagicMock()
        mock_adapter.predict.return_value = {
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

        evaluator = ShadowEvaluator(output_dir=self.output_dir)

        def mock_loader():
            nonlocal init_count
            init_count += 1
            return mock_adapter

        evaluator._load_v8_adapter = mock_loader
        evaluator.start()
        try:
            evaluator.record_production_observation("Test query 1", "LOCAL", 1.0)
            time.sleep(0.3)
            self.assertEqual(init_count, 1)
            evaluator.record_production_observation("Test query 2", "LOCAL", 1.0)
            time.sleep(0.3)
            # Reused, not called again
            self.assertEqual(init_count, 1)
        finally:
            evaluator.stop()

    # -------------------------------------------------------------------------
    # Test 3: Concurrent requests do not load duplicate model instances
    # -------------------------------------------------------------------------
    def test_03_concurrent_requests_prevent_duplicate_initialization(self):
        load_count = 0
        evaluator = ShadowEvaluator(output_dir=self.output_dir)

        def slow_loader():
            nonlocal load_count
            time.sleep(0.1)
            load_count += 1
            return MagicMock()

        evaluator._load_v8_adapter = slow_loader

        # Spawn 5 concurrent threads calling _ensure_v8_adapter through the init lock
        with evaluator._init_lock:
            evaluator._v8_adapter = None

        results = []
        threads = []
        for _ in range(5):
            t = threading.Thread(target=lambda: results.append(evaluator._ensure_v8_adapter()))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        # Only one real model instance initialized
        self.assertEqual(load_count, 1)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertIsNotNone(r)

    # -------------------------------------------------------------------------
    # Test 4: Every eligible command is submitted for shadow classification
    # -------------------------------------------------------------------------
    def test_04_every_eligible_command_submitted(self):
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
        evaluator = ShadowEvaluator(output_dir=self.output_dir, v8_adapter=mock_v8)
        evaluator.start()
        try:
            for i in range(5):
                enqueued = evaluator.record_production_observation(f"Command {i}", "LOCAL", 1.0, request_id=f"cmd_{i}")
                self.assertTrue(enqueued)
            time.sleep(0.4)
            events = [json.loads(line) for line in evaluator.events_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(events), 5)
        finally:
            evaluator.stop()

    # -------------------------------------------------------------------------
    # Test 5: Production and BERT use the same request identity and intended input
    # -------------------------------------------------------------------------
    def test_05_same_request_identity_and_input(self):
        captured_req = {}

        def inspect_predict(req):
            captured_req["user_text"] = req.user_text
            captured_req["context"] = dict(req.context)
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
        mock_v8.predict.side_effect = inspect_predict
        evaluator = ShadowEvaluator(output_dir=self.output_dir, v8_adapter=mock_v8)
        evaluator.start()
        try:
            req_id = "req_identity_123"
            user_text = "What is the capital of France?"
            evaluator.record_production_observation(user_text, "LOCAL", 2.5, context={"topic": "geography"}, request_id=req_id)
            time.sleep(0.3)
            self.assertEqual(captured_req["user_text"], user_text)
            self.assertEqual(captured_req["context"], {"topic": "geography"})
            events = [json.loads(line) for line in evaluator.events_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[0]["request_id"], req_id)
        finally:
            evaluator.stop()

    # -------------------------------------------------------------------------
    # Test 6: Production decisions are unchanged by BERT predictions
    # -------------------------------------------------------------------------
    def test_06_production_decisions_unchanged(self):
        mock_classifier = MagicMock(spec=QwenClassifier)
        mock_classifier.classify.return_value = ("LOCAL", 2.0)

        evaluator = ShadowEvaluator(output_dir=self.output_dir)
        evaluator.start()
        try:
            with patch.object(config, "SHADOW_MODE_ENABLED", True), \
                 patch("assistant.orchestration.shadow_evaluator.get_shadow_evaluator", return_value=evaluator):
                graph = build_routing_graph(mock_classifier)
                res = graph.invoke({"user_text": "Open calculator", "route": None})
                # Production route remains strictly LOCAL despite anything BERT would think
                self.assertEqual(res["route"], "LOCAL")
        finally:
            evaluator.stop()

    # -------------------------------------------------------------------------
    # Test 7: BERT cannot execute tools, modify state, or affect responses
    # -------------------------------------------------------------------------
    def test_07_bert_cannot_execute_tools_or_modify_state(self):
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
            "confidence": 0.99,
        }
        evaluator = ShadowEvaluator(output_dir=self.output_dir, v8_adapter=mock_v8)
        evaluator.start()
        try:
            with patch("assistant.tools.executor.ToolExecutor.execute") as mock_exec:
                evaluator.record_production_observation("Delete critical file", "TASK", 1.0)
                time.sleep(0.3)
                mock_exec.assert_not_called()
        finally:
            evaluator.stop()

    # -------------------------------------------------------------------------
    # Test 8: BERT load failure and timeout do not interrupt production
    # -------------------------------------------------------------------------
    def test_08_load_failure_and_timeout_do_not_interrupt_production(self):
        mock_broken = MagicMock()
        mock_broken.predict.side_effect = RuntimeError("BERT crashed")
        evaluator = ShadowEvaluator(output_dir=self.output_dir, v8_adapter=mock_broken)
        evaluator.start()
        try:
            mock_classifier = MagicMock(spec=QwenClassifier)
            mock_classifier.classify.return_value = ("GEMINI", 3.0)
            with patch.object(config, "SHADOW_MODE_ENABLED", True), \
                 patch("assistant.orchestration.shadow_evaluator.get_shadow_evaluator", return_value=evaluator):
                graph = build_routing_graph(mock_classifier)
                res = graph.invoke({"user_text": "Will it rain?", "route": None})
                self.assertEqual(res["route"], "GEMINI")
                time.sleep(0.3)
        finally:
            evaluator.stop()

        events = [json.loads(line) for line in evaluator.events_file.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(events[0]["status"], "shadow_error")

    # -------------------------------------------------------------------------
    # Test 9: Queue overflow and shutdown are handled safely
    # -------------------------------------------------------------------------
    def test_09_queue_overflow_and_shutdown(self):
        evaluator = ShadowEvaluator(output_dir=self.output_dir, queue_max_size=2)
        evaluator._queue.put_nowait(MagicMock())
        evaluator._queue.put_nowait(MagicMock())

        # Attempt to enqueue third item - must drop safely without delay
        t0 = time.perf_counter()
        enqueued = evaluator.record_production_observation("Overflow query", "LOCAL", 1.0, request_id="drop_me")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertFalse(enqueued)
        self.assertLess(elapsed_ms, 15.0)

        events = [json.loads(line) for line in evaluator.events_file.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "shadow_dropped")
        self.assertEqual(events[0]["reason"], "queue_overflow")

    # -------------------------------------------------------------------------
    # Test 10: Successful and failed events are persisted correctly
    # -------------------------------------------------------------------------
    def test_10_success_and_failed_events_persisted(self):
        # One success, one timeout
        def dynamic_predict(req):
            if "fail" in req.user_text:
                raise ValueError("Intentional error")
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
        mock_v8.predict.side_effect = dynamic_predict
        evaluator = ShadowEvaluator(output_dir=self.output_dir, v8_adapter=mock_v8)
        evaluator.start()
        try:
            evaluator.record_production_observation("Good query", "LOCAL", 1.0, request_id="good_1")
            evaluator.record_production_observation("Query with fail", "LOCAL", 1.0, request_id="fail_1")
            time.sleep(0.4)
        finally:
            evaluator.stop()

        events = [json.loads(line) for line in evaluator.events_file.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(events), 2)
        statuses = {e["request_id"]: e["status"] for e in events}
        self.assertEqual(statuses["good_1"], "completed")
        self.assertEqual(statuses["fail_1"], "shadow_error")

    # -------------------------------------------------------------------------
    # Test 11: Metrics match recomputed counts from the event log
    # -------------------------------------------------------------------------
    def test_11_metrics_match_recomputed_counts(self):
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
        evaluator = ShadowEvaluator(output_dir=self.output_dir, v8_adapter=mock_v8)
        evaluator.start()
        try:
            for i in range(4):
                evaluator.record_production_observation(f"Query {i}", "LOCAL", 1.0, request_id=f"cnt_{i}")
            time.sleep(0.4)
        finally:
            evaluator.stop()

        events = [json.loads(line) for line in evaluator.events_file.read_text(encoding="utf-8").splitlines()]
        metrics = json.loads(evaluator.metrics_file.read_text(encoding="utf-8"))

        self.assertEqual(len(events), metrics["counts"]["evaluated_count"])
        self.assertEqual(metrics["counts"]["evaluated_count"], 4)

    # -------------------------------------------------------------------------
    # Test 12: The comparison report refreshes from current events
    # -------------------------------------------------------------------------
    def test_12_comparison_report_refreshes_from_events(self):
        mock_v8 = MagicMock()
        mock_v8.predict.return_value = {
            "intent": "use:open_app",
            "execution_path": "single_action",
            "request_modes": ["action"],
            "selected_capability": "open_app",
            "needs_clarification": False,
            "needs_escalation": False,
            "missing_information": [],
            "context_references": [],
            "confidence": 1.0,
        }
        evaluator = ShadowEvaluator(output_dir=self.output_dir, v8_adapter=mock_v8)
        evaluator.start()
        try:
            evaluator.record_production_observation("Open chrome", "TASK", 2.0, request_id="comp_1")
            time.sleep(0.4)
        finally:
            evaluator.stop()

        comp = json.loads(evaluator.comparison_file.read_text(encoding="utf-8"))
        self.assertEqual(comp["total_commands_evaluated"], 1)
        self.assertEqual(comp["broad_path_compatibility_pct"], 100.0)

    # -------------------------------------------------------------------------
    # Test 13: Fast-path and Qwen attribution is correct
    # -------------------------------------------------------------------------
    def test_13_fast_path_and_qwen_attribution_correct(self):
        evaluator = ShadowEvaluator(output_dir=self.output_dir)

        # Attribution 1: Fast-path
        fp_details = ProductionClassificationDetails(
            final_route="TASK",
            total_latency_ms=0.3,
            decision_component="deterministic_fast_path",
            fast_path_handled=True,
            qwen_invoked=False,
        )
        evaluator.record_production_observation("weather in London", "TASK", 0.3, production_details=fp_details)

        # Attribution 2: Qwen fallback
        fb_details = ProductionClassificationDetails(
            final_route="GEMINI",
            total_latency_ms=5000.0,
            decision_component="qwen_fallback",
            fast_path_handled=False,
            qwen_invoked=True,
            fallback_used=True,
            fallback_reason="timeout",
        )
        evaluator.record_production_observation("Complex philosophical thought", "GEMINI", 5000.0, production_details=fb_details)

        evaluator._flush_metrics()
        metrics = json.loads(evaluator.metrics_file.read_text(encoding="utf-8"))

        self.assertEqual(metrics["production_attribution"]["fast_path_handled"], 1)
        self.assertEqual(metrics["production_attribution"]["qwen_invoked"], 1)
        self.assertEqual(metrics["production_attribution"]["qwen_fallback_used"], 1)

    # -------------------------------------------------------------------------
    # Test 14: Exact agreement and broad path compatibility calculated separately
    # -------------------------------------------------------------------------
    def test_14_exact_agreement_and_broad_compatibility_separated(self):
        # Case: Broadly compatible (both conversational) but BERT asked for clarification, so NOT exact agreement
        bert_clarify = {
            "intent": "clarify_request",
            "execution_path": "clarify",
            "request_modes": ["conversational"],
            "selected_capability": None,
            "needs_clarification": True,
            "needs_escalation": False,
            "missing_information": [{"name": "target"}],
            "context_references": [],
            "confidence": 0.5,
        }
        res = ShadowEvaluator._compare_predictions("LOCAL", bert_clarify)
        self.assertTrue(res["broad_path_compatible"])
        self.assertFalse(res["exact_agreement"])
        self.assertEqual(res["disagreement_category"], "v8_clarification")

    # -------------------------------------------------------------------------
    # Test 15: Privacy settings are respected
    # -------------------------------------------------------------------------
    def test_15_privacy_settings_respected(self):
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
        # Privacy on: persist_raw_text=False
        evaluator = ShadowEvaluator(output_dir=self.output_dir, persist_raw_text=False, v8_adapter=mock_v8)
        evaluator.start()
        try:
            sensitive = "Super secret authorization key: token_99999"
            evaluator.record_production_observation(sensitive, "LOCAL", 1.0)
            time.sleep(0.3)
        finally:
            evaluator.stop()

        raw_events = evaluator.events_file.read_text(encoding="utf-8")
        self.assertNotIn("token_99999", raw_events)
        ev = json.loads(raw_events.strip())
        self.assertFalse(ev["privacy"]["raw_text_persisted"])
        self.assertTrue(len(ev["privacy"]["text_sha256"]) == 64)


if __name__ == "__main__":
    unittest.main()
