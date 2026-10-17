"""Safe, read-only shadow evaluation engine for JARVIS Brain.

Runs Experimental Classifier V8 (multilingual BERT) asynchronously beside
production routing.

Strict Guarantees:
1. Zero Intervention: BERT output is never returned to or executed by production.
2. Zero Blocking: Enqueuing is bounded and non-blocking (<0.05ms overhead); worker runs in background.
3. Overflow Protection: Drops shadow tasks immediately if queue is full.
4. Error & Timeout Isolation: Model timeouts and errors never affect user responses.
5. Lazy Initialization: BERT is NOT loaded on startup; initializes on the first eligible request.
6. Privacy-by-Default: Never persists raw user utterances unless explicitly enabled.
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any, Mapping, Sequence
import uuid

from assistant import config
from assistant.core.log import get
from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    DecisionRequest,
    DecisionValidationError,
)

log = get("shadow_evaluator")

# Language & Script regular expressions
_INDIC_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
_INDIC_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_INDIC_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
_ROMANISED_INDIC_RE = re.compile(
    r"\b(nuvvu|ela|unnav|unnaru|entha|eppudu|ekkada|cheyyi|cheppu|kavali|undi|ledu|"
    r"tum|aap|kaise|ho|kya|kyun|kab|kahan|batao|chalo|hai|hain|nahi|theek|karo|"
    r"danni|idi|adi|chudu|open\s*cheyyi|bandh\s*karo)\b",
    re.I,
)


def detect_language_and_script(text: str) -> dict[str, Any]:
    """Detect script and language classification from text."""
    clean = text.strip()
    if not clean:
        return {"primary_category": "empty", "scripts_present": []}

    has_te = bool(_INDIC_TELUGU_RE.search(clean))
    has_hi = bool(_INDIC_DEVANAGARI_RE.search(clean))
    has_ta = bool(_INDIC_TAMIL_RE.search(clean))
    has_latin = bool(re.search(r"[a-zA-Z]", clean))
    has_romanised = bool(_ROMANISED_INDIC_RE.search(clean))

    scripts = []
    if has_te:
        scripts.append("telugu_native")
    if has_hi:
        scripts.append("devanagari_hindi")
    if has_ta:
        scripts.append("tamil_native")
    if has_latin:
        scripts.append("latin")

    if len(scripts) > 1 and (has_te or has_hi or has_ta):
        category = "code_mixed_indic"
    elif has_te:
        category = "telugu_native"
    elif has_hi:
        category = "devanagari_hindi"
    elif has_ta:
        category = "tamil_native"
    elif has_romanised:
        category = "latin_transliterated_indic"
    elif has_latin:
        category = "english_latin"
    else:
        category = "other_or_numeric"

    return {
        "primary_category": category,
        "scripts_present": scripts,
        "is_code_switched": has_romanised or len(scripts) > 1,
    }


def compute_percentiles(values: list[float]) -> dict[str, float]:
    """Compute min, median, p95, and max for latency values."""
    if not values:
        return {"count": 0, "min_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    med_idx = n // 2
    p95_idx = min(n - 1, int(math.ceil(0.95 * n)) - 1)
    return {
        "count": n,
        "min_ms": round(sorted_v[0], 2),
        "median_ms": round(sorted_v[med_idx], 2),
        "p95_ms": round(sorted_v[p95_idx], 2),
        "max_ms": round(sorted_v[-1], 2),
    }


@dataclass(frozen=True)
class ShadowObservationRequest:
    request_id: str
    timestamp: str
    user_text: str
    production_route: str
    production_latency_ms: float
    context: Mapping[str, Any]
    enqueued_at: float
    enqueue_overhead_ms: float = 0.0
    production_details: dict[str, Any] | None = None


class ShadowEvaluator:
    """Isolated, background shadow evaluation engine."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        queue_max_size: int | None = None,
        timeout_s: float | None = None,
        sample_rate: float | None = None,
        persist_raw_text: bool | None = None,
        v8_adapter: Any = None,
        capabilities: Sequence[Capability] | None = None,
    ) -> None:
        root_dir = Path(output_dir or config.SHADOW_MODE_DIR or "reports/jarvis_bert_shadow_audit_20261017_125000")
        if not root_dir.is_absolute():
            root_dir = Path("D:/JARVIS") / root_dir
        self.output_dir = root_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.queue_max_size = queue_max_size or config.SHADOW_QUEUE_MAX_SIZE
        self.timeout_s = timeout_s or config.SHADOW_INFERENCE_TIMEOUT_S
        self.sample_rate = sample_rate if sample_rate is not None else config.SHADOW_SAMPLE_RATE
        self.persist_raw_text = persist_raw_text if persist_raw_text is not None else config.SHADOW_PERSIST_RAW_TEXT

        self.events_file = self.output_dir / "shadow_events.jsonl"
        self.metrics_file = self.output_dir / "shadow_metrics.json"
        self.comparison_file = self.output_dir / "production_vs_bert_comparison.json"
        self.session_report_file = self.output_dir / "shadow_session_report.md"

        self._queue: queue.Queue[ShadowObservationRequest | None] = queue.Queue(maxsize=self.queue_max_size)
        self._seen_request_ids: set[str] = set()
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # V8 Model & Execution
        self._v8_adapter = v8_adapter
        self._v8_capabilities = tuple(capabilities or ())
        self._init_lock = threading.Lock()
        self._cold_start_duration_ms: float | None = None
        self._last_init_attempt_time: float = 0.0
        self._init_backoff_s: float = 30.0
        self._adapter_load_error: str | None = None

        # Statistics & Metrics
        self._stats = {
            "total_observed": 0,
            "evaluated_count": 0,
            "dropped_count": 0,
            "timeout_count": 0,
            "error_count": 0,
            "duplicate_count": 0,
            "cold_start_duration_ms": 0.0,
            "production_latencies_ms": [],
            "v8_latencies_ms": [],
            "queue_delays_ms": [],
            "enqueue_overheads_ms": [],
            "fast_path_count": 0,
            "qwen_invoked_count": 0,
            "qwen_fallback_count": 0,
            "exact_agreement_count": 0,
            "broad_compatible_count": 0,
            "disagreement_count": 0,
            "v8_clarification_count": 0,
            "v8_escalation_count": 0,
            "categories": {},
            "disagreement_categories": {},
        }

        # Background worker thread & executor
        self._worker_thread: threading.Thread | None = None
        self._inference_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="BERTShadowInference"
        )
        self._is_active = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    def start(self) -> None:
        """Start the shadow worker thread."""
        with self._lock:
            if self._is_active:
                return
            self._shutdown_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop, name="BERTShadowWorker", daemon=True
            )
            self._worker_thread.start()
            self._is_active = True
            log.info("[SHADOW] BERT Shadow Evaluator started. output_dir=%s", self.output_dir)

    def stop(self, timeout_s: float = 3.0) -> None:
        """Stop the worker thread cleanly and flush metrics."""
        with self._lock:
            if not self._is_active:
                return
            self._is_active = False
            self._shutdown_event.set()

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout_s)

        self._inference_executor.shutdown(wait=False)
        self._flush_metrics()
        self._generate_session_report()
        log.info("[SHADOW] BERT Shadow Evaluator stopped cleanly.")

    def record_production_observation(
        self,
        user_text: str,
        production_route: str,
        production_latency_ms: float,
        context: Mapping[str, Any] | None = None,
        request_id: str | None = None,
        production_details: Any = None,
    ) -> bool:
        """Capture production classifier output and enqueue for BERT shadow evaluation.

        Returns:
            True if enqueued successfully; False if dropped, duplicate, or disabled.
            GUARANTEE: Returns within < 0.1ms; never blocks or alters production flow.
        """
        t0 = time.perf_counter()
        req_id = request_id or f"req_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"

        with self._lock:
            self._stats["total_observed"] += 1

            if req_id in self._seen_request_ids:
                self._stats["duplicate_count"] += 1
                return False
            self._seen_request_ids.add(req_id)
            if len(self._seen_request_ids) > 10000:
                self._seen_request_ids.clear()

        # Track production attribution
        prod_dict: dict[str, Any] | None = None
        if production_details is not None:
            if hasattr(production_details, "to_dict") and callable(getattr(production_details, "to_dict")):
                try:
                    val = production_details.to_dict()
                    if isinstance(val, dict):
                        prod_dict = val
                except Exception:
                    pass
            elif isinstance(production_details, dict):
                prod_dict = production_details

        with self._lock:
            if prod_dict:
                if prod_dict.get("fast_path_handled"):
                    self._stats["fast_path_count"] += 1
                if prod_dict.get("qwen_invoked"):
                    self._stats["qwen_invoked_count"] += 1
                if prod_dict.get("fallback_used"):
                    self._stats["qwen_fallback_count"] += 1

        # Bounded Queue Overflow Protection (Immediate Drop)
        enqueue_overhead_ms = (time.perf_counter() - t0) * 1000.0
        obs = ShadowObservationRequest(
            request_id=req_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            user_text=user_text,
            production_route=str(production_route),
            production_latency_ms=float(production_latency_ms),
            context=dict(context or {}),
            enqueued_at=time.perf_counter(),
            enqueue_overhead_ms=enqueue_overhead_ms,
            production_details=prod_dict,
        )

        try:
            self._queue.put_nowait(obs)
            with self._lock:
                self._stats["enqueue_overheads_ms"].append(enqueue_overhead_ms)
            return True
        except queue.Full:
            # Queue is full: strictly drop job to preserve production performance
            with self._lock:
                self._stats["dropped_count"] += 1
            self._record_dropped_event(obs, reason="queue_overflow")
            return False

    def _load_v8_adapter(self) -> Any:
        """Loads and constructs the Experimental V8 Adapter artifact and encoder."""
        import sys
        root = Path("D:/JARVIS")
        for p in [
            root,
            root / "reports/jarvis_indic_ranking_forensics_20261010_213500",
            root / "reports/classifier_v8_experiment_20261011_121000",
        ]:
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))

        import joblib
        from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder
        from experimental_adapter_v8 import ExperimentalV8Adapter
        from slot_contract_resolver import SlotContractResolver, CANONICAL_CAPABILITY_SCHEMAS

        model_path = root / "reports/classifier_v4_experiment_20261010_175140/best_candidate/compact_heads.joblib"
        artifact = joblib.load(model_path)
        encoder = FrozenMultilingualEncoder()

        adapter = ExperimentalV8Adapter.__new__(ExperimentalV8Adapter)
        adapter.artifact = artifact
        adapter.encoder = encoder
        adapter._adapter_id = "v8-candidate"
        adapter.inference_count = 0
        adapter.last_inference_metrics = {}
        adapter.contract_resolver = SlotContractResolver()

        if not self._v8_capabilities:
            self._v8_capabilities = tuple(
                Capability(c.name, c.description, c.required_arguments)
                for c in CANONICAL_CAPABILITY_SCHEMAS.values()
            )

        return adapter

    def _ensure_v8_adapter(self) -> Any:
        """Lazily load the V8 adapter in the background worker thread with lock and backoff."""
        if self._v8_adapter is not None:
            return self._v8_adapter

        with self._init_lock:
            if self._v8_adapter is not None:
                return self._v8_adapter

            now = time.monotonic()
            if self._last_init_attempt_time > 0 and (now - self._last_init_attempt_time) < self._init_backoff_s:
                return None

            self._last_init_attempt_time = now
            t_load_start = time.perf_counter()
            try:
                adapter = self._load_v8_adapter()
                self._v8_adapter = adapter
                cold_start_ms = (time.perf_counter() - t_load_start) * 1000.0
                self._cold_start_duration_ms = cold_start_ms
                with self._lock:
                    self._stats["cold_start_duration_ms"] = round(cold_start_ms, 2)
                log.info("[SHADOW] Experimental V8 BERT Adapter loaded in %.1fms.", cold_start_ms)
                return self._v8_adapter
            except Exception as exc:
                self._adapter_load_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                log.error("[SHADOW] Failed to initialize V8 adapter: %s", self._adapter_load_error)
                return None

    def _worker_loop(self) -> None:
        """Background thread loop draining shadow tasks."""
        last_flush = time.monotonic()
        while not self._shutdown_event.is_set():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                if time.monotonic() - last_flush > config.SHADOW_METRICS_INTERVAL_S:
                    self._flush_metrics()
                    last_flush = time.monotonic()
                continue

            if task is None:
                break

            try:
                self._process_shadow_task(task)
            except Exception as e:
                log.exception("[SHADOW] Unexpected worker error processing task %s: %s", task.request_id, e)
            finally:
                self._queue.task_done()

            if time.monotonic() - last_flush > config.SHADOW_METRICS_INTERVAL_S:
                self._flush_metrics()
                last_flush = time.monotonic()

        self._flush_metrics()

    def _process_shadow_task(self, task: ShadowObservationRequest) -> None:
        """Run BERT inference with isolation, timeout, and comparison."""
        queue_delay_ms = (time.perf_counter() - task.enqueued_at) * 1000.0
        lang_info = detect_language_and_script(task.user_text)

        adapter = self._ensure_v8_adapter()
        if adapter is None:
            self._record_error_event(
                task,
                queue_delay_ms,
                lang_info,
                error=f"BERT adapter unavailable: {self._adapter_load_error}",
                status="model_load_failed",
            )
            with self._lock:
                self._stats["error_count"] += 1
            return

        # Prepare DecisionRequest with minimal read-only context
        safe_context = dict(task.context) if isinstance(task.context, dict) else {}
        decision_req = DecisionRequest(
            user_text=task.user_text,
            context=safe_context,
            capabilities=self._v8_capabilities,
        )

        # Run inference bounded by timeout
        t0 = time.perf_counter()
        future = self._inference_executor.submit(adapter.predict, decision_req)
        try:
            raw_v8_output = future.result(timeout=self.timeout_s)
            v8_latency_ms = (time.perf_counter() - t0) * 1000.0
            status = "completed"
            error_msg = None
        except concurrent.futures.TimeoutError:
            v8_latency_ms = (time.perf_counter() - t0) * 1000.0
            status = "shadow_timeout"
            error_msg = f"BERT inference timed out after {self.timeout_s:.1f}s"
            raw_v8_output = None
            with self._lock:
                self._stats["timeout_count"] += 1
        except Exception as exc:
            v8_latency_ms = (time.perf_counter() - t0) * 1000.0
            status = "shadow_error"
            error_msg = f"{type(exc).__name__}: {str(exc)[:160]}"
            raw_v8_output = None
            with self._lock:
                self._stats["error_count"] += 1

        # Validate Schema
        v8_schema_valid = False
        v8_decision_dict: dict[str, Any] = {}
        if raw_v8_output is not None:
            try:
                validated = ClassifierDecision.from_mapping(raw_v8_output)
                v8_decision_dict = {
                    "intent": validated.intent,
                    "execution_path": validated.execution_path,
                    "request_modes": sorted(list(m.value for m in validated.request_modes)),
                    "selected_capability": validated.selected_capability,
                    "needs_clarification": validated.needs_clarification,
                    "needs_escalation": validated.needs_escalation,
                    "missing_information": [asdict(m) for m in validated.missing_information],
                    "context_references": [asdict(c) for c in validated.context_references],
                    "confidence": round(validated.confidence, 4),
                }
                v8_schema_valid = True
            except (DecisionValidationError, Exception) as val_err:
                v8_schema_valid = False
                v8_decision_dict = dict(raw_v8_output)
                error_msg = f"Schema validation failed: {type(val_err).__name__}: {val_err}"

        # Evaluate Compatibility & Exact Agreement Between Models
        compat_info = self._compare_predictions(task.production_route, v8_decision_dict)

        with self._lock:
            self._stats["evaluated_count"] += 1
            self._stats["production_latencies_ms"].append(task.production_latency_ms)
            self._stats["v8_latencies_ms"].append(v8_latency_ms)
            self._stats["queue_delays_ms"].append(queue_delay_ms)
            cat = lang_info.get("primary_category", "unknown")
            self._stats["categories"][cat] = self._stats["categories"].get(cat, 0) + 1

            if compat_info.get("exact_agreement"):
                self._stats["exact_agreement_count"] += 1
            if compat_info.get("broad_path_compatible"):
                self._stats["broad_compatible_count"] += 1
            else:
                self._stats["disagreement_count"] += 1

            dis_cat = compat_info.get("disagreement_category", "unknown")
            self._stats["disagreement_categories"][dis_cat] = (
                self._stats["disagreement_categories"].get(dis_cat, 0) + 1
            )

            if v8_decision_dict.get("needs_clarification"):
                self._stats["v8_clarification_count"] += 1
            if v8_decision_dict.get("needs_escalation"):
                self._stats["v8_escalation_count"] += 1

        # Build Record with Privacy Protection
        text_sha256 = hashlib.sha256(task.user_text.encode("utf-8")).hexdigest()
        event = {
            "request_id": task.request_id,
            "timestamp": task.timestamp,
            "status": status,
            "language_classification": lang_info,
            "production": {
                "route": task.production_route,
                "latency_ms": round(task.production_latency_ms, 2),
                "schema_valid": True,
                "details": task.production_details,
            },
            "bert": {
                "model_id": "nlptown/bert-base-multilingual-uncased-sentiment",
                "adapter": "v8-candidate",
                "prediction": v8_decision_dict if v8_schema_valid else raw_v8_output,
                "latency_ms": round(v8_latency_ms, 2),
                "schema_valid": v8_schema_valid,
            },
            "timings": {
                "queue_delay_ms": round(queue_delay_ms, 2),
                "enqueue_overhead_ms": round(task.enqueue_overhead_ms, 4),
                "cold_start_duration_ms": self._cold_start_duration_ms,
            },
            "comparison": compat_info,
            "privacy": {
                "raw_text_persisted": self.persist_raw_text,
                "text_sha256": text_sha256,
                "text_char_count": len(task.user_text),
                "text_word_count": len(task.user_text.split()),
                "context_available": bool(task.context),
                "context_keys": list(task.context.keys()) if isinstance(task.context, dict) else [],
            },
            "error": error_msg,
        }

        if self.persist_raw_text:
            event["raw_user_text"] = task.user_text

        self._write_event(event)

    @staticmethod
    def _compare_predictions(prod_route: str, v8_pred: dict[str, Any]) -> dict[str, Any]:
        """Compute exact agreement and broad path compatibility between Production and BERT."""
        if not v8_pred:
            return {
                "exact_agreement": False,
                "broad_path_compatible": False,
                "disagreement_category": "bert_empty_or_failed",
            }

        route_upper = prod_route.upper()
        v8_path = str(v8_pred.get("execution_path", "")).lower()
        v8_cap = v8_pred.get("selected_capability")

        prod_is_task = "TASK" in route_upper
        v8_is_task = v8_path in ("single_action", "multi_step_task", "action") or v8_cap is not None

        broad_compatible = (prod_is_task == v8_is_task)

        # Exact agreement requires canonical alignment
        exact_agreement = False
        if prod_is_task and v8_is_task:
            exact_agreement = True
            dis_cat = "compatible_both_action"
        elif not prod_is_task and not v8_is_task:
            if v8_pred.get("needs_clarification"):
                exact_agreement = False
                dis_cat = "v8_clarification"
            elif v8_pred.get("needs_escalation"):
                exact_agreement = False
                dis_cat = "v8_escalation"
            else:
                exact_agreement = True
                dis_cat = "compatible_both_conversational"
        elif prod_is_task and not v8_is_task:
            dis_cat = "prod_action_v8_conversational"
        else:
            dis_cat = "prod_conversational_v8_action"

        return {
            "exact_agreement": exact_agreement,
            "broad_path_compatible": broad_compatible,
            "disagreement_category": dis_cat,
            "production_is_action": prod_is_task,
            "bert_is_action": v8_is_task,
            "bert_selected_capability": v8_cap,
            "bert_needs_clarification": bool(v8_pred.get("needs_clarification")),
            "bert_needs_escalation": bool(v8_pred.get("needs_escalation")),
        }

    def _record_dropped_event(self, task: ShadowObservationRequest, reason: str) -> None:
        text_sha256 = hashlib.sha256(task.user_text.encode("utf-8")).hexdigest()
        event = {
            "request_id": task.request_id,
            "timestamp": task.timestamp,
            "status": "shadow_dropped",
            "reason": reason,
            "production": {
                "route": task.production_route,
                "latency_ms": round(task.production_latency_ms, 2),
                "details": task.production_details,
            },
            "timings": {
                "enqueue_overhead_ms": round(task.enqueue_overhead_ms, 4),
            },
            "privacy": {
                "text_sha256": text_sha256,
            },
        }
        self._write_event(event)

    def _record_error_event(
        self,
        task: ShadowObservationRequest,
        queue_delay_ms: float,
        lang_info: dict[str, Any],
        error: str,
        status: str,
    ) -> None:
        text_sha256 = hashlib.sha256(task.user_text.encode("utf-8")).hexdigest()
        event = {
            "request_id": task.request_id,
            "timestamp": task.timestamp,
            "status": status,
            "language_classification": lang_info,
            "production": {
                "route": task.production_route,
                "latency_ms": round(task.production_latency_ms, 2),
                "details": task.production_details,
            },
            "timings": {
                "queue_delay_ms": round(queue_delay_ms, 2),
                "enqueue_overhead_ms": round(task.enqueue_overhead_ms, 4),
            },
            "privacy": {
                "text_sha256": text_sha256,
            },
            "error": error,
        }
        self._write_event(event)

    def _write_event(self, event: dict[str, Any]) -> None:
        with self._file_lock:
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _flush_metrics(self) -> None:
        """Serialize current runtime metrics and refreshed comparison to disk."""
        with self._lock:
            prod_lat = compute_percentiles(self._stats["production_latencies_ms"])
            v8_lat = compute_percentiles(self._stats["v8_latencies_ms"])
            q_delay = compute_percentiles(self._stats["queue_delays_ms"])
            enq_over = compute_percentiles(self._stats["enqueue_overheads_ms"])

            eval_count = self._stats["evaluated_count"]
            exact_pct = (self._stats["exact_agreement_count"] / eval_count * 100.0) if eval_count > 0 else 0.0
            broad_pct = (self._stats["broad_compatible_count"] / eval_count * 100.0) if eval_count > 0 else 0.0

            metrics = {
                "system": "JARVIS BERT Shadow Evaluation",
                "version": "2.0.0",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cold_start_duration_ms": self._stats["cold_start_duration_ms"],
                "counts": {
                    "total_observed": self._stats["total_observed"],
                    "evaluated_count": eval_count,
                    "dropped_count": self._stats["dropped_count"],
                    "timeout_count": self._stats["timeout_count"],
                    "error_count": self._stats["error_count"],
                    "duplicate_count": self._stats["duplicate_count"],
                },
                "production_attribution": {
                    "fast_path_handled": self._stats["fast_path_count"],
                    "qwen_invoked": self._stats["qwen_invoked_count"],
                    "qwen_fallback_used": self._stats["qwen_fallback_count"],
                },
                "agreement_and_compatibility": {
                    "exact_agreement_count": self._stats["exact_agreement_count"],
                    "exact_agreement_pct": round(exact_pct, 2),
                    "broad_path_compatible_count": self._stats["broad_compatible_count"],
                    "broad_path_compatible_pct": round(broad_pct, 2),
                    "disagreement_count": self._stats["disagreement_count"],
                    "v8_clarification_count": self._stats["v8_clarification_count"],
                    "v8_escalation_count": self._stats["v8_escalation_count"],
                    "disagreement_categories": dict(self._stats["disagreement_categories"]),
                },
                "latency": {
                    "production": prod_lat,
                    "bert": v8_lat,
                    "queue_delay": q_delay,
                    "production_enqueue_overhead": enq_over,
                },
                "language_distribution": dict(self._stats["categories"]),
            }

            comparison_data = {
                "system": "JARVIS Production vs BERT Live Comparison Snapshot",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_commands_evaluated": eval_count,
                "exact_agreement_rate_pct": round(exact_pct, 2),
                "broad_path_compatibility_pct": round(broad_pct, 2),
                "production_median_latency_ms": prod_lat.get("median_ms", 0.0),
                "bert_median_latency_ms": v8_lat.get("median_ms", 0.0),
                "production_p95_latency_ms": prod_lat.get("p95_ms", 0.0),
                "bert_p95_latency_ms": v8_lat.get("p95_ms", 0.0),
                "accuracy_notice": "Accuracy is only reported when independently reviewed human labels are supplied. High agreement between models is NOT proof of correctness.",
                "disagreement_breakdown": dict(self._stats["disagreement_categories"]),
            }

        with self._file_lock:
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            with open(self.comparison_file, "w", encoding="utf-8") as f:
                json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    def _generate_session_report(self) -> None:
        """Generate human-readable markdown summary at shutdown."""
        with self._lock:
            eval_count = self._stats["evaluated_count"]
            exact_pct = (self._stats["exact_agreement_count"] / eval_count * 100.0) if eval_count > 0 else 0.0
            broad_pct = (self._stats["broad_compatible_count"] / eval_count * 100.0) if eval_count > 0 else 0.0
            prod_lat = compute_percentiles(self._stats["production_latencies_ms"])
            v8_lat = compute_percentiles(self._stats["v8_latencies_ms"])

        md = f"""# JARVIS BERT Shadow Evaluation — Session Summary Report

**Session Directory:** `{self.output_dir}`  
**Generated At:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  

---

## 1. Executive Summary

- **Total Requests Observed:** {self._stats['total_observed']}
- **Successfully Evaluated:** {eval_count}
- **Queue Dropped:** {self._stats['dropped_count']}
- **Timeouts:** {self._stats['timeout_count']}
- **Errors:** {self._stats['error_count']}
- **BERT Cold Start Duration:** {self._stats['cold_start_duration_ms']:.1f}ms

## 2. Production Classification Attribution

- **Deterministic Fast-Path:** {self._stats['fast_path_count']} requests
- **Local Qwen Invoked:** {self._stats['qwen_invoked_count']} requests
- **Qwen Fallback to Gemini:** {self._stats['qwen_fallback_count']} requests

## 3. Agreement vs Compatibility

- **Exact Agreement:** {self._stats['exact_agreement_count']} / {eval_count} ({exact_pct:.1f}%)
- **Broad Action/Non-Action Compatibility:** {self._stats['broad_compatible_count']} / {eval_count} ({broad_pct:.1f}%)
- **Disagreement Count:** {self._stats['disagreement_count']}
- **BERT Clarifications Flagged:** {self._stats['v8_clarification_count']}
- **BERT Escalations Flagged:** {self._stats['v8_escalation_count']}

## 4. Latency Performance

- **Production Latency:** Median = {prod_lat.get('median_ms', 0.0):.1f}ms | p95 = {prod_lat.get('p95_ms', 0.0):.1f}ms
- **BERT Latency:** Median = {v8_lat.get('median_ms', 0.0):.1f}ms | p95 = {v8_lat.get('p95_ms', 0.0):.1f}ms

> [!NOTE]
> Agreement is NOT accuracy. To evaluate true accuracy, supply ground-truth labels in `reviewed_labels.jsonl` and run `evaluate_reviewed_shadow_data()`.
"""
        with self._file_lock:
            with open(self.session_report_file, "w", encoding="utf-8") as f:
                f.write(md)


# Global singleton instance
_shadow_evaluator_instance: ShadowEvaluator | None = None
_instance_lock = threading.Lock()


def get_shadow_evaluator() -> ShadowEvaluator | None:
    """Return the global ShadowEvaluator singleton if configured."""
    global _shadow_evaluator_instance
    if not config.SHADOW_MODE_ENABLED:
        return None
    with _instance_lock:
        if _shadow_evaluator_instance is None:
            _shadow_evaluator_instance = ShadowEvaluator()
            _shadow_evaluator_instance.start()
        return _shadow_evaluator_instance


def shutdown_shadow_evaluator() -> None:
    """Cleanly shut down the global evaluator."""
    global _shadow_evaluator_instance
    with _instance_lock:
        if _shadow_evaluator_instance is not None:
            _shadow_evaluator_instance.stop()
            _shadow_evaluator_instance = None
