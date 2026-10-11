"""JSON-lines worker for the isolated Laya Phase 4 benchmark environment."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any

import laya
import torch


MODEL_ID = "convaiinnovations/laya"
SUBFOLDER = "multilingual"


def question_set(capabilities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    capability_options = {"none": "No advertised capability should be selected."}
    capability_options.update({item["name"]: item["description"] for item in capabilities})
    intent_options = {
        "respond": "Converse or answer stable knowledge directly.",
        "recall": "Recall information from prior conversation or supplied context.",
        "research": "Obtain current or external information.",
        "clarify_request": "Ask for essential missing information.",
        "unsupported_request": "The requested operation has no advertised capability.",
    }
    intent_options.update({
        f"use:{item['name']}": f"Perform the operation described by capability {item['name']}: {item['description']}"
        for item in capabilities
    })
    return {
        "execution_path": {
            "type": "choice",
            "instructions": "Choose the semantic execution path for the user's request, considering context and advertised capabilities.",
            "criteria": {
                "direct_response": "Answer or converse without a capability.",
                "memory_recall": "Use prior conversation or supplied memory.",
                "research": "Use one capability to obtain current or external information.",
                "single_action": "One advertised capability completes the request.",
                "multi_step_task": "Several steps or retrieval followed by synthesis are required.",
                "clarify": "Essential information or a reference is unresolved.",
                "escalate": "No advertised capability can perform the requested operation.",
            },
        },
        "intent": {"type": "choice", "instructions": "Choose the user's underlying intent.", "criteria": intent_options},
        "request_modes": {
            "type": "choice", "instructions": "Choose the complete set of broad request modes.",
            "criteria": {
                "conversational": "Conversational or stable knowledge response only.",
                "memory": "Memory recall only.",
                "research+action": "External/current research requiring a capability.",
                "action": "Action-oriented without research.",
            },
        },
        "capability": {"type": "choice", "instructions": "Choose only from the advertised capabilities, or none.", "criteria": capability_options},
        "clarification": {
            "type": "noul", "instructions": "Is essential information missing such that JARVIS must ask a clarifying question?",
            "criteria": {"false": "The request is sufficiently specified or unsupported.", "true": "An essential argument or reference is unresolved."},
        },
        "escalation": {
            "type": "noul", "instructions": "Is the requested operation unsupported by every advertised capability?",
            "criteria": {"false": "The request can be answered, clarified, or handled by an advertised capability.", "true": "No advertised capability can perform the requested operation."},
        },
    }


def confidence(answer: dict[str, Any]) -> float:
    value = answer.get("confidence")
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    probabilities = answer.get("probabilities", {})
    if isinstance(probabilities, dict) and probabilities:
        return max(float(item) for item in probabilities.values())
    if "noul" in answer:
        probability = float(answer["noul"])
        return max(probability, 1.0 - probability)
    return 0.0


def decide(agent: Any, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    state_data = {
        "user_text": request["user_text"],
        "context": request.get("context", {}),
        "available_capabilities": request.get("capabilities", []),
    }
    # Laya 0.3.20's multilingual tokenizer rejects some Unicode-containing dict
    # states. Its documented string-state interface preserves the same data.
    state = json.dumps(state_data, ensure_ascii=False, sort_keys=True)
    questions = question_set(state_data["available_capabilities"])
    started = time.perf_counter_ns()
    result = agent.predict(state, questions, max_len=1024)
    model_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    answers = result.get("answers", result)
    path = answers["execution_path"]["choice"]
    clarification_p = float(answers["clarification"].get("noul", 0.0))
    escalation_p = float(answers["escalation"].get("noul", 0.0))
    if clarification_p >= 0.5 or path == "clarify":
        path = "clarify"
    if escalation_p >= 0.5 or path == "escalate":
        path = "escalate" if escalation_p >= clarification_p else "clarify"
    needs_clarification = path == "clarify"
    needs_escalation = path == "escalate"

    advertised = {item["name"] for item in state_data["available_capabilities"]}
    selected = answers["capability"]["choice"]
    if selected == "none" or selected not in advertised or path not in {"research", "single_action", "multi_step_task"}:
        selected = None
    mode_choice = answers["request_modes"]["choice"]
    modes = {"conversational": ["conversational"], "memory": ["memory"],
             "research+action": ["research", "action"], "action": ["action"]}[mode_choice]
    intent = answers["intent"]["choice"]
    if intent.startswith("use:") and intent[4:] not in advertised:
        intent = "unsupported_request"
    missing = []
    references = []
    if needs_clarification:
        missing = [{"name": "required_information", "description": "An essential argument or reference is unresolved."}]
        references = [{"reference": "unresolved request reference", "kind": "unspecified", "resolved_value": None}]
    confidences = [confidence(value) for value in answers.values() if isinstance(value, dict)]
    decision = {
        "intent": intent, "execution_path": path, "request_modes": modes,
        "selected_capability": selected, "context_references": references,
        "missing_information": missing, "confidence": min(confidences) if confidences else 0.0,
        "needs_clarification": needs_clarification, "needs_escalation": needs_escalation,
        "evidence": [{"kind": "adapter", "reference": "laya-multilingual-0.3.20"}],
    }
    return decision, {"model_inference_ms": model_ms, "preprocessing_ms": 0.0, "postprocessing_ms": 0.0}


def emit(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    rss_before = _rss()
    started = time.perf_counter_ns()
    agent = laya.load(MODEL_ID, subfolder=SUBFOLDER, device="cpu")
    load_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    emit({"status": "ready", "adapter_id": "laya-multilingual-0.3.20", "package_version": laya.__version__,
          "model_id": MODEL_ID, "subfolder": SUBFOLDER, "device": "cpu", "torch_cuda_available": torch.cuda.is_available(),
          "model_load_ms": load_ms, "rss_before_bytes": rss_before, "rss_after_bytes": _rss()})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                break
            decision, timings = decide(agent, request)
            emit({"status": "success", "decision": decision, "timings": timings})
        except Exception as exc:
            emit({"status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def _rss() -> int | None:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        return None


if __name__ == "__main__":
    main()
