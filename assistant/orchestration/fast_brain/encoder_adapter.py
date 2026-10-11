"""Offline, local-files-only encoder foundation for Fast Brain experiments.

This module intentionally produces raw tensors only.  It does not derive a
FastBrainDecision, route requests, or execute any JARVIS capability.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import CapabilityFamily, IntentFamily, MissingInformation


class AdapterStatus(str, Enum):
    NOT_INITIALIZED = "not_initialized"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    ERROR = "error"


def _values(enum_type: type[Enum]) -> tuple[str, ...]:
    return tuple(member.value for member in enum_type)


INTENT_CLASSES = _values(IntentFamily)
CAPABILITY_CLASSES = _values(CapabilityFamily)
MISSING_INFORMATION_CLASSES = _values(MissingInformation)


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str = "distilbert-base-multilingual-cased"
    local_model_path: str | Path | None = None
    max_length: int = 160
    projection_dims: tuple[int, int] = (256, 128)
    device: str = "cpu"
    dtype: str = "float32"
    freeze_encoder: bool = True


@dataclass
class EncoderOutput:
    status: AdapterStatus
    intent_logits: Any = None
    capability_logits: Any = None
    clarification_logit: Any = None
    missing_information_logits: Any = None
    auxiliary_logits: Mapping[str, Any] = field(default_factory=dict)
    token_count: int = 0
    sequence_length: int = 0
    truncated: bool = False
    tokenization_latency_ms: float = 0.0
    encoder_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    error: str | None = None


class MultilingualEncoderAdapter:
    """Local-only DistilBERT-shaped encoder with raw classification heads.

    Construction never downloads artifacts.  A missing path results in the
    ``UNAVAILABLE`` state; inference returns a structured unavailable result.
    """

    backend_name = "multilingual_encoder_raw_logits"

    def __init__(self, config: EncoderConfig | None = None, *, auto_load: bool = True) -> None:
        self.config = config or EncoderConfig()
        self.status = AdapterStatus.NOT_INITIALIZED
        self.tokenizer: Any = None
        self.encoder: Any = None
        self.projection: Any = None
        self.heads: Any = None
        self.last_error: str | None = None
        if auto_load:
            self.load()

    @property
    def class_mappings(self) -> dict[str, tuple[str, ...]]:
        return {
            "intent": INTENT_CLASSES,
            "capability": CAPABILITY_CLASSES,
            "missing_information": MISSING_INFORMATION_CLASSES,
        }

    def load(self) -> AdapterStatus:
        path = Path(self.config.local_model_path) if self.config.local_model_path else None
        if path is None or not path.exists() or not path.is_dir():
            self.status = AdapterStatus.UNAVAILABLE
            self.last_error = "local model checkpoint directory is unavailable"
            return self.status
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            self.encoder = AutoModel.from_pretrained(str(path), local_files_only=True)
            self.encoder.to(self.config.device)
            self.encoder.eval()
            if self.config.freeze_encoder:
                for parameter in self.encoder.parameters():
                    parameter.requires_grad_(False)
            hidden_size = int(getattr(self.encoder.config, "hidden_size"))
            first, second = self.config.projection_dims
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, first), torch.nn.ReLU(),
                torch.nn.Linear(first, second), torch.nn.ReLU(),
            ).to(self.config.device)
            self.heads = torch.nn.ModuleDict({
                "intent": torch.nn.Linear(second, len(INTENT_CLASSES)),
                "capability": torch.nn.Linear(second, len(CAPABILITY_CLASSES)),
                "clarification": torch.nn.Linear(second, 1),
                "missing_information": torch.nn.Linear(second, len(MISSING_INFORMATION_CLASSES)),
            }).to(self.config.device)
            self.status = AdapterStatus.AVAILABLE
            self.last_error = None
        except ImportError as exc:
            self.status = AdapterStatus.UNAVAILABLE
            self.last_error = f"optional encoder dependency unavailable: {exc}"
        except Exception as exc:
            self.status = AdapterStatus.INVALID
            self.last_error = f"local model load failed: {exc}"
        return self.status

    @staticmethod
    def masked_mean_pool(hidden_states: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        denominator = mask.sum(dim=1).clamp(min=1)
        return (hidden_states * mask).sum(dim=1) / denominator

    def tokenize_metadata(self, transcript: str) -> dict[str, Any]:
        if self.tokenizer is None:
            return {"status": self.status.value, "token_count": 0, "sequence_length": 0, "truncated": False}
        raw = self.tokenizer(transcript, add_special_tokens=True, truncation=False)
        bounded = self.tokenizer(
            transcript, add_special_tokens=True, truncation=True,
            max_length=self.config.max_length,
        )
        raw_count = len(raw.get("input_ids", []))
        bounded_count = len(bounded.get("input_ids", []))
        return {
            "status": self.status.value,
            "token_count": raw_count,
            "sequence_length": bounded_count,
            "truncated": raw_count > bounded_count,
        }

    def infer(self, transcript: str) -> EncoderOutput:
        started = time.perf_counter_ns()
        if self.status is not AdapterStatus.AVAILABLE:
            return EncoderOutput(status=self.status, error=self.last_error,
                                 total_latency_ms=(time.perf_counter_ns() - started) / 1_000_000)
        try:
            import torch
            token_started = time.perf_counter_ns()
            encoded = self.tokenizer(
                [transcript], padding=True, truncation=True,
                max_length=self.config.max_length, return_tensors="pt",
            )
            metadata = self.tokenize_metadata(transcript)
            encoded = {key: value.to(self.config.device) for key, value in encoded.items()}
            token_ms = (time.perf_counter_ns() - token_started) / 1_000_000
            model_started = time.perf_counter_ns()
            context = torch.inference_mode() if self.config.freeze_encoder else torch.enable_grad()
            with context:
                hidden = self.encoder(**encoded).last_hidden_state
                pooled = self.masked_mean_pool(hidden, encoded["attention_mask"])
                projected = self.projection(pooled)
                outputs = {name: head(projected) for name, head in self.heads.items()}
            model_ms = (time.perf_counter_ns() - model_started) / 1_000_000
            total_ms = (time.perf_counter_ns() - started) / 1_000_000
            return EncoderOutput(
                status=AdapterStatus.AVAILABLE,
                intent_logits=outputs["intent"], capability_logits=outputs["capability"],
                clarification_logit=outputs["clarification"],
                missing_information_logits=outputs["missing_information"],
                token_count=metadata["token_count"], sequence_length=metadata["sequence_length"],
                truncated=metadata["truncated"], tokenization_latency_ms=token_ms,
                encoder_latency_ms=model_ms, total_latency_ms=total_ms,
            )
        except Exception as exc:
            self.status = AdapterStatus.ERROR
            self.last_error = f"inference failed: {exc}"
            return EncoderOutput(status=self.status, error=self.last_error,
                                 total_latency_ms=(time.perf_counter_ns() - started) / 1_000_000)

