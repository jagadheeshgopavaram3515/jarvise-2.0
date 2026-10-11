from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from assistant import config
from assistant.llm.ollama import OllamaClient

PROMPT_VERSION = "base-v1"


@dataclass
class AdapterResult:
    system: str
    case_id: str
    status: str
    raw_output: str = ""
    parsed_output: Any = None
    model: str = ""
    prompt_chars: int = 0
    latency_ms: float = 0.0
    parsing_latency_ms: float = 0.0
    timeout_s: float = 0.0
    exception: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseQwenAdapter:
    """Benchmark transport using OllamaClient's existing prompt builder.

    It intentionally does not use OllamaClient.stream(), because that production
    method converts transport failures into spoken fallback text. Benchmark output
    must retain the failure state.
    """

    name = "base_qwen"

    def __init__(self, timeout_s: float | None = None):
        self.client = OllamaClient()
        self.timeout_s = float(timeout_s or config.OLLAMA_TIMEOUT_S)

    def run(self, case) -> dict[str, Any]:
        envelope = {
            "request": case.user_input,
            "context": case.context,
            "capabilities": case.available_capabilities,
            "observations": case.observations,
        }
        prompt = ("Return JSON with keys action, tool_name, tool_args, final_response, thought.\n"
                  + json.dumps(envelope, ensure_ascii=False, sort_keys=True))
        started = time.perf_counter()
        result = AdapterResult(self.name, case.id, "MODEL_ERROR", model=self.client.model,
                               prompt_chars=len(prompt), timeout_s=self.timeout_s)
        try:
            payload = {"model": self.client.model, "prompt": self.client._build_prompt(prompt),
                       "stream": False, "format": "json", "keep_alive": config.OLLAMA_KEEP_ALIVE,
                       "options": {"temperature": 0.3}}
            result.prompt_chars = len(payload["prompt"])
            req = urllib.request.Request(f"{self.client.base_url}/api/generate",
                                         data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json", "User-Agent": "JARVIS-Replay"})
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
            result.raw_output = str(data.get("response", ""))
            parse_started = time.perf_counter()
            result.parsed_output = json.loads(result.raw_output)
            result.parsing_latency_ms = (time.perf_counter() - parse_started) * 1000
            result.status = "MODEL_SUCCESS"
        except TimeoutError as exc:
            result.status, result.exception = "OLLAMA_TIMEOUT", repr(exc)
        except (urllib.error.URLError, OSError) as exc:
            result.status, result.exception = "OLLAMA_ERROR", repr(exc)
        except json.JSONDecodeError as exc:
            result.status, result.exception = "PARSER_ERROR", repr(exc)
        except Exception as exc:
            result.status, result.exception = "MODEL_ERROR", repr(exc)
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result.to_dict()
