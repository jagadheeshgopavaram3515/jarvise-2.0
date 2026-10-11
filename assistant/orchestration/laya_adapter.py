"""Offline-only Laya adapter used by the Phase 4 comparison benchmark."""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Mapping

from assistant.orchestration.decision import DecisionRequest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = ROOT / ".benchmark-envs" / "laya" / "Scripts" / "python.exe"
DEFAULT_WORKER = ROOT / "benchmarks" / "fast_classifier" / "laya_worker.py"


class LayaOfflineAdapter:
    """Persistent subprocess boundary for one-forward Laya decisions.

    Laya is intentionally installed in an isolated benchmark environment. This
    module is not imported by production routing and never executes a capability.
    """

    def __init__(
        self,
        python_executable: str | Path = DEFAULT_PYTHON,
        worker_path: str | Path = DEFAULT_WORKER,
    ) -> None:
        command = [str(python_executable), "-u", str(worker_path)]
        child_env = dict(os.environ)
        child_env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
        self.process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=child_env,
        )
        self._lock = threading.Lock()
        ready = self._read_message()
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(f"Laya worker failed to start: {ready}")
        self.metadata = ready
        self._adapter_id = str(ready["adapter_id"])
        self.last_inference_metrics: dict[str, float] = {}

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    def predict(self, request: DecisionRequest) -> Mapping[str, Any]:
        payload = {
            "user_text": request.user_text,
            "context": dict(request.context),
            "capabilities": [
                {
                    "name": capability.name,
                    "description": capability.description,
                    "required_arguments": list(capability.required_arguments),
                }
                for capability in request.capabilities
            ],
        }
        with self._lock:
            if self.process.stdin is None:
                raise RuntimeError("Laya worker input is closed")
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            response = self._read_message()
        if response.get("status") != "success":
            detail = response.get("traceback") or response.get("error", "Laya worker failed")
            raise RuntimeError(str(detail))
        self.last_inference_metrics = {
            key: float(value)
            for key, value in response.get("timings", {}).items()
            if key.endswith("_ms") and isinstance(value, (int, float))
        }
        return response["decision"]

    def close(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin is not None:
                try:
                    self.process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                    self.process.stdin.flush()
                except OSError:
                    pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)

    def _read_message(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("Laya worker output is closed")
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"Laya worker exited ({self.process.poll()}): {stderr[-2000:]}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError("Laya worker returned a non-object")
        return value

    def __enter__(self) -> "LayaOfflineAdapter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["LayaOfflineAdapter"]

