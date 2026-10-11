from __future__ import annotations

from typing import Any

from .schema import ReplayCase


class FastBrainNotImplemented:
    name = "fast_brain_qwen"

    def run(self, case: ReplayCase) -> dict[str, Any]:
        return {"status": "NOT_IMPLEMENTED", "system": self.name, "case_id": case.id}
