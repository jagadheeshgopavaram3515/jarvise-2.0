"""
Performance instrumentation for the voice pipeline.

A Perf object is created when the user stops speaking and travels with the
work across the STT / Gemini / TTS threads. When the assistant's audio
starts, the Perf is committed: printed as [PERF] lines, appended to
performance_log.json, and folded into a rolling 20-interaction average.

Pure measurement — no functional behaviour depends on this.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from assistant import config
from assistant.core.log import get

_LOG_PATH = os.path.join(config.DATA_DIR, "performance_log.json")
_ROLLING_N = 20
# Cap the on-disk log: it's pure instrumentation, so old records are dead weight.
# Translate the line budget into a record budget (~16 serialized lines per record).
_MAX_RECORDS = max(_ROLLING_N, config.PERF_LOG_MAX_LINES // config.PERF_LOG_LINES_PER_RECORD)
log = get("perf")


@dataclass
class Perf:
    """Timing accumulator for one user→assistant interaction."""
    t_start: float            # perf_counter() when the user stopped speaking
    speech_start: float = 0.0
    stt_queued: float = 0.0
    stt_start: float = 0.0
    stt_end: float = 0.0
    dispatch_start: float = 0.0
    gemini_start: float = 0.0
    gemini_first: float = 0.0
    gemini_end: float = 0.0
    tts_queued: float = 0.0
    tts_synth_start: float = 0.0
    tts_synth_end: float = 0.0
    tts_play_start: float = 0.0
    tts_play_end: float = 0.0
    stt: float = 0.0          # transcription time
    gemini: float = 0.0       # request → first response text
    tts_synth: float = 0.0    # synthesis start → audio ready
    tts_start: float = 0.0    # synthesis start → playback begins
    total: float = 0.0        # user stopped speaking → audio starts
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    committed: bool = field(default=False, repr=False)

    def stamp(self, name: str, value: float | None = None):
        setattr(self, name, value if value is not None else time.perf_counter())

    def since_end(self, timestamp: float) -> float:
        return max(0.0, timestamp - self.t_start) if timestamp else 0.0

    def trace(self, stage: str, **extra):
        now = time.perf_counter()
        details = " ".join(f"{k}={v}" for k, v in extra.items() if v is not None)
        log.info(
            "TRACE %s %-16s +%.3fs %s",
            self.trace_id,
            stage,
            self.since_end(now),
            details,
        )


class PerfTracker:
    def __init__(self, path: str = _LOG_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._records: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if os.path.exists(self.path):
            try:
                data = json.load(open(self.path, encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return []

    def commit(self, perf: Perf):
        """Record one interaction; print [PERF]/[AVG] and persist."""
        if perf.committed:
            return
        perf.committed = True

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "trace_id": perf.trace_id,
            "speech_duration": round(perf.t_start - perf.speech_start, 2) if perf.speech_start else 0.0,
            "stt_queue_wait": round(perf.stt_start - perf.stt_queued, 2) if perf.stt_start and perf.stt_queued else 0.0,
            "stt": round(perf.stt, 2),
            "dispatch_wait": round(perf.dispatch_start - perf.stt_end, 2) if perf.dispatch_start and perf.stt_end else 0.0,
            "gemini": round(perf.gemini, 2),
            "gemini_total": round(perf.gemini_end - perf.gemini_start, 2) if perf.gemini_end and perf.gemini_start else 0.0,
            "tts_queue_wait": round(perf.tts_synth_start - perf.tts_queued, 2) if perf.tts_synth_start and perf.tts_queued else 0.0,
            "tts_synth": round(perf.tts_synth, 2),
            "tts_start": round(perf.tts_start, 2),
            "tts_play": round(perf.tts_play_end - perf.tts_play_start, 2) if perf.tts_play_end and perf.tts_play_start else 0.0,
            "tts": round(perf.tts_synth, 2),   # alias for the example schema
            "total": round(perf.total, 2),
        }

        log.info(
            "SUMMARY %s speech=%.2fs stt_wait=%.2fs stt=%.2fs dispatch_wait=%.2fs "
            "gemini_first=%.2fs gemini_total=%.2fs tts_wait=%.2fs synth=%.2fs "
            "audio_start=%.2fs play=%.2fs total=%.2fs",
            perf.trace_id,
            record["speech_duration"],
            record["stt_queue_wait"],
            record["stt"],
            record["dispatch_wait"],
            record["gemini"],
            record["gemini_total"],
            record["tts_queue_wait"],
            record["tts_synth"],
            record["tts_start"],
            record["tts_play"],
            record["total"],
        )

        with self._lock:
            self._records.append(record)
            self._trim()
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(self._records, f, indent=2)
            except Exception as e:
                print(f"[PERF] log write error: {e}")
            self._print_averages()

    def _trim(self):
        """Drop the oldest records so the log stays under the line budget."""
        overflow = len(self._records) - _MAX_RECORDS
        if overflow > 0:
            del self._records[:overflow]
            log.info("trimmed %d old record(s); keeping latest %d", overflow, _MAX_RECORDS)

    def _print_averages(self):
        window = self._records[-_ROLLING_N:]
        if not window:
            return

        def avg(key):
            vals = [r.get(key, 0.0) for r in window]
            return sum(vals) / len(vals)

        print(f"[AVG] (last {len(window)})")
        print(f"STT: {avg('stt'):.2f} s")
        print(f"Gemini: {avg('gemini'):.2f} s")
        print(f"TTS: {avg('tts_synth'):.2f} s")
        print(f"Total: {avg('total'):.2f} s")


# Singleton used across services.
tracker = PerfTracker()
