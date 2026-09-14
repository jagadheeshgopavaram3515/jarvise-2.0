"""
Voice-activity detection wrapper around webrtcvad.

webrtcvad wants 16-bit mono PCM frames of exactly 10/20/30 ms. We feed it
30 ms frames and expose a tiny is_speech() helper plus a graceful fallback
to a simple energy gate if webrtcvad isn't installed.
"""
from __future__ import annotations

import numpy as np

from assistant import config

try:
    import webrtcvad
    _HAS_WEBRTC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_WEBRTC = False


class VAD:
    def __init__(self, aggressiveness: int = config.VAD_AGGRESSIVENESS,
                 sample_rate: int = config.STT_SAMPLE_RATE):
        self.sample_rate = sample_rate
        if _HAS_WEBRTC:
            self._vad = webrtcvad.Vad(aggressiveness)
        else:
            self._vad = None
            # Energy threshold for the fallback gate (tuned for normalized int16).
            self._energy_threshold = 500.0

    def is_speech(self, frame_int16: bytes) -> bool:
        if self._vad is not None:
            try:
                return self._vad.is_speech(frame_int16, self.sample_rate)
            except Exception:
                return False
        # Fallback: RMS energy gate.
        samples = np.frombuffer(frame_int16, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(samples ** 2)))
        return rms > self._energy_threshold
