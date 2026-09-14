"""
Optional acoustic echo-control hook.

Python bindings for full WebRTC Audio Processing / SpeexDSP AEC are not
standard on Windows. This module tries known bindings if present and otherwise
uses a conservative fallback: while Jarvis is speaking, very quiet mic frames
are suppressed before VAD sees them. That reduces self-triggering without
pretending to be hardware-grade AEC.
"""
from __future__ import annotations

import numpy as np

from assistant import config
from assistant.core.log import get

log = get("aec")


class EchoCanceller:
    def __init__(self, sample_rate: int = config.STT_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.backend = "suppression"
        self._apm = None
        if not config.ENABLE_AEC:
            self.backend = "disabled"
            return
        self._try_webrtc_apm()

    def _try_webrtc_apm(self):
        try:
            # Optional third-party packages expose different APIs. Keep this
            # deliberately defensive; the fallback is always safe.
            import webrtc_audio_processing as wap  # type: ignore

            self._apm = wap.AudioProcessingModule(
                enable_aec=True,
                enable_ns=True,
                enable_agc=False,
            )
            self.backend = "webrtc_audio_processing"
            log.info("AEC enabled via %s", self.backend)
        except Exception:
            log.info("AEC backend unavailable; using echo suppression fallback")

    def process_audio(self, mic_audio: bytes, speaker_reference=None,
                      assistant_speaking: bool = False) -> bytes:
        """Process mic audio before VAD/STT.

        `speaker_reference` is reserved for real WebRTC/Speex/ReSpeaker AEC.
        The current fallback intentionally preserves existing suppression-only
        behavior.
        """
        if not config.ENABLE_AEC or self.backend == "disabled":
            return mic_audio
        if self._apm is not None:
            try:
                processed = self._process_apm(mic_audio, speaker_reference)
                return processed or mic_audio
            except Exception:
                return mic_audio
        return self._suppression_only(mic_audio, assistant_speaking, speaker_reference)

    def process_capture(self, frame: bytes, assistant_speaking: bool) -> bytes:
        return self.process_audio(frame, speaker_reference=None,
                                  assistant_speaking=assistant_speaking)

    def _process_apm(self, mic_audio: bytes, speaker_reference) -> bytes:
        """Drive a real APM. Feed the far-end (render) signal first so it can
        model and subtract the echo path. Binding APIs vary, so each step is
        best-effort and the worst case degrades to plain near-end processing."""
        if speaker_reference:
            for meth in ("process_reverse_stream", "analyze_reverse_stream",
                         "set_render_signal"):
                fn = getattr(self._apm, meth, None)
                if fn is not None:
                    try:
                        fn(speaker_reference)
                        break
                    except Exception:
                        pass
        return self._apm.process_stream(mic_audio)

    def _suppression_only(self, mic_audio: bytes, assistant_speaking: bool,
                          speaker_reference=None) -> bytes:
        # FIX G: prefer the actual speaker reference. If Jarvis isn't really
        # outputting audio right now there's no echo to cancel — pass the user
        # through. This replaces the old blind energy gate that muted real speech.
        ref_rms = self._rms(speaker_reference) if speaker_reference else 0.0
        if not (assistant_speaking or ref_rms > 0.0):
            return mic_audio

        mic_rms = self._rms(mic_audio)
        if mic_rms == 0.0:
            return mic_audio
        if ref_rms > 0.0:
            # With a reference we can compare levels: genuine user speech rides
            # clearly above the echo bed; anything at/below it is echo → mute.
            return mic_audio if mic_rms > ref_rms * 1.6 else bytes(len(mic_audio))
        # No reference available (e.g. pygame backend) → fixed fallback gate.
        return mic_audio if mic_rms >= 900.0 else bytes(len(mic_audio))

    @staticmethod
    def _rms(pcm: bytes) -> float:
        if not pcm:
            return 0.0
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
