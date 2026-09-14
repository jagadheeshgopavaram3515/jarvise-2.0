"""
Cloud STT for conversational requests (Phase 1).

Whisper remains the command + fallback engine. For *conversational* finals
(stories, explanations, long chats), we optionally re-transcribe the captured
audio with Gemini's multimodal audio for higher accuracy, then dispatch that
text instead of Whisper's. Any failure/timeout → keep the Whisper transcript.

This deliberately reuses the existing GOOGLE_API_KEY. Note: each conversational
turn then costs one extra Gemini call, so it is OFF by default (HYBRID_STT) to
avoid reigniting rate limits — flip it on when quota allows.
"""
from __future__ import annotations

import io
import re
import wave
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import numpy as np

from assistant import config
from assistant.core.log import get

log = get("stt.cloud")

# Deterministic local commands → Whisper only (never sent to the cloud).
_COMMAND_RE = re.compile(
    r"\b(time|date|open|close|search|play|volume|mute|unmute|sleep|wake|"
    r"remind|reminder|timer|alarm|stop|cancel|pause|resume|quit|exit)\b",
    re.I,
)


def looks_like_command(text: str) -> bool:
    """True for short deterministic commands that should stay on Whisper."""
    t = (text or "").strip()
    if not t:
        return False
    if _COMMAND_RE.search(t):
        return True
    return len(t.split()) <= 2          # very short utterances are command-like


def _encode_wav(audio: np.ndarray, sample_rate: int = config.STT_SAMPLE_RATE) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class GeminiAudioSTT:
    """Transcribe audio via Gemini multimodal. Best-effort: returns None on any
    failure/timeout so the caller falls back to Whisper."""

    def __init__(self):
        from google import genai
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)
        try:
            from google.genai import types
            self._types = types
        except Exception:
            self._types = None
        self.model = config.HYBRID_STT_MODEL
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="CloudSTT")

    def transcribe(self, audio: np.ndarray) -> str | None:
        if self._types is None:
            return None
        try:
            fut = self._executor.submit(self._transcribe_blocking, audio)
            return fut.result(timeout=config.HYBRID_STT_TIMEOUT_S)
        except FuturesTimeout:
            log.debug("cloud STT > %.1fs — Whisper fallback", config.HYBRID_STT_TIMEOUT_S)
            return None
        except Exception:
            log.debug("cloud STT error — Whisper fallback", exc_info=True)
            return None

    def _transcribe_blocking(self, audio: np.ndarray) -> str | None:
        wav = _encode_wav(audio)
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[
                self._types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                "Transcribe this audio verbatim in its original language "
                "(English, Hindi, Telugu, or a mix). Output ONLY the transcript "
                "text — no quotes, labels, or commentary.",
            ],
        )
        text = (getattr(resp, "text", "") or "").strip()
        return text or None
