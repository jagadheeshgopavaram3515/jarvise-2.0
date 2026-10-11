"""Gemini-only transcription with one outstanding SDK request and no retries."""
from __future__ import annotations

import io
import json
import re
import threading
import time
import uuid
import wave
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeout

import numpy as np

from assistant import config
from assistant.core.log import get

log = get("stt.cloud")

# Retained command heuristic for compatibility; it does not select an STT provider.
_COMMAND_RE = re.compile(
    r"\b(time|date|open|close|search|play|volume|mute|unmute|sleep|wake|"
    r"remind|reminder|timer|alarm|stop|cancel|pause|resume|quit|exit)\b",
    re.I,
)


def looks_like_command(text: str) -> bool:
    """True for short deterministic commands (all speech still uses Gemini)."""
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
    """Return a transcript or None. A timeout does not cancel a synchronous call.

    The timed-out request retains its slot until the SDK returns. Subsequent
    callers return None promptly; completed expired results are never reused.
    """

    def __init__(self):
        started = time.monotonic()
        try:
            if not config.GOOGLE_API_KEY:
                raise ValueError("missing Google credential")
            if not config.HYBRID_STT_MODEL.strip():
                raise ValueError("missing STT model")
            from google import genai
            from google.genai import types
            self._types = types
            # SDK timeout is in milliseconds. Disable its automatic retries:
            # one accepted final utterance must produce only one API attempt.
            self._client = genai.Client(
                api_key=config.GOOGLE_API_KEY,
                http_options=types.HttpOptions(
                    timeout=max(1, int(config.HYBRID_STT_TIMEOUT_S * 1000)),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        except Exception as exc:
            log.error("STT_INIT category=initialization_error provider=gemini model=%s timeout_s=%.3f "
                      "elapsed_s=%.4f in_flight=false error_type=%s credential_present=%s",
                      config.HYBRID_STT_MODEL, config.HYBRID_STT_TIMEOUT_S,
                      time.monotonic() - started, type(exc).__name__, bool(config.GOOGLE_API_KEY))
            raise RuntimeError("Gemini STT initialization failed; check STT_INIT log") from None
        self.model = config.HYBRID_STT_MODEL
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="CloudSTT")
        self._request_lock = threading.Lock()
        self._inflight: Future | None = None
        self._inflight_id = ""
        self._closed = False
        self._event("ready", "ready", "", time.monotonic(), False)

    def _event(self, event, category, request_id, started, in_flight, **extra):
        # Never include exception messages, credentials, audio or transcript text.
        record = dict(event=event, category=category, utterance_id=request_id,
                      provider="gemini", model=self.model,
                      timeout_s=config.HYBRID_STT_TIMEOUT_S,
                      timeout_source=config.HYBRID_STT_TIMEOUT_SOURCE,
                      elapsed_s=round(time.monotonic() - started, 4),
                      in_flight=bool(in_flight), **extra)
        level = log.info if category in {"ready", "success", "submitted", "completed"} else log.warning
        level("STT_EVENT %s", json.dumps(record, ensure_ascii=True))

    @staticmethod
    def _error_category(exc):
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code in (401, 403, "401", "403"):
            return "authentication_error"
        if code in (429, "429"):
            return "rate_limit"
        if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
            return "network_timeout"
        if code is not None:
            return "api_error"
        if isinstance(exc, ConnectionError) or type(exc).__name__ in {
            "ConnectError", "ReadError", "WriteError", "RemoteProtocolError", "NetworkError"
        }:
            return "network_error"
        return "request_error"

    def transcribe(self, audio: np.ndarray, *, utterance_id: str = "") -> str | None:
        started = time.monotonic()
        request_id = utterance_id or uuid.uuid4().hex[:8]
        if self._types is None:
            self._event("result", "initialization_error", request_id, started, False)
            return None
        expired = threading.Event()
        fut = None
        try:
            with self._request_lock:
                if self._closed:
                    self._event("result", "closed", request_id, started, False)
                    return None
                # A caller timeout does not cancel the synchronous SDK request.
                # Keep its slot until it finishes; never queue behind it.
                if self._inflight is not None:
                    if not self._inflight.done():
                        self._event("result", "occupied", request_id, started, True,
                                    active_utterance_id=self._inflight_id)
                        return None
                    self._inflight = None
                self._event("request_start", "submitted", request_id, started, False)
                fut = self._executor.submit(self._transcribe_blocking, audio, request_id, started)
                self._inflight = fut
                self._inflight_id = request_id
            # Completion is observable even after the caller timed out. It has
            # no delivery callback and cannot post a late transcript to the bus.
            def finished(done):
                exc = done.exception() if not done.cancelled() else None
                category = self._error_category(exc) if exc else "completed"
                self._event("request_end", category, request_id, started, False,
                            expired=expired.is_set(), error_type=type(exc).__name__ if exc else None)
            fut.add_done_callback(finished)
            # Only this call's future can supply this call's result. Completed
            # expired futures are discarded on the next admission, never reused.
            text = fut.result(timeout=config.HYBRID_STT_TIMEOUT_S)
            self._event("result", "success" if text else "empty_response",
                        request_id, started, False)
            return text
        except FuturesTimeout:
            # A TimeoutError raised by the SDK is different from our caller's
            # bounded wait expiring. Neither condition cancels a running call.
            sdk_timeout = fut is not None and fut.done() and isinstance(fut.exception(), TimeoutError)
            if not sdk_timeout:
                expired.set()
            self._event("result", "network_timeout" if sdk_timeout else "timeout", request_id, started,
                        fut is not None and not fut.done())
            return None
        except Exception as exc:
            self._event("result", "submission_failure" if fut is None else self._error_category(exc),
                        request_id, started, fut is not None and not fut.done(),
                        error_type=type(exc).__name__)
            return None

    def close(self):
        with self._request_lock:
            self._closed = True
        # Never wait indefinitely for a stuck SDK worker or claim cancellation.
        self._executor.shutdown(wait=False)

    @property
    def in_flight(self):
        with self._request_lock:
            return self._inflight is not None and not self._inflight.done()

    def _transcribe_blocking(self, audio: np.ndarray, request_id="", started=None) -> str | None:
        wav = _encode_wav(audio)
        self._event("sdk_start", "submitted", request_id, started or time.monotonic(), True)
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[
                self._types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                "Transcribe this audio verbatim in its original language "
                "(English, Hindi, Telugu, or a mix). If there is no clearly "
                "intelligible human speech, output an empty string; never infer "
                "words from silence, room noise, music, or background sounds. "
                "Output ONLY the transcript "
                "text — no quotes, labels, or commentary.",
            ],
        )
        text = (getattr(resp, "text", "") or "").strip()
        return text or None
