"""Gemini-only STT service. Legacy offline benchmark types remain inert below."""
from __future__ import annotations

import glob
import os
import queue
import statistics
import site
import sys
import threading
import time
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from assistant import config
from assistant.core.events import Bus, Transcript
from assistant.core.log import get
from assistant.stt.cloud import GeminiAudioSTT
from collections import Counter, OrderedDict, deque

log = get("stt")
_SUPPORTED_LANGS = {"en", "hi", "te", "ta"}
_SUPPORTED_TEXT = re.compile(r"[A-Za-z\u0900-\u097F\u0C00-\u0C7F\u0B80-\u0BFF]")

# ----------------------------------------------------------------- VAD prefilter
# Fix 3: a cheap speech check BEFORE whisper, so silence/noise never costs a
# multi-second decode. Uses Silero VAD if installed; otherwise an RMS energy
# gate (no extra dependency) \u2014 so the empty-audio skip works either way.
_silero = None
_silero_failed = False


def _get_silero():
    global _silero, _silero_failed
    if _silero is not None or _silero_failed:
        return _silero
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        _silero = (load_silero_vad(), get_speech_timestamps)
        log.info("[VAD] Silero pre-filter loaded")
    except Exception:
        _silero_failed = True
        log.info("[VAD] Silero unavailable; using RMS energy pre-filter")
    return _silero


def _lang_from_script(text: str) -> str:
    """Infer language code from the script of cloud-STT text (Gemini returns no
    language code). Telugu/Devanagari/Tamil -> te/hi/ta so the reply + voice match."""
    for ch in text:
        o = ord(ch)
        if 0x0C00 <= o <= 0x0C7F:
            return "te"
        if 0x0900 <= o <= 0x097F:
            return "hi"
        if 0x0B80 <= o <= 0x0BFF:
            return "ta"
    return "en"


def _has_real_speech(audio: np.ndarray, *, energy_only: bool = False) -> bool:
    """True if the segment plausibly contains speech (else skip whisper)."""
    if audio is None or audio.size == 0:
        return False
    s = None if energy_only else _get_silero()
    if s is not None:
        model, get_speech_timestamps = s
        try:
            ts = get_speech_timestamps(audio, model, threshold=0.5,
                                       min_speech_duration_ms=200)
            return len(ts) > 0
        except Exception:
            log.debug("[VAD] Silero check failed; falling back to RMS", exc_info=True)
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    return rms > config.STT_PREFILTER_RMS


def _add_cuda_dll_dirs():
    """Make CTranslate2 find cuBLAS/cuDNN DLLs from the nvidia-* pip wheels.

    On Windows the DLLs ship under site-packages/nvidia/<lib>/bin and aren't on
    PATH by default, so ctranslate2 fails to load CUDA. We register those dirs.
    No-op on non-Windows or if the packages aren't installed.
    """
    if not sys.platform.startswith("win"):
        return
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    for root in roots:
        for dll_dir in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _pick_device() -> tuple[str, str]:
    """Return (device, compute_type) without requiring torch."""
    device = config.STT_DEVICE
    if device == "auto":
        _add_cuda_dll_dirs()
        device = "cuda" if _cuda_available() else "cpu"
    elif device == "cuda":
        _add_cuda_dll_dirs()
    if config.STT_COMPUTE_TYPE:
        return device, config.STT_COMPUTE_TYPE
    return device, (config.STT_GPU_COMPUTE_TYPE if device == "cuda"
                    else config.STT_CPU_COMPUTE_TYPE)


@dataclass
class STTResult:
    text: str
    language: str | None
    language_probability: float
    audio_sec: float
    queue_wait: float
    transcribe_time: float
    decode_time: float
    language_detect_time: float
    total_stt: float
    model_size: str
    device: str
    compute_type: str


class FasterWhisperSTT:
    """Legacy offline benchmark API; STTService never constructs this class.

    Retained because scratch/stt_benchmark.py still imports it explicitly.
    Local-model imports remain lazy and are unreachable from the voice pipeline.
    """
    def __init__(self):
        from faster_whisper import WhisperModel
        device, compute_type = _pick_device()
        self.device = device
        self.compute_type = compute_type
        self._models = {}
        self._lock = threading.Lock()
        # Fix 2: language-detection cache (finals only; single final worker).
        self._detected_lang: str | None = None
        self._lang_hits = 0
        self._lang_lock = threading.Lock()
        self._load_model(config.STT_MODEL_SIZE)
        if config.STT_FAST_MODEL_SIZE != config.STT_MODEL_SIZE:
            self._load_model(config.STT_FAST_MODEL_SIZE)
        self._warmup()

    def _model_kwargs(self) -> dict:
        """faster-whisper constructor args, tuned for CPU (Fix 6)."""
        kwargs = {"device": self.device, "compute_type": self.compute_type}
        if self.device == "cpu":
            if config.STT_CPU_THREADS:
                kwargs["cpu_threads"] = config.STT_CPU_THREADS   # Pi 5: 4 cores
            if config.STT_NUM_WORKERS:
                kwargs["num_workers"] = config.STT_NUM_WORKERS
        return kwargs

    def _load_model(self, model_size: str):
        from faster_whisper import WhisperModel
        with self._lock:
            if model_size in self._models:
                return self._models[model_size]
            kwargs = self._model_kwargs()
            t0 = time.perf_counter()
            try:
                model = WhisperModel(model_size, **kwargs)
            except Exception:
                if self.device != "cuda":
                    raise
                log.exception("CUDA STT model load failed; falling back to CPU int8")
                self.device = "cpu"
                self.compute_type = config.STT_CPU_COMPUTE_TYPE
                model = WhisperModel(model_size, **self._model_kwargs())
            self._models[model_size] = model
            log.info("STT model loaded size=%s device=%s compute=%s load=%.2fs",
                     model_size, self.device, self.compute_type, time.perf_counter() - t0)
            return model

    def _warmup(self):
        """Run one tiny decode so the first real user utterance is not cold."""
        try:
            import numpy as np
            samples = max(1, int(config.STT_SAMPLE_RATE * config.STT_WARMUP_SECONDS))
            audio = np.zeros(samples, dtype=np.float32)
            for model_size, model in self._models.items():
                segments, _ = model.transcribe(
                    audio,
                    language=config.STT_LANGUAGE,
                    beam_size=1,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )
                list(segments)
                log.info("STT warmup complete model=%s", model_size)
        except Exception:
            log.debug("STT warmup skipped", exc_info=True)
        if config.STT_LANGUAGE is None:
            log.info("[STT] language=auto")
        else:
            log.info("[STT] language=%s (forced)", config.STT_LANGUAGE)

    def _transcribe_once(self, audio, model_size: str, queue_wait: float,
                         is_partial: bool, language: str | None) -> STTResult:
        model = self._load_model(model_size)
        audio_sec = len(audio) / config.STT_SAMPLE_RATE
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            audio,
            language=language,                # None = auto-detect (finals only)
            beam_size=config.STT_BEAM_SIZE,
            vad_filter=False,                 # webrtcvad already endpointed; avoid double-VAD trim
            condition_on_previous_text=False, # lower latency, less drift
        )
        t_after_setup = time.perf_counter()
        text = " ".join(s.text.strip() for s in segments).strip()
        t_end = time.perf_counter()
        return STTResult(
            text=text,
            language=getattr(info, "language", None),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            audio_sec=audio_sec,
            queue_wait=queue_wait,
            transcribe_time=t_end - t0,
            decode_time=t_end - t_after_setup,
            language_detect_time=t_after_setup - t0,
            total_stt=t_end - t0 + queue_wait,
            model_size=model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def _language_for(self, is_partial: bool) -> str | None:
        """Decide the decode language. Forced config wins; partials assume EN
        (Fix 2 prior); finals reuse a cached confident detection for a while
        (Fix 2 cache) before running real auto-detection again."""
        forced = config.STT_LANGUAGE
        if forced is not None:
            return forced
        if is_partial:
            return (config.STT_PARTIAL_LANGUAGE
                    if config.STT_LANGUAGE_DETECT_ONLY_ON_FINAL else None)
        with self._lang_lock:
            if self._detected_lang and self._lang_hits < config.STT_LANG_CACHE_REUSE:
                self._lang_hits += 1
                return self._detected_lang          # reuse → no detect cost
        return None                                  # auto-detect this final

    def _update_lang_cache(self, used_language: str | None, result: "STTResult"):
        # Only learn from a REAL detection (used_language was None). When we
        # forced/reused a language, whisper reports prob≈1.0 — ignore it so the
        # reuse counter actually expires.
        if used_language is not None:
            return
        # Only learn an EXPECTED language — never cache a noise hallucination
        # (e.g. 'ru'/'ur') and then reuse it for the next 8 utterances.
        if (result.language and result.language_probability >= config.STT_LANG_CACHE_MIN_PROB
                and result.language in config.STT_EXPECTED_LANGS):
            with self._lang_lock:
                self._detected_lang = result.language
                self._lang_hits = 0
                log.info("[LANG] detected=%s prob=%.2f (cache reset)",
                         result.language, result.language_probability)

    def _language_acceptable(self, result: "STTResult") -> bool:
        """Reject noise hallucinations WITHOUT eating good speech.

        language_probability is NOT a quality signal — clear short English often
        scores ~0.45-0.50. So we judge by SCRIPT (the reliable tell for
        Cyrillic/Arabic/CJK hallucinations) and only drop on a *very* low
        confidence floor for in-script text that's likely pure noise.
        """
        text, lang, prob = result.text or "", result.language, result.language_probability
        letters = [c for c in text if c.isalpha()]
        if letters:
            has_supported = any(_SUPPORTED_TEXT.match(c) for c in letters)
            if not has_supported:
                log.warning("[LANG FILTER] out-of-script hallucination dropped lang=%s text=%r",
                            lang, text[:60])
                return False
        return True

    def transcribe(self, audio, *, queue_wait: float = 0.0,
                   is_partial: bool = False):
        # Fix 3: skip whisper entirely on silence/noise (no multi-second decode
        # that returns empty text). Finals only — partials are already cheap.
        if (not is_partial and config.STT_VAD_PREFILTER
                and not _has_real_speech(audio)):
            log.info("[STT SKIP] VAD pre-filter: no speech in %.2fs audio",
                     len(audio) / config.STT_SAMPLE_RATE)
            return None

        fast_model = config.STT_FAST_MODEL_SIZE
        final_model = config.STT_MODEL_SIZE
        model_size = fast_model if is_partial else final_model
        language = self._language_for(is_partial)
        log.info("STT model selected=%s partial=%s lang=%s", model_size, is_partial,
                 language or "auto")
        result = self._transcribe_once(audio, model_size, queue_wait, is_partial, language)
        self._update_lang_cache(language, result)

        # Adaptive model switching: if the fast model sees non-English or weak
        # confidence on a final utterance, retry on the multilingual final model.
        if (not is_partial and model_size != final_model
                and (result.language not in {"en"} or result.language_probability < 0.65)):
            log.info("STT model switched to %s", final_model)
            language = self._language_for(is_partial=False)
            result = self._transcribe_once(audio, final_model, queue_wait, is_partial, language)
            self._update_lang_cache(language, result)

        # Fix 1: drop low-confidence / impossible-language finals (noise
        # hallucinations into Russian/Urdu/etc. that were generating bad LLM
        # calls). Partials are forced-EN hints, so they're exempt.
        if not is_partial and not self._language_acceptable(result):
            return None
        return result


class STTProfiler:
    def __init__(self, window: int = config.STT_PROFILE_WINDOW):
        self.window = max(1, window)
        self.rows: list[STTResult] = []

    def add(self, result: STTResult):
        self.rows.append(result)
        if len(self.rows) % self.window != 0:
            return
        recent = self.rows[-self.window:]

        def avg(attr: str) -> float:
            return statistics.mean(getattr(r, attr) for r in recent)

        log.info(
            "[PERF] STT_AVG n=%d audio_sec=%.2fs queue_wait=%.2fs "
            "transcribe_time=%.2fs decode_time=%.2fs language_detect_time=%.2fs "
            "total_stt=%.2fs",
            self.window,
            avg("audio_sec"),
            avg("queue_wait"),
            avg("transcribe_time"),
            avg("decode_time"),
            avg("language_detect_time"),
            avg("total_stt"),
        )


class STTService(threading.Thread):
    """Bounded FIFO of finals, with only one transcription worker submitted.

    Whisper partials are retired. Finals retain the same Transcript contract,
    including during_speech, so dispatcher keyword interruption still works.
    """

    def __init__(self, bus: Bus):
        super().__init__(name="STT", daemon=True)
        self.bus = bus
        self._cloud = None
        self.initialized = threading.Event()
        self._init_lock = threading.Lock()
        self._final_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="STT-Final")
        self._final_slot = threading.Lock()
        self._seen_lock = threading.Lock()
        self._seen_ids = OrderedDict()
        self._pending = deque()
        self._active = None
        self.diagnostic_counts = Counter()
        self._diagnostic_lock = threading.Lock()
        self.bus.stt_ready.clear()

    def _initialize(self):
        with self._init_lock:
            if self.initialized.is_set():
                return
            try:
                self._cloud = GeminiAudioSTT()
                self.bus.stt_ready.set()
                self.bus.post_gui("notification", "STT ready (Gemini)")
                log.info("STT_READY provider=gemini model=%s timeout_s=%.3f partials=false",
                         config.HYBRID_STT_MODEL, config.HYBRID_STT_TIMEOUT_S)
            except Exception as exc:
                log.error("STT_INIT category=initialization_error error_type=%s", type(exc).__name__)
                self.bus.post_gui("notification", "Gemini STT initialization failed; check STT_INIT log")
            finally:
                self.initialized.set()

    def run(self):
        if config.FLAGS.demo_mode:
            self.initialized.set()
            return
        self._initialize()
        try:
            while not self.bus.shutdown.is_set():
                self._advance_finals()
                try:
                    utt = self.bus.stt_in_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if getattr(utt, "kind", "speech") == "filler":
                        self._forward_filler(utt)
                    elif utt.is_partial:
                        self._diagnostic(utt, "partial_disabled")
                    else:
                        self._enqueue_final(utt)
                except Exception as exc:
                    log.error("STT_DISPATCH category=dispatch_error error_type=%s", type(exc).__name__)
        finally:
            while self._pending:
                self._diagnostic(self._pending.popleft()[0], "shutdown")
            while True:
                try:
                    self._diagnostic(self.bus.stt_in_q.get_nowait(), "shutdown")
                except queue.Empty:
                    break
            self._final_executor.shutdown(wait=False)
            if self._cloud is not None:
                self._cloud.close()

    def _enqueue_final(self, utt):
        """Called only by the consumer thread; never blocks microphone capture."""
        admitted = time.perf_counter()
        if utt.interrupt_epoch is None:
            utt.interrupt_epoch = self.bus.interrupt_epoch
        queued = getattr(utt.perf, "stt_queued", 0.0) if utt.perf is not None else 0.0
        captured = queued or admitted
        # Filter before admission so noise cannot consume the short FIFO.
        if self._cancelled_or_expired(utt, captured):
            return
        if config.STT_VAD_PREFILTER and not self._prefilter(utt):
            return
        if len(self._pending) >= config.STT_FINAL_QUEUE_CAPACITY:
            self._diagnostic(utt, "queue_overflow", queue_depth=len(self._pending))
            return  # keep older accepted speech; explicitly reject newest
        self._pending.append((utt, captured))
        self._diagnostic(utt, "queued", queue_depth=len(self._pending))
        self._advance_finals()

    def _cancelled_or_expired(self, utt, captured):
        if self.bus.shutdown.is_set():
            self._diagnostic(utt, "shutdown")
            return True
        if utt.interrupt_epoch != self.bus.interrupt_epoch:
            self._diagnostic(utt, "cancelled")
            return True
        age = max(0.0, time.perf_counter() - captured)
        if age > config.STT_FINAL_QUEUE_MAX_AGE_S:
            self._diagnostic(utt, "queue_expired", elapsed=age)
            return True
        return False

    def _advance_finals(self):
        # Remove invalid pending work even when an SDK request remains stuck.
        retained = deque()
        while self._pending:
            entry = self._pending.popleft()
            if not self._cancelled_or_expired(*entry):
                retained.append(entry)
        self._pending = retained
        if self._active is not None:
            if not self._active.done():
                return
            self._active = None
        # Caller timeout doesn't free the SDK slot. Wait here without submitting
        # another request or accumulating work in either executor's own queue.
        if self._cloud is not None and self._cloud.in_flight:
            return
        if not self._pending or not self._final_slot.acquire(blocking=False):
            return
        utt, captured = self._pending.popleft()
        try:
            self._active = self._final_executor.submit(self._process_final, utt, captured)
        except Exception as exc:
            self._final_slot.release()
            self._diagnostic(utt, "submission_failure", error_type=type(exc).__name__)

    def _prefilter(self, utt):
        audio = utt.audio
        if _has_real_speech(audio, energy_only=True):
            return True
        samples = np.asarray(audio, dtype=np.float32) if audio is not None else np.array([])
        rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        # Observe silence dilution without trimming audio or changing admission.
        # Frame energy is not a speech classifier: loud noise can also exceed it.
        frame_size = max(1, config.STT_SAMPLE_RATE * config.VAD_FRAME_MS // 1000)
        frames = [samples[start:start + frame_size]
                  for start in range(0, samples.size, frame_size)]
        frame_rms = [float(np.sqrt(np.mean(frame ** 2))) for frame in frames]
        self._diagnostic(utt, "no_speech", rms=round(rms, 6), peak=round(peak, 6),
                         max_frame_rms=round(max(frame_rms, default=0.0), 6),
                         above_threshold_ms=round(sum(frame.size for frame, value in zip(frames, frame_rms)
                                                      if value > config.STT_PREFILTER_RMS)
                                                  * 1000 / config.STT_SAMPLE_RATE, 1),
                         threshold=config.STT_PREFILTER_RMS,
                         audio_ms=round(samples.size * 1000 / config.STT_SAMPLE_RATE, 1),
                         filter_reason="empty" if not samples.size else "rms_below_threshold")
        return False

    def _process_final(self, utt, captured=None):
        try:
            if captured is not None and self._cancelled_or_expired(utt, captured):
                return
            self._transcribe_and_emit(utt)
        finally:
            self._final_slot.release()

    def _diagnostic(self, utt, category, *, elapsed=0.0, in_flight=False, **extra):
        with self._diagnostic_lock:
            self.diagnostic_counts[category] += 1
        log.info("STT_DELIVERY utterance_id=%s provider=gemini model=%s timeout_s=%.3f "
                 "category=%s elapsed_s=%.4f in_flight=%s %s",
                 utt.utterance_id, config.HYBRID_STT_MODEL, config.HYBRID_STT_TIMEOUT_S,
                 category, elapsed, in_flight, extra)
        if utt.perf is not None:
            utt.perf.trace("stt_delivered" if category == "delivered" else
                           "stt_queued" if category == "queued" else "transcript_dropped",
                           utterance_id=utt.utterance_id, category=category)

    def _forward_filler(self, utt):
        self.bus.transcript_q.put(
            Transcript(text="", during_speech=utt.during_speech,
                       utterance_id=utt.utterance_id, kind="filler")
        )

    def _transcribe_and_emit(self, utt):
        t0 = time.perf_counter()
        epoch = self.bus.interrupt_epoch if utt.interrupt_epoch is None else utt.interrupt_epoch
        utt.interrupt_epoch = epoch
        if epoch != self.bus.interrupt_epoch:
            self._diagnostic(utt, "cancelled")
            return
        if utt.is_partial:
            self._diagnostic(utt, "partial_disabled")
            return
        if utt.utterance_id:
            with self._seen_lock:
                if utt.utterance_id in self._seen_ids:
                    self._diagnostic(utt, "duplicate")
                    return
                self._seen_ids[utt.utterance_id] = None
                if len(self._seen_ids) > 256:
                    self._seen_ids.popitem(last=False)
        if self.bus.shutdown.is_set():
            self._diagnostic(utt, "shutdown")
            return
        queued = getattr(utt.perf, "stt_queued", 0.0) if utt.perf is not None else 0.0
        age = max(0.0, t0 - queued) if queued else 0.0
        if config.STT_MAX_STALE_S > 0 and age > config.STT_MAX_STALE_S:
            self._diagnostic(utt, "stale", elapsed=age)
            return
        if utt.perf is not None:
            utt.perf.stamp("stt_start", t0)
            utt.perf.trace("stt_start", provider="gemini", utterance_id=utt.utterance_id,
                           audio_sec=f"{len(utt.audio) / config.STT_SAMPLE_RATE:.2f}", partial=False)
        try:
            if self._cloud is None:
                self._diagnostic(utt, "unavailable")
                return
            # Retain the existing lightweight cloud energy gate. Never load
            # Silero or a local transcription model on the active path.
            if config.STT_VAD_PREFILTER and not self._prefilter(utt):
                return
            text = self._cloud.transcribe(utt.audio, utterance_id=utt.utterance_id)
            dt = time.perf_counter() - t0
            if utt.perf is not None:
                utt.perf.stamp("stt_end")
                utt.perf.stt = dt
            if not text or not text.strip():
                self._diagnostic(utt, "no_result", elapsed=dt, in_flight=self._cloud.in_flight)
                return
            if self.bus.shutdown.is_set():
                self._diagnostic(utt, "shutdown", elapsed=dt)
                return
            if epoch != self.bus.interrupt_epoch:
                self._diagnostic(utt, "cancelled", elapsed=dt)
                return
            text = text.strip()
            lang = _lang_from_script(text)
            self.bus.post_gui("subtitle", text, role="user")
            self._post_transcript(utt, text, lang, 1.0)
            self._diagnostic(utt, "delivered", elapsed=dt)
        except Exception as exc:
            self._diagnostic(utt, "request_error", elapsed=time.perf_counter() - t0,
                             error_type=type(exc).__name__)

    def _post_transcript(self, utt, text, lang, lang_prob):
        self.bus.transcript_q.put(
            Transcript(text=text, language=lang, language_probability=lang_prob,
                       during_speech=utt.during_speech, perf=utt.perf,
                       utterance_id=utt.utterance_id, is_partial=utt.is_partial,
                        sequence=utt.sequence, kind=getattr(utt, "kind", "speech"),
                        interrupt_epoch=utt.interrupt_epoch)
        )

    @staticmethod
    def _is_usable_transcript(text, lang=None, lang_prob=1.0):
        # Compatibility helper: don't reject short commands or other languages.
        return bool(text and text.strip())

    def transcribe_blocking(self, audio):
        """Compatibility API: return (text, language), or ("", None)."""
        self._initialize()
        if self._cloud is None:
            return "", None
        text = self._cloud.transcribe(audio)
        return (text, _lang_from_script(text)) if text else ("", None)
