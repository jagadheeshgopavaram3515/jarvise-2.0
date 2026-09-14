"""
STTService — faster-whisper consumer thread.

Consumes Utterance objects from stt_in_q, transcribes them with
faster-whisper (multilingual: Telugu / Hindi / English / code-switched),
and pushes Transcript objects to transcript_q.

faster-whisper is chosen over Vosk/recognize_google because:
  * Genuine multilingual + code-switch handling (your "Fuel entha undi?" case)
  * Runs locally — no network round-trip (kills the recognize_google latency)
  * int8 on the GTX 1650 transcribes a short utterance in well under 500 ms
"""
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
from assistant.stt import cloud as cloud_mod

log = get("stt")
_SUPPORTED_LANGS = {"en", "hi", "te"}
_SUPPORTED_TEXT = re.compile(r"[A-Za-z\u0900-\u097F\u0C00-\u0C7F]")

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
    language code). Telugu/Devanagari → te/hi so the reply + voice match."""
    for ch in text:
        o = ord(ch)
        if 0x0C00 <= o <= 0x0C7F:
            return "te"
        if 0x0900 <= o <= 0x097F:
            return "hi"
    return "en"


def _has_real_speech(audio: np.ndarray) -> bool:
    """True if the segment plausibly contains speech (else skip whisper)."""
    if audio is None or audio.size == 0:
        return False
    s = _get_silero()
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
            in_script = sum(1 for c in letters if _SUPPORTED_TEXT.match(c))
            if in_script / len(letters) < 0.5:
                log.warning("[LANG FILTER] out-of-script hallucination dropped lang=%s text=%r",
                            lang, text[:60])
                return False
        if prob < config.STT_MIN_LANG_PROB:
            log.warning("[LANG CONFIDENCE] dropping very-low-confidence lang=%s prob=%.2f text=%r",
                        lang, prob, text[:60])
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
    def __init__(self, bus: Bus):
        super().__init__(name="STT", daemon=True)
        self.bus = bus
        self.engine: FasterWhisperSTT | None = None
        self.profiler = STTProfiler()
        self._profiler_lock = threading.Lock()
        self._cloud = None                 # Phase 1: Gemini-audio STT (conversational)
        # Fix 4: two single-worker lanes — partials (tiny) and finals (base) run
        # on different models/threads, so a slow final never blocks live partials.
        self._partial_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="STT-Partial")
        self._final_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="STT-Final")
        self._partial_lock = threading.Lock()
        self._latest_partial = None        # keep-newest coalescing slot
        self._partial_scheduled = False

    def run(self):
        if config.FLAGS.demo_mode:
            return
        log.info("Loading faster-whisper model=%s ...", config.STT_MODEL_SIZE)
        try:
            self.engine = FasterWhisperSTT()
            log.info("STT ready on device=%s compute=%s", self.engine.device, self.engine.compute_type)
            self.bus.stt_ready.set()
            self.bus.post_gui("notification", f"STT ready ({self.engine.device})")
        except Exception as e:
            log.exception("STT init failed")
            self.bus.post_gui("notification", f"STT init failed: {e}")
            return

        # Phase 1: optional Gemini-audio STT for conversational finals.
        if config.HYBRID_STT:
            try:
                from assistant.stt.cloud import GeminiAudioSTT
                self._cloud = GeminiAudioSTT()
                log.info("[HYBRID STT] Gemini audio enabled (model=%s)", config.HYBRID_STT_MODEL)
            except Exception:
                log.info("[HYBRID STT] init failed — Whisper-only", exc_info=True)

        while not self.bus.shutdown.is_set():
            try:
                utt = self.bus.stt_in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # A malformed item or a transient dispatch error must never kill the
            # STT thread (that would silence the assistant and force a costly
            # model reload on recovery). Per-utterance transcription is already
            # guarded inside _transcribe_and_emit; this guards the routing too.
            try:
                # Fillers carry no query — forward the signal cheaply (no transcription).
                if getattr(utt, "kind", "speech") == "filler":
                    self._forward_filler(utt)
                    continue

                # Fix 4: hand off to the right lane and keep consuming immediately so a
                # multi-second final transcription cannot stall live partials.
                if utt.is_partial:
                    with self._partial_lock:
                        self._latest_partial = utt            # keep only the newest
                        need_submit = not self._partial_scheduled
                        self._partial_scheduled = True
                    if need_submit:
                        self._partial_executor.submit(self._process_latest_partial)
                else:
                    self._final_executor.submit(self._transcribe_and_emit, utt)
            except Exception:
                log.exception("STT dispatch error — skipped, thread stays alive")

        self._partial_executor.shutdown(wait=False)
        self._final_executor.shutdown(wait=False)

    def _forward_filler(self, utt):
        log.info("[FILLER] passthrough id=%s during_speech=%s",
                 utt.utterance_id, utt.during_speech)
        self.bus.transcript_q.put(
            Transcript(text="", during_speech=utt.during_speech,
                       utterance_id=utt.utterance_id, kind="filler")
        )

    def _process_latest_partial(self):
        """Partial lane: always transcribe the NEWEST queued partial, dropping
        any that were superseded while a previous one was decoding."""
        with self._partial_lock:
            utt = self._latest_partial
            self._latest_partial = None
            self._partial_scheduled = False
        if utt is not None:
            self._transcribe_and_emit(utt)

    def _transcribe_and_emit(self, utt):
        """Transcribe one utterance and post the Transcript. Runs on a lane
        executor (partial or final). Safe to run two of these concurrently:
        partials and finals use different model objects; shared profiler is
        locked and the transcript queue is thread-safe."""
        t0 = time.perf_counter()
        # Drop finals that languished in a backlog — answering 25s-old speech is
        # worse than skipping it. Stale items are discarded fast (no transcribe),
        # draining the queue down to the newest utterance.
        if (not utt.is_partial and config.STT_MAX_STALE_S > 0
                and utt.perf is not None):
            age = t0 - getattr(utt.perf, "stt_queued", t0)
            if age > config.STT_MAX_STALE_S:
                log.info("[STT STALE] dropping final queued %.1fs ago (backlog)", age)
                utt.perf.trace("stt_dropped_stale", age=f"{age:.1f}s")
                return
        if utt.perf is not None:
            utt.perf.stamp("stt_start", t0)
            utt.perf.trace("stt_start", audio_sec=f"{len(utt.audio) / config.STT_SAMPLE_RATE:.2f}",
                           partial=utt.is_partial, seq=utt.sequence)

        # ---- Cloud-FIRST for finals (accurate Telugu/Hindi/English) ----------
        # Gemini audio handles multilingual far better than the local base model,
        # so for real-speech finals we use it as the primary and skip the slow,
        # Telugu-blind Whisper pass entirely. Whisper remains the fallback.
        if not utt.is_partial and self._cloud is not None:
            if config.STT_VAD_PREFILTER and not _has_real_speech(utt.audio):
                if utt.perf is not None:
                    utt.perf.trace("transcript_dropped", reason="vad_prefilter_no_speech")
                return
            cloud_text = self._cloud.transcribe(utt.audio)
            if cloud_text:
                lang = _lang_from_script(cloud_text)
                log.info("[HYBRID STT] cloud lang=%s text=%r", lang, cloud_text)
                if utt.perf is not None:
                    utt.perf.stamp("stt_end")
                    utt.perf.stt = time.perf_counter() - t0
                    utt.perf.trace("stt_done", took=f"{utt.perf.stt:.2f}s", lang=lang,
                                   prob="1.00", text=repr(cloud_text[:80]))
                self.bus.post_gui("subtitle", cloud_text, role="user")
                self._post_transcript(utt, cloud_text, lang, 1.0)
                return
            log.info("[HYBRID STT] cloud empty/failed — Whisper fallback")

        # ---- Whisper (local) path: partials, or cloud disabled/failed --------
        try:
            queue_wait = max(0.0, t0 - getattr(utt.perf, "stt_queued", t0)) if utt.perf is not None else 0.0
            result = self.engine.transcribe(
                utt.audio, queue_wait=queue_wait, is_partial=utt.is_partial,
            )
        except Exception as e:
            log.exception("STT transcribe error")
            self.bus.post_gui("notification", f"STT error: {e}")
            return
        if result is None:   # Fix 3: VAD pre-filter skipped a silent/noise segment
            if utt.perf is not None:
                utt.perf.trace("transcript_dropped", reason="vad_prefilter_no_speech")
            return
        text, lang, lang_prob = result.text, result.language, result.language_probability
        dt = result.transcribe_time
        with self._profiler_lock:
            self.profiler.add(result)
        if utt.perf is not None:
            utt.perf.stamp("stt_end")
            utt.perf.trace("stt_done", took=f"{dt:.2f}s", lang=lang, prob=f"{lang_prob:.2f}", text=repr(text[:80]))
        log.info("STT %.2fs lang=%s prob=%.2f text=%r", dt, lang, lang_prob, text)
        if utt.perf is not None:
            utt.perf.stt = dt
        if not self._is_usable_transcript(text, lang, lang_prob):
            if utt.perf is not None:
                utt.perf.trace("transcript_dropped", reason="empty_or_short_noise")
            return

        if not utt.is_partial:
            self.bus.post_gui("subtitle", text, role="user")
        else:
            log.info("[STT PARTIAL] id=%s seq=%d text=%r", utt.utterance_id, utt.sequence, text)
        self._post_transcript(utt, text, lang, lang_prob)

    def _post_transcript(self, utt, text, lang, lang_prob):
        self.bus.transcript_q.put(
            Transcript(text=text, language=lang, language_probability=lang_prob,
                       during_speech=utt.during_speech,
                       perf=utt.perf,
                       utterance_id=utt.utterance_id,
                       is_partial=utt.is_partial,
                       sequence=utt.sequence,
                       kind=getattr(utt, "kind", "speech"))
        )

    @staticmethod
    def _is_usable_transcript(text: str, lang: str | None, lang_prob: float) -> bool:
        stripped = text.strip()
        if not stripped:
            log.info("[STT DROP] empty transcript")
            return False
        if lang_prob < 0.20 and len(stripped) < 3:
            log.info("[STT DROP] low confidence short noise lang=%s prob=%.2f text=%r",
                     lang, lang_prob, stripped)
            return False
        return True

    def transcribe_blocking(self, audio):
        """Used by demo/console path; lazy-inits the engine."""
        if self.engine is None:
            self.engine = FasterWhisperSTT()
        result = self.engine.transcribe(audio)
        if result is None:
            return "", None
        return result.text, result.language
