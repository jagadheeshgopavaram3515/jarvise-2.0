"""
EdgeStreamingTTS — in-memory, interruptible Edge-TTS playback.

Fixes the original disk-MP3 bottleneck:
  * edge_tts streams audio chunks which we accumulate in a BytesIO (NO file
    on disk).
  * MP3 bytes are decoded to PCM in memory (miniaudio) and played through
    sounddevice in small blocks. Between blocks we poll the interrupt event,
    so playback stops within ~one block (~20 ms) on barge-in.
  * Voice is auto-selected from the reply's script (Devanagari→Hindi,
    Telugu block→Telugu, else English) so multilingual replies sound right.

Fallback chain: sounddevice+miniaudio → pygame(BytesIO). Both in-memory; no
file is ever written to disk.
"""
from __future__ import annotations

import asyncio
import io
import os
import threading
import time
import wave

try:
    import edge_tts
except Exception:  # pragma: no cover - dependency/runtime environment issue
    edge_tts = None

from assistant import config
from assistant.core.log import get

log = get("tts")

try:
    import numpy as np
    import sounddevice as sd
    import miniaudio
    _STREAMING_OK = True
except Exception:
    _STREAMING_OK = False

# Unicode block ranges for script detection.
_DEVANAGARI = range(0x0900, 0x0980)   # Hindi
_TELUGU = range(0x0C00, 0x0C80)       # Telugu
_TAMIL = range(0x0B80, 0x0C00)        # Tamil


def detect_voice(text: str) -> str:
    for ch in text:
        o = ord(ch)
        if o in _TELUGU:
            return config.VOICE_MAP["te"]
        if o in _TAMIL:
            # Fall back to English if no Tamil voice is configured, so we never
            # hand Edge a Tamil string with a non-Tamil voice (→ NoAudioReceived).
            return config.VOICE_MAP.get("ta") or config.VOICE_MAP["en"]
        if o in _DEVANAGARI:
            return config.VOICE_MAP["hi"]
    return config.VOICE_MAP["en"]  # romanised / English


def has_native_script(text: str) -> bool:
    """True if the text contains Telugu, Tamil or Devanagari (Hindi) characters —
    i.e. it needs an Edge native voice, not the English-only Piper voice."""
    return any(ord(ch) in _TELUGU or ord(ch) in _TAMIL or ord(ch) in _DEVANAGARI
               for ch in text)


def voice_style_for_emotion(emotion: str) -> tuple[str, str]:
    """Return Edge-TTS (rate, pitch) style knobs for the current mood."""
    styles = {
        "happy": ("+6%", "+4Hz"),
        "curious": ("+3%", "+2Hz"),
        "serious": ("-4%", "-2Hz"),
        "excited": ("+10%", "+6Hz"),
        "empathetic": ("-6%", "-4Hz"),
    }
    return styles.get(emotion, ("+0%", "+0Hz"))


def style_for(voice: str, emotion: str) -> tuple[str, str]:
    """Per-language tone wins over emotion (e.g. a calmer, deeper Telugu)."""
    if voice == config.VOICE_MAP.get("te") and config.TTS_TE_RATE:
        return config.TTS_TE_RATE, config.TTS_TE_PITCH
    if voice == config.VOICE_MAP.get("hi") and config.TTS_HI_RATE:
        return config.TTS_HI_RATE, config.TTS_HI_PITCH
    return voice_style_for_emotion(emotion)


class EdgeStreamingTTS:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._lock = threading.Lock()
        self._block = 1024  # frames per write — small for fast interruption
        # FIX G: optional sink (set by TTSService to bus.push_speaker_ref) that
        # receives every PCM block we play, so AEC has a far-end echo reference.
        self.ref_sink = None

    # ---- public API ----------------------------------------------------
    def synthesize_bytes(self, text: str, voice: str, emotion: str = "curious") -> bytes:
        """Run Edge-TTS and return the full MP3 payload from memory."""
        if edge_tts is None:
            raise RuntimeError("edge_tts is not installed in this Python environment")
        with self._lock:
            return self._loop.run_until_complete(self._collect(text, voice, emotion))

    async def _collect(self, text: str, voice: str, emotion: str) -> bytes:
        buf = io.BytesIO()
        rate, pitch = style_for(voice, emotion)   # per-language tone (Telugu calmer/deeper)
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    async def _collect_raw(self, text: str, voice: str, rate: str, pitch: str) -> bytes:
        buf = io.BytesIO()
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    def synthesize_backchannel(self, text: str) -> bytes:
        """Soft, slightly slower listening sound — independent of turn emotion."""
        if edge_tts is None:
            return b""
        text = text.replace("*", "").strip()
        if not text:
            return b""
        voice = detect_voice(text)
        with self._lock:
            return self._loop.run_until_complete(
                self._collect_raw(text, voice, rate="-8%", pitch="-2Hz"))

    def play_backchannel(self, mp3: bytes, stop, volume: float = 0.7,
                         on_play_start=None):
        """Play a short clip at reduced volume, stopping the instant stop() is
        True (i.e. the real response begins). Best-effort: silently no-ops if
        in-memory streaming isn't available (it never blocks the turn queue)."""
        if not mp3 or not _STREAMING_OK or stop():
            return
        try:
            decoded = miniaudio.decode(mp3)
            samples = np.frombuffer(decoded.samples, dtype=np.int16)
            if decoded.nchannels > 1:
                samples = samples.reshape(-1, decoded.nchannels)
            vol = max(0.0, min(1.0, volume))
            scaled = (samples.astype(np.float32) * vol).astype(np.int16)
            with sd.OutputStream(samplerate=decoded.sample_rate,
                                 channels=decoded.nchannels, dtype="int16") as stream:
                idx, total = 0, len(scaled)
                step = self._block * decoded.nchannels if scaled.ndim == 1 else self._block
                first = True
                while idx < total:
                    if stop():
                        break
                    if first:
                        if on_play_start:
                            on_play_start()
                        first = False
                    end = min(idx + step, total)
                    block = scaled[idx:end]
                    stream.write(block)
                    # Feed the echo reference too, so AEC suppresses our own
                    # backchannel from the still-open mic (no self-transcription).
                    if self.ref_sink is not None:
                        try:
                            self.ref_sink(block.tobytes())
                        except Exception:
                            pass
                    idx = end
        except Exception:
            pass

    def synthesize_for_playback(self, text: str, emotion: str = "curious") -> tuple[bytes, float]:
        text = text.replace("*", "").strip()
        if len(text) < 2:
            return b"", 0.0
        voice = detect_voice(text)
        t0 = time.perf_counter()
        # Synthesis must NEVER raise into the TTS thread: a service hiccup or a
        # text/voice mismatch (e.g. edge_tts NoAudioReceived) would otherwise kill
        # the thread and silence Jarvis for the rest of the session. Degrade to
        # silence (empty MP3) — the turn still completes and the conversation
        # flows on; only this one sentence is skipped.
        try:
            mp3 = self.synthesize_bytes(text, voice, emotion)
        except Exception as e:
            log.warning("TTS synth failed (voice=%s): %s — skipping this line", voice, e)
            return b"", time.perf_counter() - t0
        return mp3, time.perf_counter() - t0

    def play_synthesized(self, mp3: bytes, interrupt: threading.Event, on_play_start=None):
        if not mp3 or interrupt.is_set():
            return
        if _STREAMING_OK:
            self._play_streaming(mp3, interrupt, on_play_start)
        else:
            self._play_pygame(mp3, interrupt, on_play_start)

    def speak(self, text: str, interrupt: threading.Event, on_play_start=None) -> float:
        """Synthesize + play `text`, aborting promptly if `interrupt` is set.

        Returns the synthesis time (audio-ready) in seconds. `on_play_start` is
        called once, the instant audio playback actually begins. Both are for
        latency instrumentation and don't affect behaviour.
        """
        text = text.replace("*", "").strip()
        if len(text) < 2:
            return 0.0
        mp3, synth_dt = self.synthesize_for_playback(text)
        if not mp3 or interrupt.is_set():
            return synth_dt

        self.play_synthesized(mp3, interrupt, on_play_start)
        return synth_dt

    # ---- playback backends ---------------------------------------------
    def _play_streaming(self, mp3: bytes, interrupt: threading.Event, on_play_start=None):
        decoded = miniaudio.decode(mp3)  # int16 interleaved
        samples = np.frombuffer(decoded.samples, dtype=np.int16)
        if decoded.nchannels > 1:
            samples = samples.reshape(-1, decoded.nchannels)
        try:
            with sd.OutputStream(samplerate=decoded.sample_rate,
                                 channels=decoded.nchannels, dtype="int16") as stream:
                idx, total = 0, len(samples)
                step = self._block * decoded.nchannels if samples.ndim == 1 else self._block
                first = True
                while idx < total:
                    if interrupt.is_set():
                        break
                    if first:
                        if on_play_start:
                            on_play_start()
                        first = False
                    end = min(idx + step, total)
                    block = samples[idx:end]
                    stream.write(block)
                    # FIX G: feed the played PCM to the echo-reference buffer.
                    if self.ref_sink is not None:
                        try:
                            self.ref_sink(block.tobytes())
                        except Exception:
                            pass
                    idx = end
        except Exception:
            self._play_pygame(mp3, interrupt, on_play_start)

    def _play_pygame(self, mp3: bytes, interrupt: threading.Event, on_play_start=None):
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        try:
            sound = pygame.mixer.Sound(io.BytesIO(mp3))
            channel = sound.play()
            if on_play_start:
                on_play_start()
            while channel and channel.get_busy():
                if interrupt.is_set():
                    channel.stop()
                    break
                time.sleep(0.02)
        except Exception:
            pass


class PiperTTS(EdgeStreamingTTS):
    """Piper TTS backend (Phase 2) — calm, formal British JARVIS voice.

    Subclasses EdgeStreamingTTS so it inherits the exact same interruptible
    miniaudio/sounddevice playback (Piper returns WAV bytes, which the inherited
    player decodes just like MP3). Only synthesis is overridden. If Piper or its
    voice model isn't available, every call transparently falls back to Edge-TTS,
    so enabling this backend can never break playback.
    """

    def __init__(self):
        super().__init__()
        self.available = False
        self._voice = None
        self._load_voice()

    def _load_voice(self):
        path = config.PIPER_MODEL_PATH
        if path and not os.path.isabs(path):
            path = os.path.join(config.DATA_DIR, path)   # resolve relative to project
        if not path or not os.path.exists(path):
            log.info("Piper model not found at %r — using Edge-TTS fallback", path)
            return
        try:
            from piper import PiperVoice
            self._voice = PiperVoice.load(path)
            self.available = True
            log.info("Piper TTS loaded: %s", path)
        except Exception:
            log.info("Piper unavailable — using Edge-TTS fallback", exc_info=True)

    # Emotion → (length_scale, noise_w_scale): pace + expressiveness so the
    # voice isn't monotone. Lower length_scale = faster/livelier; higher = slower
    # and gentler. Higher noise_w = more vocal variation (warmth).
    _PIPER_STYLE = {
        "happy": (0.93, 0.85),
        "excited": (0.88, 0.9),
        "curious": (0.97, 0.8),
        "serious": (1.02, 0.7),
        "empathetic": (1.1, 0.8),   # slower, gentler for sad/low moods
    }

    def _syn_config(self, emotion: str):
        from piper import SynthesisConfig
        length_scale, noise_w = self._PIPER_STYLE.get(emotion, (0.98, 0.8))
        return SynthesisConfig(length_scale=length_scale, noise_w_scale=noise_w)

    def _synth_wav(self, text: str, emotion: str = "curious") -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._voice.synthesize_wav(text, wf, syn_config=self._syn_config(emotion))
        return buf.getvalue()

    def synthesize_for_playback(self, text: str, emotion: str = "curious") -> tuple[bytes, float]:
        text = text.replace("*", "").strip()
        if len(text) < 2:
            return b"", 0.0
        # Telugu / Hindi (native script) MUST use their Edge voices — Piper is
        # English-only and would read them with an English accent.
        if not self.available or has_native_script(text):
            return super().synthesize_for_playback(text, emotion)
        t0 = time.perf_counter()
        try:
            return self._synth_wav(text, emotion), time.perf_counter() - t0
        except Exception:
            log.debug("Piper synth failed — Edge fallback", exc_info=True)
            return super().synthesize_for_playback(text, emotion)

    def synthesize_backchannel(self, text: str) -> bytes:
        if not self.available:
            return super().synthesize_backchannel(text)
        text = text.replace("*", "").strip()
        if not text:
            return b""
        try:
            return self._synth_wav(text)
        except Exception:
            return super().synthesize_backchannel(text)


def make_tts_engine() -> EdgeStreamingTTS:
    """Factory: pick the TTS backend (Phase 2). Piper falls back to Edge per-call."""
    if config.TTS_BACKEND.lower() == "piper":
        log.info("TTS backend = piper")
        return PiperTTS()
    return EdgeStreamingTTS()
