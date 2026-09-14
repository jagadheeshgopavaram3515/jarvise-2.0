"""
AudioInputService — the microphone producer thread.

Continuously captures mic audio with sounddevice, segments it into
utterances with VAD, and pushes finished utterances onto stt_in_q.

Key real-time properties:
  * NEVER stops capturing — it records even while the assistant is speaking
    (full-duplex), tagging those utterances so the dispatcher can treat them
    as potential barge-in.
  * Emits as soon as a silence gap is detected, so STT can start instantly.
"""
from __future__ import annotations

import collections
import queue
import re
import threading
import time

import numpy as np

from assistant import config
from assistant.audio.aec import EchoCanceller
from assistant.audio.vad import VAD, _HAS_WEBRTC
from assistant.core.events import Bus, Utterance, new_utterance_id
from assistant.core.log import get
from assistant.core.perf import Perf

log = get("audio")

# FIX F: short voiced blips below the min-utterance length used to be dropped.
# Many are conversational fillers ("hmm", "ahh") that the dispatcher can use as
# backchannel / soft barge-in signals, so we tag and forward them instead.
_FILLER_MIN_FRAMES = max(1, 150 // config.VAD_FRAME_MS)   # >= ~150 ms of voiced audio
_FILLER_ENERGY_THRESHOLD = 200.0                          # int16 RMS gate (skip breaths/noise)

try:
    import sounddevice as sd
    _HAS_SD = True
except Exception:
    _HAS_SD = False


class AudioInputService(threading.Thread):
    def __init__(self, bus: Bus):
        super().__init__(name="AudioInput", daemon=True)
        self.bus = bus
        self.vad = VAD()
        self.aec = EchoCanceller()
        self.sample_rate = config.STT_SAMPLE_RATE
        self.frame_len = int(self.sample_rate * config.VAD_FRAME_MS / 1000)  # samples/frame
        self._silence_frames = max(1, config.VAD_SILENCE_MS // config.VAD_FRAME_MS)
        self._silence_frames_max = max(1, config.VAD_SILENCE_MAX_MS // config.VAD_FRAME_MS)
        self._min_frames = max(1, config.VAD_MIN_UTTERANCE_MS // config.VAD_FRAME_MS)
        self._max_frames = max(1, config.VAD_MAX_UTTERANCE_MS // config.VAD_FRAME_MS)
        self._ring: "queue.Queue[bytes]" = queue.Queue()
        self._partial_interval_frames = max(1, config.STT_PARTIAL_INTERVAL_MS // config.VAD_FRAME_MS)
        self._partial_min_frames = max(1, config.STT_PARTIAL_MIN_AUDIO_MS // config.VAD_FRAME_MS)

    def _endpoint_frames(self, voiced_len: int) -> int:
        """Adaptive silence needed to end the turn — grows with speech length."""
        voiced_sec = voiced_len * config.VAD_FRAME_MS / 1000.0
        needed_ms = min(config.VAD_SILENCE_MAX_MS,
                        config.VAD_SILENCE_MS + voiced_sec * config.VAD_SILENCE_GROW_MS_PER_SEC)
        return max(1, int(needed_ms / config.VAD_FRAME_MS))

    # sounddevice callback runs in PortAudio's thread — keep it trivial.
    def _callback(self, indata, frames, time_info, status):  # noqa: D401
        self._ring.put(bytes(indata))

    def run(self):
        if config.FLAGS.demo_mode or not _HAS_SD:
            log.warning("AudioInput disabled (demo_mode=%s, sounddevice=%s)",
                        config.FLAGS.demo_mode, _HAS_SD)
            return  # In demo mode the GUI/console drives input instead.
        # Self-healing capture: if the mic stream or the segment loop dies (a USB
        # mic unplugged, a PortAudio glitch), log it and re-open the stream after
        # a short back-off instead of letting the thread terminate. This keeps
        # AudioInput alive across transient hardware faults without a full restart.
        backoff = 1.0
        while not self.bus.shutdown.is_set():
            try:
                dev = sd.default.device
                name = sd.query_devices(kind="input")["name"]
                log.info("Mic open: device=%s '%s' rate=%d frame=%d webrtcvad=%s",
                         dev, name, self.sample_rate, self.frame_len, _HAS_WEBRTC)
                with sd.RawInputStream(samplerate=self.sample_rate, blocksize=self.frame_len,
                                       dtype="int16", channels=1, callback=self._callback):
                    self._segment_loop()
                backoff = 1.0  # clean exit (shutdown) — loop condition will end us
            except Exception as e:  # pragma: no cover - hardware dependent
                log.exception("Mic error — reopening stream after %.1fs", backoff)
                self.bus.post_gui("notification", f"Mic error: {e}")
                # Sleep in slices so shutdown stays responsive.
                slept = 0.0
                while slept < backoff and not self.bus.shutdown.is_set():
                    time.sleep(0.2)
                    slept += 0.2
                backoff = min(backoff * 2, 15.0)   # capped exponential back-off

    def _segment_loop(self):
        voiced: list[bytes] = []
        triggered = False
        silence_run = 0
        # Small pre-roll so we don't clip the start of speech.
        preroll = collections.deque(maxlen=self._silence_frames)

        frames_seen = speech_seen = 0
        heartbeat = max(1, int(2000 / config.VAD_FRAME_MS))  # ~2s
        guard_s = config.SPEECH_TAIL_GUARD_MS / 1000.0
        was_speaking = False
        speech_ended_at = 0.0
        started_during_speech = False
        speech_started_at = 0.0
        utterance_id = ""
        partial_seq = 0
        last_partial_at = 0
        # Backchannel: fire a "pause" cue once when the user goes silent
        # mid-utterance (voiced→silent for BACKCHANNEL_PAUSE_MS) without the turn
        # actually ending. Re-armed on the next voiced frame.
        pause_frames = max(1, config.BACKCHANNEL_PAUSE_MS // config.VAD_FRAME_MS)
        pause_fired = False
        # Phase 4: one brief "I'm listening, sir" once an utterance runs long.
        long_ack_frames = max(1, config.BACKCHANNEL_LONG_SPEECH_MS // config.VAD_FRAME_MS)
        long_ack_fired = False

        while not self.bus.shutdown.is_set():
            try:
                frame = self._ring.get(timeout=0.5)
            except queue.Empty:
                continue
            if len(frame) < self.frame_len * 2:  # 2 bytes/sample
                continue

            # Half-duplex gate: ignore mic while the assistant speaks (and a
            # short guard after) so it doesn't transcribe its own voice.
            if not config.FULL_DUPLEX:
                speaking_now = self.bus.state.speaking
                if speaking_now:
                    was_speaking = True
                    if triggered or voiced:
                        voiced, triggered, silence_run = [], False, 0
                        preroll.clear()
                    continue
                if was_speaking:
                    was_speaking = False
                    speech_ended_at = time.monotonic()
                if time.monotonic() - speech_ended_at < guard_s:
                    continue

            # Suppress echo ONLY while audio is actually playing (during_speech),
            # NOT for the whole turn — so the gaps between sentence chunks stay
            # fully open and you can speak/command in those breaks. The speaker-
            # reference PCM lets the AEC tell your voice from Jarvis's echo.
            audio_playing = self.bus.state.audio_state.during_speech
            frame = self.aec.process_audio(
                frame,
                speaker_reference=self.bus.get_speaker_reference(),
                assistant_speaking=audio_playing,
            )
            speech = self.vad.is_speech(frame)
            frames_seen += 1
            speech_seen += 1 if speech else 0
            if frames_seen % heartbeat == 0:
                log.info("mic alive: %d frames, %d voiced in last window",
                         frames_seen, speech_seen)
                speech_seen = 0

            if not triggered:
                preroll.append(frame)
                if speech:
                    triggered = True
                    voiced = list(preroll)
                    silence_run = 0
                    # FIX C: tag off the turn-wide `speaking` flag, not the racy
                    # per-block `during_speech` (which clears between sentences and
                    # mis-tagged 19/20 barge-in windows as False).
                    started_during_speech = (self.bus.state.speaking
                                             or self.bus.state.audio_state.during_speech)
                    speech_started_at = time.perf_counter()
                    utterance_id = new_utterance_id()
                    partial_seq = 0
                    last_partial_at = len(voiced)
                    log.info("speech START during_speech=%s", started_during_speech)
            else:
                voiced.append(frame)
                silence_run = 0 if speech else silence_run + 1
                # Adaptive: the longer the utterance, the more pause we allow
                # before deciding the user is done (so long sentences with
                # mid-thought pauses are captured fully).
                endpoint_frames = self._endpoint_frames(len(voiced))
                if config.SEMANTIC_ENDPOINTING and self._looks_like_complete_thought(voiced):
                    endpoint_frames = min(
                        endpoint_frames,
                        max(1, config.SEMANTIC_ENDPOINT_SILENCE_MS // config.VAD_FRAME_MS),
                    )
                end_of_utt = silence_run >= endpoint_frames
                too_long = len(voiced) >= self._max_frames

                # Mid-speech pause → a single "pause" backchannel cue.
                if speech:
                    pause_fired = False
                elif (not pause_fired and not end_of_utt and not too_long
                      and silence_run == pause_frames
                      and self.bus.backchannel is not None):
                    pause_fired = True
                    self.bus.backchannel.maybe_fire("", "pause", utterance_id,
                                                    self.bus.state.emotion)

                # Phase 4: long continuous speech → one brief active-listening ack.
                if (not long_ack_fired and len(voiced) >= long_ack_frames
                        and not end_of_utt and not too_long
                        and self.bus.backchannel is not None):
                    long_ack_fired = True
                    self.bus.backchannel.maybe_fire("", "active_listening",
                                                    utterance_id, self.bus.state.emotion)

                if (config.STT_ENABLE_PARTIALS
                        and self.bus.state.speaking  # partials ONLY while we speak → fast "stop" barge-in
                        and len(voiced) >= self._partial_min_frames
                        and len(voiced) - last_partial_at >= self._partial_interval_frames
                        and not end_of_utt
                        and not too_long):
                    partial_seq += 1
                    last_partial_at = len(voiced)
                    self._emit_partial(voiced, speech_started_at, started_during_speech,
                                       utterance_id, partial_seq)

                if end_of_utt or too_long:
                    # "User stopped speaking" = now minus the trailing silence.
                    now = time.perf_counter()
                    t_speech_end = (now - silence_run * config.VAD_FRAME_MS / 1000.0
                                    if end_of_utt else now)
                    if len(voiced) >= self._min_frames:
                        log.info("speech END -> emit utterance (%d frames)", len(voiced))
                        self._emit(voiced, speech_started_at, t_speech_end,
                                   started_during_speech, utterance_id, partial_seq + 1)
                    else:
                        # FIX F: too short for a real query — classify as a filler
                        # backchannel signal instead of silently dropping.
                        self._emit_filler(voiced, started_during_speech, utterance_id)
                    voiced, triggered, silence_run = [], False, 0
                    started_during_speech = False
                    speech_started_at = 0.0
                    utterance_id = ""
                    partial_seq = 0
                    last_partial_at = 0
                    pause_fired = False
                    long_ack_fired = False
                    preroll.clear()

    def _emit(self, frames: list[bytes], t_speech_start: float, t_speech_end: float,
              during_speech: bool, utterance_id: str, sequence: int):
        pcm = b"".join(frames)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        perf = Perf(t_start=t_speech_end, speech_start=t_speech_start)
        perf.stamp("stt_queued")
        duration = max(0.0, t_speech_end - t_speech_start)
        perf.trace(
            "speech_end",
            duration=f"{duration:.2f}s",
            frames=len(frames),
            during_speech=during_speech,
        )
        utt = Utterance(audio=audio, during_speech=during_speech, perf=perf,
                        utterance_id=utterance_id, is_partial=False, sequence=sequence)
        try:
            self.bus.stt_in_q.put_nowait(utt)
            perf.trace("stt_queued", qsize=self.bus.stt_in_q.qsize())
        except queue.Full:
            perf.trace("stt_queue_full")
            pass

    def _emit_partial(self, frames: list[bytes], t_speech_start: float,
                      during_speech: bool, utterance_id: str, sequence: int):
        pcm = b"".join(frames)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        perf = Perf(t_start=time.perf_counter(), speech_start=t_speech_start)
        perf.stamp("stt_queued")
        utt = Utterance(audio=audio, during_speech=during_speech, perf=perf,
                        utterance_id=utterance_id, is_partial=True, sequence=sequence)
        try:
            self.bus.stt_in_q.put_nowait(utt)
            log.info("[STT PARTIAL QUEUED] id=%s seq=%d audio=%.2fs qsize=%d",
                     utterance_id, sequence, len(audio) / self.sample_rate,
                     self.bus.stt_in_q.qsize())
        except queue.Full:
            perf.trace("stt_partial_queue_full")

    def _emit_filler(self, frames: list[bytes], during_speech: bool, utterance_id: str):
        """FIX F: surface a short voiced blip ("hmm"/"ahh") as a backchannel filler.

        We skip STT entirely for these (no point transcribing a grunt, and it
        keeps the queue clear) — the dispatcher reacts to the `kind="filler"`
        signal rather than to any text.
        """
        if len(frames) < _FILLER_MIN_FRAMES:
            log.info("speech too short (%d frames), dropped", len(frames))
            return
        pcm = b"".join(frames)
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        if rms <= _FILLER_ENERGY_THRESHOLD:
            log.info("speech too short (%d frames), dropped (low energy %.0f)", len(frames), rms)
            return
        audio = samples.astype(np.float32) / 32768.0
        utt = Utterance(audio=audio, during_speech=during_speech,
                        utterance_id=utterance_id or new_utterance_id(),
                        kind="filler")
        try:
            self.bus.stt_in_q.put_nowait(utt)
            log.info("[FILLER] backchannel blip emitted (%d frames, rms=%.0f)", len(frames), rms)
        except queue.Full:
            pass

    def _looks_like_complete_thought(self, frames: list[bytes]) -> bool:
        """Cheap acoustic/size heuristic for short commands before silence max."""
        audio_ms = len(frames) * config.VAD_FRAME_MS
        if audio_ms < config.SEMANTIC_ENDPOINT_MIN_MS:
            return False
        # Semantic endpointing is conservative until partial text feedback is
        # available in the capture thread. Short question/command utterances
        # get a lower silence threshold; long narration keeps adaptive VAD.
        return audio_ms < 3500
