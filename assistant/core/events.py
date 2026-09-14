"""
Shared inter-thread plumbing: queues, events and global app state.

This is the backbone of the producer/consumer architecture. Every service
talks to others ONLY through these queues + events, so no service ever
blocks another.

         mic frames           transcript            text-to-speak
AudioInput ───▶ stt_in_q ─▶ STT ─▶ transcript_q ─▶ Dispatcher ─▶ tts_q ─▶ TTS
                                                        │
                                                        └─▶ gui_q ─▶ GUI (main thread)
"""
import collections
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

_turnlog = logging.getLogger("turn")


class Mode(Enum):
    AWAKE = auto()
    SLEEP = auto()


@dataclass
class Utterance:
    """A captured chunk of speech handed from AudioInput to STT."""
    audio: np.ndarray            # float32 mono @ 16 kHz, range [-1, 1]
    during_speech: bool = False  # was the assistant speaking when captured?
    perf: object = None          # perf.Perf, created when the user stopped speaking
    utterance_id: str = ""
    is_partial: bool = False
    sequence: int = 0
    kind: str = "speech"         # "speech" | "filler" (FIX F: backchannel blips)


@dataclass
class Transcript:
    text: str
    language: Optional[str] = None   # detected lang code, e.g. "en"/"hi"/"te"
    language_probability: float = 0.0
    during_speech: bool = False
    perf: object = None              # carried through from the originating Utterance
    utterance_id: str = ""
    is_partial: bool = False
    sequence: int = 0
    kind: str = "speech"             # "speech" | "filler" (FIX F)


def new_utterance_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class SpeechItem:
    """A unit of text queued for the TTS service.

    Every item is stamped with the turn_id it belongs to. `is_end` marks the
    END(turn_id) sentinel — the ONLY thing that flips speaking off.
    """
    text: str
    perf: object = None              # perf.Perf to finalize when this audio starts
    turn_id: str = ""                # which interaction this belongs to
    is_end: bool = False             # END(turn_id) sentinel


@dataclass
class GuiEvent:
    kind: str                    # "status" | "subtitle" | "notification" | "mode"
    text: str = ""
    role: str = "assistant"      # for subtitle


class AppState:
    """Thread-safe-ish shared state (single writer per flag in practice)."""

    def __init__(self):
        self.mode = Mode.AWAKE
        self._lock = threading.Lock()
        self._speaking = threading.Event()  # set while TTS is producing audio
        self.audio_state = AudioState()
        self.emotion = "curious"

    # --- assistant-speaking flag (read by AudioInput/STT for barge-in) ---
    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    def set_speaking(self, value: bool):
        if value:
            self._speaking.set()
        else:
            self._speaking.clear()

    def set_mode(self, mode: Mode):
        with self._lock:
            self.mode = mode

    def set_emotion(self, emotion: str):
        with self._lock:
            self.emotion = emotion


class AudioState:
    """Playback-derived audio state for duplex tagging and future AEC refs."""

    def __init__(self):
        self._during_speech = threading.Event()

    @property
    def during_speech(self) -> bool:
        return self._during_speech.is_set()

    def set_during_speech(self, value: bool):
        if value:
            self._during_speech.set()
        else:
            self._during_speech.clear()


@dataclass
class Bus:
    """The single object every service receives — its wiring harness."""
    stt_in_q: "queue.Queue[Utterance]" = field(default_factory=lambda: queue.Queue(maxsize=64))
    transcript_q: "queue.Queue[Transcript]" = field(default_factory=lambda: queue.Queue(maxsize=64))
    tts_q: "queue.Queue[SpeechItem]" = field(default_factory=lambda: queue.Queue(maxsize=128))
    gui_q: "queue.Queue[GuiEvent]" = field(default_factory=lambda: queue.Queue(maxsize=256))

    # Control signals
    shutdown: threading.Event = field(default_factory=threading.Event)
    interrupt: threading.Event = field(default_factory=threading.Event)  # stop TTS now
    stt_ready: threading.Event = field(default_factory=threading.Event)

    state: AppState = field(default_factory=AppState)

    # Perf object the NEXT enqueued speech should adopt (set by the dispatcher
    # so the first spoken line of an interaction carries its timing context).
    _pending_perf: object = None

    # The turn currently being produced/spoken. TTS ignores any item whose
    # turn_id != this (stale items from interrupted turns). None = idle.
    current_turn_id: Optional[str] = None

    # Backchannel engine (set by the pipeline once TTS exists). Producers call
    # bus.backchannel.maybe_fire(...) — guarded against None everywhere.
    backchannel: object = None

    # FIX G: speaker-reference ring buffer for echo cancellation. The TTS engine
    # pushes each PCM block it sends to the device; AudioInput reads the most
    # recent block as the far-end reference so AEC can subtract / gate echo
    # instead of blind-gating on energy. Each entry is (monotonic_ts, pcm_bytes).
    _speaker_ref: "collections.deque" = field(
        default_factory=lambda: collections.deque(maxlen=200))
    _speaker_ref_lock: threading.Lock = field(default_factory=threading.Lock)

    # ---- convenience helpers -------------------------------------------
    def post_gui(self, kind: str, text: str = "", role: str = "assistant"):
        try:
            self.gui_q.put_nowait(GuiEvent(kind, text, role))
        except queue.Full:
            pass

    def set_pending_perf(self, perf):
        self._pending_perf = perf

    # ---- speaker reference (echo cancellation, FIX G) ------------------
    def push_speaker_ref(self, pcm: bytes):
        """Called by the TTS engine for every audio block it plays."""
        if not pcm:
            return
        with self._speaker_ref_lock:
            self._speaker_ref.append((time.monotonic(), pcm))

    def get_speaker_reference(self, max_age: float = 0.4):
        """Most recent speaker PCM (echo reference), or None if playback is
        idle/stale — so AEC only suppresses when there's real echo to cancel."""
        with self._speaker_ref_lock:
            if not self._speaker_ref:
                return None
            ts, pcm = self._speaker_ref[-1]
        return pcm if (time.monotonic() - ts) <= max_age else None

    def clear_speaker_ref(self):
        with self._speaker_ref_lock:
            self._speaker_ref.clear()

    # ---- turn lifecycle ------------------------------------------------
    def start_turn(self) -> str:
        """Begin a new interaction; returns its turn_id."""
        turn_id = uuid.uuid4().hex[:8]
        self.current_turn_id = turn_id
        _turnlog.info("[TURN START] %s", turn_id)
        return turn_id

    def end_turn(self, turn_id: str):
        """Mark the end of a turn's generation by queuing END(turn_id).

        speaking flips False only when TTS consumes this sentinel — never on an
        empty queue — so long stories can't stop midway.
        """
        self.tts_q.put(SpeechItem("", turn_id=turn_id, is_end=True))

    def speak(self, text: str, perf=None):
        """Enqueue text (stamped with the current turn_id) for TTS.

        The first speech of an interaction adopts the pending Perf (so total
        latency is measured to *this* audio).
        """
        if not (text and text.strip()):
            return
        if perf is None and self._pending_perf is not None:
            perf, self._pending_perf = self._pending_perf, None
        if perf is not None and hasattr(perf, "stamp"):
            perf.stamp("tts_queued")
            perf.trace("tts_queued", text=repr(text.strip()[:80]))
        turn_id = self.current_turn_id or ""
        if turn_id:
            _turnlog.info("[SENTENCE] %s %r", turn_id, text.strip()[:50])
        self.tts_q.put(SpeechItem(text.strip(), perf, turn_id=turn_id))

    def say(self, text: str):
        """One-shot system speech (greeting, reminder, wake-up) as its own
        self-contained turn, so it plays and cleanly flips speaking back off."""
        if not (text and text.strip()):
            return
        turn_id = self.start_turn()
        self.speak(text)
        self.end_turn(turn_id)

    def request_interrupt(self):
        """Barge-in: stop playback, clear the queue, invalidate the turn.

        Any items already queued (or arriving late) carry the old turn_id, so
        TTS drops them automatically once current_turn_id has moved on.
        """
        old = self.current_turn_id
        _turnlog.info("[BARGE-IN] %s", old)
        self.interrupt.set()
        self.current_turn_id = None          # invalidate: old items become stale
        self.state.set_speaking(False)
        try:
            while True:
                self.tts_q.get_nowait()
        except queue.Empty:
            pass
