"""
TTSService — the speech consumer thread.

Pulls sentences off tts_q and renders them with EdgeStreamingTTS. Manages the
shared `speaking` flag (so AudioInput/dispatcher know barge-in is possible) and
honours the interrupt event for instant stop.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from assistant import config
from assistant.core.events import Bus
from assistant.core.log import get
from assistant.core.perf import tracker
from assistant.tts.engine import make_tts_engine

log = get("tts")


class TTSService(threading.Thread):
    def __init__(self, bus: Bus):
        super().__init__(name="TTS", daemon=True)
        self.bus = bus
        self.engine = make_tts_engine()   # Phase 2: edge | piper (auto-fallback)
        # FIX G: stream every played PCM block into the bus echo-reference buffer.
        self.engine.ref_sink = self.bus.push_speaker_ref
        self._synth_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TTS-Synth")

    def run(self):
        while not self.bus.shutdown.is_set():
            try:
                item = self.bus.tts_q.get(timeout=0.3)
            except queue.Empty:
                continue  # NEVER finish on empty queue — only END does that

            cur = self.bus.current_turn_id

            # --- END sentinel: the ONLY thing that ends a turn --------------
            if item.is_end:
                if item.turn_id == cur:
                    log.info("[END] %s", item.turn_id)
                    self._finish_speaking()
                # stale END (older interrupted turn) → ignore
                continue

            # --- stale item from an interrupted/old turn → drop ------------
            if cur is None or item.turn_id != cur:
                log.info("[DROPPED STALE] %s", item.turn_id)
                continue

            # The TTS thread must survive ANY error while speaking one item — if
            # it dies, `speaking` stays stuck and Jarvis goes silent for the rest
            # of the session. Catch here, drop just this item, and keep going; the
            # turn's END sentinel will still arrive and clear the speaking flag.
            try:
                self._speak_turn_item(item, cur)
            except Exception as e:
                log.exception("TTS item failed — skipping, thread stays alive")
                self.bus.post_gui("notification", f"TTS error: {e}")

    def _speak_turn_item(self, item, cur):
        """Speak one item, pre-synthesizing the next queued item if available."""
        current = item
        current_audio = None
        current_synth = 0.0
        self.bus.interrupt.clear()
        self.bus.state.set_speaking(True)
        self.bus.post_gui("status", "Speaking...")

        if hasattr(self.engine, "synthesize_for_playback") and not config.FLAGS.demo_mode:
            if current.perf is not None and hasattr(current.perf, "stamp"):
                current.perf.stamp("tts_synth_start")
                current.perf.trace("tts_synth_start", text=repr(current.text[:80]))
            current_audio, current_synth = self.engine.synthesize_for_playback(
                current.text, self.bus.state.emotion
            )
            if current.perf is not None and hasattr(current.perf, "stamp"):
                current.perf.stamp("tts_synth_end")
                current.perf.trace("tts_synth_done", took=f"{current_synth:.2f}s")

        while current and not self.bus.shutdown.is_set():
            if self.bus.current_turn_id != cur or current.turn_id != cur:
                log.info("[DROPPED STALE] %s", current.turn_id)
                return

            next_item = None
            next_future = None
            try:
                nxt = self.bus.tts_q.get_nowait()
                if nxt.is_end:
                    if nxt.turn_id == cur:
                        next_item = nxt
                    else:
                        log.info("[DROPPED STALE END] %s", nxt.turn_id)
                elif nxt.turn_id == cur:
                    next_item = nxt
                    if hasattr(self.engine, "synthesize_for_playback") and not config.FLAGS.demo_mode:
                        next_future = self._synth_pool.submit(
                            self.engine.synthesize_for_playback,
                            nxt.text,
                            self.bus.state.emotion,
                        )
                else:
                    log.info("[DROPPED STALE] %s", nxt.turn_id)
            except queue.Empty:
                pass

            self.bus.post_gui("subtitle", current.text, role="assistant")

            log.info("speak turn=%s: %r", cur, current.text[:80])
            if config.FLAGS.demo_mode:
                print(f"\n[JARVIS]: {current.text}\n")
            else:
                self._speak_measured(current.text, current.perf, current_audio, current_synth)

            if next_item is None:
                return
            if next_item.is_end:
                if self.bus.current_turn_id == cur:
                    log.info("[END] %s", cur)
                    self._finish_speaking()
                return
            current = next_item
            if next_future is not None:
                try:
                    current_audio, current_synth = next_future.result()
                except Exception as e:
                    log.exception("TTS pre-synth error")
                    self.bus.post_gui("notification", f"TTS error: {e}")
                    current_audio, current_synth = None, 0.0
            else:
                current_audio, current_synth = None, 0.0

    def _speak_measured(self, text: str, perf, audio=None, synth_dt: float = 0.0):
        """Synthesize+play, capturing TTS synth/start and total latency."""
        play_begin = {}

        def on_play_start():
            t = play_begin.setdefault("t", time.perf_counter())
            self.bus.state.audio_state.set_during_speech(True)
            log.info("audio_state during_speech=True")
            if perf is not None and hasattr(perf, "stamp") and not perf.tts_play_start:
                perf.stamp("tts_play_start", t)
                perf.trace("tts_play_start")

        t_synth_start = perf.tts_synth_start if perf is not None and getattr(perf, "tts_synth_start", 0.0) else time.perf_counter()
        try:
            if audio is not None and hasattr(self.engine, "play_synthesized"):
                self.engine.play_synthesized(audio, self.bus.interrupt,
                                             on_play_start=on_play_start)
            else:
                if perf is not None and hasattr(perf, "stamp"):
                    perf.stamp("tts_synth_start", t_synth_start)
                    perf.trace("tts_synth_start", text=repr(text[:80]))
                synth_dt = self.engine.speak(text, self.bus.interrupt,
                                             on_play_start=on_play_start)
                if perf is not None and hasattr(perf, "stamp") and not perf.tts_synth_end:
                    synth_end = play_begin.get("t", time.perf_counter())
                    perf.stamp("tts_synth_end", synth_end)
                    perf.trace("tts_synth_done", took=f"{synth_dt:.2f}s")
        except Exception as e:
            log.exception("TTS error")
            self.bus.post_gui("notification", f"TTS error: {e}")
        finally:
            self.bus.state.audio_state.set_during_speech(False)
            log.info("audio_state during_speech=False")

        if perf is not None and not perf.committed:
            t_play = play_begin.get("t", time.perf_counter())
            perf.stamp("tts_play_end")
            if not perf.tts_play_start:
                perf.stamp("tts_play_start", t_play)
            perf.tts_synth = synth_dt
            perf.tts_start = t_play - t_synth_start
            perf.total = t_play - perf.t_start
            perf.trace("tts_play_done", play=f"{perf.tts_play_end - perf.tts_play_start:.2f}s")
            tracker.commit(perf)

    def speak_backchannel(self, text: str):
        """Play a short listening sound OUTSIDE the main turn machinery.

        Called from the BackchannelEngine worker thread. It deliberately:
          * never sets state.speaking (it is not a "turn"),
          * never creates a turn_id or touches tts_q,
          * never writes to conversation history,
          * stops the instant the real response begins (state.speaking) or an
            interrupt/shutdown fires.
        """
        if config.FLAGS.demo_mode:
            return
        text = (text or "").strip()
        if not text:
            return
        try:
            mp3 = self.engine.synthesize_backchannel(text)
        except Exception:
            log.debug("backchannel synth failed", exc_info=True)
            return
        if not mp3 or self.bus.state.speaking or self.bus.interrupt.is_set():
            return

        def _stop():
            return (self.bus.state.speaking or self.bus.interrupt.is_set()
                    or self.bus.shutdown.is_set())

        log.info("backchannel: %r", text)
        self.engine.play_backchannel(mp3, _stop, volume=config.BACKCHANNEL_VOLUME)

    def _finish_speaking(self):
        self.bus.state.set_speaking(False)
        self.bus.clear_speaker_ref()  # FIX G: no echo to cancel once we stop
        self.bus.post_gui("status", "Listening...")
        self.bus.post_gui("subtitle", "", role="assistant")
