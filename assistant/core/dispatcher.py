"""
Dispatcher — the brain that routes transcripts.

Consumes Transcript objects and decides:
  1. Barge-in: if the user speaks while the assistant is talking, stop speech
     immediately (always for interrupt-words; optionally for any utterance).
  2. Wake/sleep: honour "wake up" / quiet-mode transitions.
  3. Commands: deterministic handlers first (fast path, no LLM).
  4. LLM: otherwise stream Gemini's reply sentence-by-sentence into tts_q.

Runs in its own thread; never blocks audio capture or playback.
"""
from __future__ import annotations

import queue
import re
import threading
import time

from assistant import config
from assistant.commands import handlers
from assistant.core.emotion import EmotionEngine
from assistant.core.events import Bus, Mode, Transcript
from assistant.core.humanizer import Humanizer
from assistant.core.log import get
from assistant.llm import GeminiClient, get_llm_client
from assistant.memory import store
from assistant.reminders.scheduler import ReminderStore

log = get("dispatch")
_PURE_STOP = {"stop", "cancel", "enough", "quiet", "halt", "shush", "wait"}
_EARLY_COMMAND = re.compile(
    r"\b(what|when|where|who|why|how|tell|search|open|play|time|date|stop|wait|jarvis)\b",
    re.I,
)


class Dispatcher(threading.Thread):
    def __init__(self, bus: Bus, reminders: ReminderStore):
        super().__init__(name="Dispatcher", daemon=True)
        self.bus = bus
        self.reminders = reminders
        self.llm = None
        self.emotion = EmotionEngine()
        self.humanizer = Humanizer()   # Phase 3: occasional response lead-ins
        self._handled_utterances: set[str] = set()
        self._partial_text: dict[str, str] = {}
        self._last_dispatched = ""        # Fix 5: drop repeated identical transcripts
        self._last_dispatched_count = 0
        self._last_dispatch_time = 0.0    # Fix 2: throttle Gemini dispatches
        self._min_dispatch_interval = config.GEMINI_MIN_DISPATCH_INTERVAL

    def run(self):
        try:
            if config.LLM_ROUTING_ENABLED:
                from assistant.orchestration import get_orchestrator
                self.llm = get_orchestrator()
            else:
                self.llm = get_llm_client()
        except Exception as e:
            self.bus.post_gui("notification", f"LLM init failed: {e}")

        while not self.bus.shutdown.is_set():
            try:
                tr = self.bus.transcript_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._route(tr)
            except Exception as e:
                self.bus.post_gui("notification", f"Dispatch error: {e}")

    # --------------------------------------------------------------------
    def _route(self, tr: Transcript):
        # FIX F: backchannel fillers ("hmm"/"ahh") are signals, never queries.
        if getattr(tr, "kind", "speech") == "filler":
            self._handle_filler(tr)
            return
        text = tr.text.strip()
        low = text.lower()
        words = set(low.split())

        # ---- Fast, rule-based barge-in (works like the spacebar) ---------
        # An interrupt word heard while Jarvis is speaking stops TTS IMMEDIATELY
        # — for a PARTIAL or a FINAL — before any LLM work. Conversation memory
        # is preserved: an in-flight reply is saved to history when cut (below).
        if self.bus.state.speaking or tr.during_speech:
            hit = words & config.INTERRUPT_WORDS
            if hit:
                self.bus.request_interrupt()
                log.info("[BARGE-IN] stop word=%s partial=%s text=%r", hit, tr.is_partial, text)
                remaining = [w for w in low.split() if w not in config.INTERRUPT_WORDS]
                # Pure stop / name-only / any partial → just halt. A FINAL like
                # "jarvis, what's the time" falls through to run the command.
                if (tr.is_partial or not remaining or words <= _PURE_STOP
                        or words <= {config.ASSISTANT_NAME.lower()}):
                    return
                text = " ".join(remaining)
                low, words = text.lower(), set(text.lower().split())
            elif config.BARGE_IN_KEYWORDS_ONLY:
                return  # non-keyword speech over our own voice → ignore (echo)
            else:
                self.bus.request_interrupt()  # general barge-in: user takes over

        # Partials are fast barge-in probes ONLY — never sent to the LLM.
        if tr.is_partial:
            return
        if tr.utterance_id and tr.utterance_id in self._handled_utterances:
            return

        log.info("route: %r (mode=%s speaking=%s)", text,
                 self.bus.state.mode.name, self.bus.state.speaking)
        if tr.perf is not None:
            tr.perf.stamp("dispatch_start")
            tr.perf.trace("dispatch_start", speaking=self.bus.state.speaking, text=repr(text[:80]))

        # ---- SLEEP mode: only "wake up" matters --------------------------
        if self.bus.state.mode is Mode.SLEEP:
            if "wake up" in low:
                self.bus.state.set_mode(Mode.AWAKE)
                self.bus.post_gui("mode", "awake")
                self.bus.say("I am back online, sir.")
            return

        # Open a turn. Every sentence is stamped with this turn_id, and END is
        # ALWAYS sent in the finally — so the turn can never leak or hang, and
        # speaking flips off only when TTS consumes that END.
        # Fix 5: drop repeated identical transcripts (the "Thank you." loop that
        # fired 4× → 4 LLM calls → 429). First occurrence passes; the 2nd+
        # identical-in-a-row is dropped; a different text resets the counter.
        # Set BEFORE start_turn and NOT cleared on turn start — dispatch is
        # sequential, so clearing per-turn would let the next duplicate through.
        if text and text.lower() == self._last_dispatched.lower():
            self._last_dispatched_count += 1
            if self._last_dispatched_count >= 2:
                log.warning("[DEDUP] dropping repeated: %r", text)
                if tr.perf is not None:
                    tr.perf.trace("dropped_duplicate", text=repr(text[:40]))
                return
        else:
            self._last_dispatched = text
            self._last_dispatched_count = 1

        perf = tr.perf
        turn_id = self.bus.start_turn()
        # Jarvis is now responding → stop & disarm backchannels for this turn.
        if self.bus.backchannel is not None:
            self.bus.backchannel.reset()
        self.bus.set_pending_perf(perf)
        try:
            mood = self.emotion.update(text)
            self.bus.state.set_emotion(mood)
            store.remember_mood(text)   # Phase 7: persist emotional state
            # ---- Commands (fast path) ------------------------------------
            if handlers.handle(text, self.bus, self.reminders):
                log.info("handled as command")
                return

            # ---- LLM streaming reply -------------------------------------
            # Fix 2: throttle ONLY the Gemini path — never dispatch faster than
            # the min interval. Commands and barge-in (handled above) are exempt.
            now = time.monotonic()
            gap = now - self._last_dispatch_time
            if gap < self._min_dispatch_interval:
                log.warning("[RATE] too fast (%.1fs < %.1fs), skipping LLM: %r",
                            gap, self._min_dispatch_interval, text)
                return
            self._last_dispatch_time = now
            log.info("-> LLM")
            self.bus.post_gui("status", "Thinking...")
            if self.llm is None:
                self.bus.speak("My language model is not available, sir.")
                return
            # Phase 3: occasional, rate-limited, non-repeating human lead-in,
            # spoken before the reply (and while Gemini's first token streams).
            if config.ENABLE_HUMANIZER:
                lead_in = self.humanizer.maybe_filler(text)
                if lead_in:
                    self.bus.speak(lead_in)
            spoke = False
            spoken_parts: list[str] = []
            t0 = time.perf_counter()
            if perf is not None:
                perf.stamp("gemini_start", t0)
                perf.trace("gemini_start")
            for sentence in self.llm.stream(text):
                if self.bus.shutdown.is_set():
                    break
                # If the user barged in mid-generation, this turn was
                # invalidated — stop producing into a dead turn. Preserve memory:
                # save what was already said so the conversation isn't lost.
                if self.bus.current_turn_id != turn_id:
                    log.info("turn %s superseded, stop streaming", turn_id)
                    if spoken_parts:
                        store.add_exchange(text, " ".join(spoken_parts))
                        log.info("saved partial reply to memory on barge-in")
                    if perf is not None:
                        perf.trace("gemini_cancelled", reason="turn_superseded")
                    return
                if not spoke and perf is not None:
                    perf.gemini = time.perf_counter() - t0  # request → first text
                    perf.stamp("gemini_first")
                    perf.trace("gemini_first_text", took=f"{perf.gemini:.2f}s", text=repr(sentence[:80]))
                self.bus.speak(sentence)
                spoke = True
                spoken_parts.append(sentence)
            if perf is not None:
                perf.stamp("gemini_end")
                perf.trace("gemini_done", total=f"{perf.gemini_end - perf.gemini_start:.2f}s")
            if not spoke:
                self.bus.speak("Sorry sir, could you rephrase that?")
        finally:
            # END only if this turn is still current (a barge-in already
            # invalidated + cleaned up any superseded turn).
            if self.bus.current_turn_id == turn_id:
                self.bus.end_turn(turn_id)

    def _handle_filler(self, tr: Transcript):
        """FIX F: react to a user backchannel blip — but never call the LLM.

        While we're speaking it's a soft barge-in hint (user wants the floor);
        while idle it means the user is mid-thought, a cue to keep waiting rather
        than to treat silence as end-of-turn.
        """
        if self.bus.state.speaking:
            log.debug("user filler during speech — potential barge-in signal id=%s",
                      tr.utterance_id)
        else:
            log.debug("user filler during thinking — extending endpoint id=%s",
                      tr.utterance_id)
        # A user filler is a strong cue to mirror with a soft backchannel.
        if self.bus.backchannel is not None:
            self.bus.backchannel.maybe_fire("", "filler", tr.utterance_id,
                                            self.bus.state.emotion)

    def _should_route_partial(self, tr: Transcript) -> bool:
        text = tr.text.strip()
        if not text or not tr.utterance_id:
            return False
        previous = self._partial_text.get(tr.utterance_id, "")
        self._partial_text[tr.utterance_id] = text
        words = text.split()
        if len(words) < 4:
            return False
        if tr.language not in {"en", "hi", "te"} and tr.language_probability < 0.70:
            return False
        if tr.utterance_id in self._handled_utterances:
            return False
        # Need some stability: latest partial should extend or repeat the prior
        # one instead of bouncing between hallucinated hypotheses.
        if previous and not (text.startswith(previous) or previous.startswith(text)
                             or len(set(previous.lower().split()) & set(text.lower().split())) >= 3):
            return False
        if text.endswith(("?", ".", "!", "।")):
            return True
        if _EARLY_COMMAND.search(text) and len(words) >= 5:
            return True
        return False
