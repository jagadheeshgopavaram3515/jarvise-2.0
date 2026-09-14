"""
Humanizer — a thin layer between Gemini and TTS (Phase 3).

It occasionally prepends a short, natural acknowledgment to a *response* so
Jarvis feels like a considered human listener rather than a vending machine.

This is NOT the backchannel engine. Backchannel makes sounds WHILE the user is
still speaking; the Humanizer adds an occasional lead-in to Jarvis's OWN reply.

Discipline (per spec):
  * Never add a filler to every response.
  * At most one filler every 3-5 responses (randomised band).
  * Never use the same filler twice in a row.
  * Curated British phrasing only — short, calm, formal.

Unlike the backchannel listening sounds (which must be LLM-generated, never a
fixed list), these response lead-ins ARE a deliberately curated set: the spec
lists the exact phrases, and they must stay calm/formal/on-brand.
"""
from __future__ import annotations

import random

# Lead-ins for requests that need a beat of thought (questions/explanations).
_THINKING = ("One moment, sir.", "Let me think.", "Indeed.", "Quite right.")
# Lead-ins that acknowledge a statement the user just made.
_ACK = ("I see.", "Understood.", "Very well.")

# Words that suggest the user wants reasoning/teaching → a "thinking" lead-in.
_THOUGHTFUL = (
    "explain", "why", "how", "what", "describe", "tell", "story", "teach",
    "philosophy", "meaning", "difference", "should i",
)


class Humanizer:
    def __init__(self, min_gap: int = 3, max_gap: int = 5):
        self._min_gap = max(1, min_gap)
        self._max_gap = max(self._min_gap, max_gap)
        self._since = 0
        self._target = random.randint(self._min_gap, self._max_gap)
        self._last: str | None = None

    def maybe_filler(self, user_text: str) -> str | None:
        """Return a lead-in to speak before the reply, or None most of the time.

        Counts responses and only emits once the randomised 3-5 gap elapses, so
        fillers stay occasional and never back-to-back identical.
        """
        self._since += 1
        if self._since < self._target:
            return None
        self._since = 0
        self._target = random.randint(self._min_gap, self._max_gap)

        pool = _THINKING if self._looks_thoughtful(user_text) else _ACK
        # Never repeat the previous filler (across both pools).
        choices = [p for p in pool if p != self._last] or list(pool)
        filler = random.choice(choices)
        self._last = filler
        return filler

    @staticmethod
    def _looks_thoughtful(user_text: str) -> bool:
        low = (user_text or "").lower()
        return any(w in low for w in _THOUGHTFUL)
