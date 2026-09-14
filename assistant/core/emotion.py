"""Small emotion engine for voice pacing and conversational color."""
from __future__ import annotations

import time

STATES = {"happy", "curious", "serious", "excited", "empathetic"}


class EmotionEngine:
    def __init__(self):
        self.state = "curious"
        self._last_update = time.monotonic()

    def update(self, user_text: str) -> str:
        self._decay()
        low = user_text.lower()
        if any(w in low for w in ("sad", "upset", "worried", "stress", "problem",
                                  "depress", "down", "low", "tired", "exhausted",
                                  "lonely", "anxious", "not okay", "not good", "hurt",
                                  "cry", "unhappy", "miserable")):
            self.state = "empathetic"
        elif any(w in low for w in ("wow", "awesome", "great", "super", "excited")):
            self.state = "excited"
        elif any(w in low for w in ("explain", "why", "how", "what is")):
            self.state = "curious"
        elif any(w in low for w in ("urgent", "serious", "important", "issue", "error")):
            self.state = "serious"
        elif any(w in low for w in ("thanks", "thank you", "nice", "good")):
            self.state = "happy"
        self._last_update = time.monotonic()
        return self.state

    def _decay(self):
        if self.state != "curious" and time.monotonic() - self._last_update > 90:
            self.state = "curious"

    def filler_for(self, user_text: str) -> str:
        low = user_text.lower()
        if self.state == "empathetic":
            return "I see."
        if self.state == "excited":
            if any(ch in user_text for ch in "!?") or "telugu" in low:
                return "Chaala bagundi sir!"
            return "That's awesome!"
        if any(w in low for w in ("explain", "why", "how", "story")):
            return "Let me think..."
        return "Hmm..."
