"""
Unit tests for the turn_id + END-sentinel story pipeline.

Uses a fake TTS engine (no audio / no network) so the queue/turn logic is
tested deterministically. Run with:  pytest -q   OR   python tests/test_story_pipeline.py
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant import config
config.FLAGS.demo_mode = False  # force the real (engine) path

from assistant.core.events import Bus
from assistant.tts.service import TTSService


class FakeEngine:
    """Records spoken text; respects the interrupt event like the real one."""
    def __init__(self):
        self.spoken = []

    def speak(self, text, interrupt, on_play_start=None):
        if on_play_start:
            on_play_start()
        # simulate short playback, honoring barge-in
        for _ in range(3):
            if interrupt.is_set():
                break
            time.sleep(0.01)
        self.spoken.append(text)
        return 0.01


def _make_service(start=True):
    """Create bus + TTS service. start=False lets a test queue items BEFORE the
    consumer runs, making turn/stale behaviour deterministic (no thread race)."""
    bus = Bus()
    svc = TTSService(bus)
    svc.engine = FakeEngine()
    if start:
        svc.start()
    return bus, svc.engine, svc


def _wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def _wait_done(bus, fake):
    """Turn fully processed: something was spoken, queue drained, not speaking."""
    return _wait(lambda: bool(fake.spoken)
                 and bus.tts_q.empty()
                 and not bus.state.speaking)


def _stop(bus, svc):
    bus.shutdown.set()
    svc.join(timeout=2)


# ---------------------------------------------------------------- tests
def test_long_story_no_sentence_loss():
    """A 60-sentence story is spoken fully; speaking ends only after END."""
    bus, fake, svc = _make_service()
    try:
        turn = bus.start_turn()
        sentences = [f"This is sentence number {i}." for i in range(60)]
        for s in sentences:
            bus.speak(s)
        bus.end_turn(turn)

        assert _wait_done(bus, fake), "never finished"
        joined = " ".join(fake.spoken)
        for s in sentences:
            assert s in joined, f"lost sentence: {s}"
        assert bus.current_turn_id == turn  # turn not invalidated
    finally:
        _stop(bus, svc)


def test_speaking_only_ends_on_END():
    """Empty queue must NOT end the turn — only END does."""
    bus, fake, svc = _make_service()
    try:
        turn = bus.start_turn()
        bus.speak("Only sentence for now.")
        # Let TTS speak it and then sit on an empty queue.
        assert _wait(lambda: fake.spoken and "Only sentence" in fake.spoken[-1])
        time.sleep(0.3)
        assert bus.state.speaking, "speaking wrongly cleared on empty queue"
        # Now send END → must finish.
        bus.end_turn(turn)
        assert _wait(lambda: not bus.state.speaking), "END did not finish turn"
    finally:
        _stop(bus, svc)


def test_stale_items_dropped():
    """Items from an old turn are dropped once the turn advances."""
    bus, fake, svc = _make_service(start=False)  # queue first, then consume
    try:
        bus.start_turn()
        bus.speak("ALPHA stale one")
        bus.speak("ALPHA stale two")
        # Advance to a new turn WITHOUT draining — old items are now stale.
        turn_b = bus.start_turn()
        bus.speak("BRAVO fresh")
        bus.end_turn(turn_b)

        svc.start()  # now the consumer runs against a fully-staged queue
        assert _wait_done(bus, fake)
        joined = " ".join(fake.spoken)
        assert "BRAVO" in joined
        assert "ALPHA" not in joined, "stale items were spoken"
    finally:
        _stop(bus, svc)


def test_barge_in_cleanup():
    """request_interrupt clears the queue, stops, and invalidates the turn."""
    bus, fake, svc = _make_service(start=False)
    try:
        turn = bus.start_turn()
        for i in range(20):
            bus.speak(f"long narration part {i}")
        # Barge-in before the consumer runs.
        bus.request_interrupt()
        assert bus.current_turn_id is None
        assert not bus.state.speaking
        assert bus.tts_q.empty(), "queue not cleared on barge-in"

        # A late END from the killed turn must NOT revive speaking.
        svc.start()
        bus.end_turn(turn)
        time.sleep(0.3)
        assert not bus.state.speaking
        assert not fake.spoken, "spoke items from an interrupted turn"
    finally:
        _stop(bus, svc)


def test_rapid_turn_switching():
    """Three quick turns; only the newest survives, no cross-talk."""
    bus, fake, svc = _make_service(start=False)
    try:
        for label in ("FIRST", "SECOND"):
            bus.start_turn()
            bus.speak(f"{label} reply")
            bus.request_interrupt()       # immediately superseded (drains queue)
        turn = bus.start_turn()
        bus.speak("THIRD reply")
        bus.end_turn(turn)

        svc.start()
        assert _wait_done(bus, fake)
        joined = " ".join(fake.spoken)
        assert "THIRD" in joined
        assert "FIRST" not in joined and "SECOND" not in joined
    finally:
        _stop(bus, svc)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    import traceback
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}:\n{traceback.format_exc()}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
