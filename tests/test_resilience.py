"""
Failure-injection / resilience tests (Phase 8).

Each test deliberately breaks a subsystem (synthesis, transcription, a tool, the
filesystem, a worker thread) and asserts Jarvis keeps running: threads survive,
failures stay local, and recovery happens automatically. No network, no mic, no
real model — everything is injected.

Run with:   pytest -q tests/test_resilience.py
       or:  python tests/test_resilience.py
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant import config  # noqa: E402
config.FLAGS.demo_mode = False  # exercise the real engine paths

from assistant.core import resilience as R  # noqa: E402
from assistant.core.health import HealthMonitor  # noqa: E402
from assistant.core.events import Bus  # noqa: E402


def _wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


# ===================================================== Phase 6: safeguards ====
def test_safe_call_swallows_and_defaults():
    assert R.safe_call(lambda: 1 / 0, default="ok", label="div") == "ok"
    assert R.safe_call(lambda x: x + 1, 41) == 42


def test_safe_decorator():
    @R.safe(default=-1, label="boom")
    def boom():
        raise RuntimeError("x")
    assert boom() == -1


def test_retry_succeeds_then_gives_up():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "done"
    assert R.retry(flaky, attempts=5, delay=0, label="flaky") == "done"

    def always():
        raise ValueError("permanent")
    assert R.retry(always, attempts=2, delay=0, default="fallback") == "fallback"


def test_safe_json_load_and_dump(tmp_path=None):
    import tempfile
    d = tempfile.mkdtemp(prefix="jarvis_res_")
    good = os.path.join(d, "good.json")
    assert R.safe_json_dump({"a": 1}, good) is True
    assert R.safe_json_load(good) == {"a": 1}
    # Missing file → default, no raise.
    assert R.safe_json_load(os.path.join(d, "missing.json"), default={}) == {}
    # Corrupt file → default, no raise.
    bad = os.path.join(d, "bad.json")
    with open(bad, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    assert R.safe_json_load(bad, default="DEF") == "DEF"


# ============================================ Phase 2: thread survivability ====
def test_supervise_loop_survives_every_exception():
    """A loop whose body always throws must keep looping, never escape."""
    stop = threading.Event()
    counter = {"n": 0}

    def body():
        counter["n"] += 1
        raise RuntimeError(f"iteration {counter['n']} boom")

    t = threading.Thread(
        target=R.supervise_loop,
        args=("Test", stop.is_set, body),
        kwargs={"idle_sleep": 0.01},
        daemon=True,
    )
    t.start()
    assert _wait(lambda: counter["n"] >= 5), "loop died on first exception"
    assert t.is_alive(), "supervised thread should still be alive"
    stop.set()
    t.join(timeout=2)
    assert not t.is_alive()


# ================================ Phase 3/4: health monitor + auto-recovery ====
def _dead_thread():
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    return t


def test_health_detects_and_restarts_critical():
    bus = Bus()
    mon = HealthMonitor(bus)
    made = {"n": 0}

    def factory():
        made["n"] += 1
        t = threading.Thread(target=lambda: time.sleep(10), name="svc", daemon=True)
        t.start()
        return t

    mon.register("Svc", _dead_thread(), factory, critical=True)
    mon._check_once()
    assert made["n"] == 1, "dead critical service was not restarted"
    assert mon.status()["Svc"] is True


def test_health_skips_noncritical():
    bus = Bus()
    mon = HealthMonitor(bus)
    made = {"n": 0}

    def factory():
        made["n"] += 1
        return threading.Thread(target=lambda: None)

    mon.register("Opt", _dead_thread(), factory, critical=False)
    mon._check_once()
    assert made["n"] == 0, "non-critical service should NOT be auto-restarted"


def test_health_respects_max_restart_cap():
    bus = Bus()
    mon = HealthMonitor(bus)
    made = {"n": 0}

    def factory():
        made["n"] += 1
        return _dead_thread()   # immediately dead → forces another restart attempt

    mon.register("Flap", _dead_thread(), factory, critical=True)
    # Far more checks than the cap; restarts must stop at HEALTH_MAX_RESTARTS.
    for _ in range(config.HEALTH_MAX_RESTARTS + 5):
        mon._check_once()
    assert made["n"] == config.HEALTH_MAX_RESTARTS, \
        f"expected {config.HEALTH_MAX_RESTARTS} restarts, got {made['n']}"


# ======================================= Phase 5/9: TTS failure isolation =====
def test_tts_synth_failure_degrades_to_silence():
    """Engine-level: a synth exception becomes empty audio, never raises."""
    from assistant.tts import engine
    e = engine.EdgeStreamingTTS()
    e.synthesize_bytes = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("NoAudioReceived"))
    mp3, dt = e.synthesize_for_playback("hello there friend")
    assert mp3 == b"", "synth failure should degrade to empty audio"


def test_tts_thread_survives_bad_synthesis():
    """Service-level: a synth that always raises must NOT kill the TTS thread;
    the turn still completes and `speaking` clears via the END sentinel."""
    from assistant.tts.service import TTSService

    class ExplodingEngine:
        ref_sink = None

        def synthesize_for_playback(self, text, emotion="curious"):
            raise RuntimeError("synthesis exploded")

        def play_synthesized(self, *a, **k):
            pass

    bus = Bus()
    svc = TTSService(bus)
    svc.engine = ExplodingEngine()
    svc.start()
    try:
        turn = bus.start_turn()
        bus.speak("this will fail to synthesize")
        bus.end_turn(turn)
        # Thread must stay alive and the turn must finish (speaking → False).
        assert _wait(lambda: not bus.state.speaking and bus.tts_q.empty()), \
            "TTS turn never completed after a synthesis failure"
        assert svc.is_alive(), "TTS thread died on a bad synthesis"
    finally:
        bus.shutdown.set()
        svc.join(timeout=2)


# ============================== Phase 5: STT failure isolation (no thread death)=
def test_stt_transcribe_failure_is_contained():
    """A transcription exception is caught inside _transcribe_and_emit and never
    propagates (so the STT thread / executor lane stays alive)."""
    from assistant.stt.recognizer import STTService
    from assistant.core.events import Utterance
    import numpy as np

    bus = Bus()
    svc = STTService.__new__(STTService)   # avoid heavy __init__ (no model load)
    svc.bus = bus
    svc._cloud = None

    class BoomEngine:
        def transcribe(self, *a, **k):
            raise RuntimeError("whisper exploded")
    svc.engine = BoomEngine()

    utt = Utterance(audio=np.zeros(1600, dtype=np.float32), is_partial=False,
                    utterance_id="x1")
    # Must NOT raise — the method swallows the transcribe error.
    svc._transcribe_and_emit(utt)
    assert bus.transcript_q.empty(), "a failed transcription should emit nothing"


# ============================ Phase 5: tool failure isolation (conversation) ===
def test_tool_failure_does_not_break_conversation():
    """A tool that fails still produces a spoken reply and never raises into the
    dispatcher (so conversation continues)."""
    from assistant.tools.registry import ToolRegistry
    from assistant.tools.executor import ToolExecutor
    from assistant.tools.schemas import ToolSpec

    reg = ToolRegistry()
    reg.register_tool(ToolSpec(name="bad", description="bad",
                               handler=lambda: (_ for _ in ()).throw(RuntimeError("tool boom"))))
    ex = ToolExecutor(reg, timeout=2)
    res = ex.execute("bad", {})
    assert res["success"] is False and "message" in res, "tool failure not contained"


# ============================ Phase 9: memory/file corruption degradation ======
def test_memory_corruption_degrades_gracefully():
    """A corrupt memory JSON returns the default instead of crashing a caller."""
    import tempfile
    d = tempfile.mkdtemp(prefix="jarvis_mem_")
    path = os.path.join(d, "memory.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("}}corrupt{{")
    assert R.safe_json_load(path, default={"topics": []}) == {"topics": []}


# ------------------------------------------------------------ script runner ---
if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}:\n{traceback.format_exc()}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
