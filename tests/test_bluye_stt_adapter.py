"""Offline boundary tests for the compatible Bluye desktop STT migration."""
import queue
import threading
from unittest.mock import Mock

import numpy as np

from assistant import config
from assistant.core.events import Bus, Transcript, Utterance
from assistant.core.perf import Perf
from assistant.stt.recognizer import STTService


def test_slow_transcript_consumer_preserves_backpressure_and_metadata():
    bus = Bus()
    bus.transcript_q = queue.Queue(maxsize=1)
    prior = Transcript(text="prior")
    bus.transcript_q.put(prior)
    service = STTService(bus)
    perf = Perf(t_start=1.0, speech_start=0.5)
    utt = Utterance(audio=np.zeros(480), utterance_id="origin", sequence=7,
                    during_speech=True, perf=perf, interrupt_epoch=3)
    entered, finished = threading.Event(), threading.Event()
    original_put = bus.transcript_q.put

    def observed_put(item):
        entered.set()
        original_put(item)

    bus.transcript_q.put = observed_put

    def deliver():
        service._post_transcript(utt, "నమస్కారం meeting repu", "te", 1.0)
        finished.set()

    worker = threading.Thread(target=deliver, daemon=True)
    worker.start()
    try:
        assert entered.wait(1)
        assert not finished.is_set()
        assert bus.transcript_q.get_nowait() is prior
        assert finished.wait(1)
        tr = bus.transcript_q.get_nowait()
        assert tr.text == "నమస్కారం meeting repu"
        assert (tr.utterance_id, tr.sequence, tr.during_speech, tr.interrupt_epoch) == (
            "origin", 7, True, 3)
        assert tr.perf is perf
        assert (perf.t_start, perf.speech_start) == (1.0, 0.5)
        assert (tr.is_partial, tr.kind, tr.language) == (False, "speech", "te")
        assert bus.tts_q.empty()
    finally:
        # Release a blocked producer even if an assertion fails.
        try:
            bus.transcript_q.get_nowait()
        except queue.Empty:
            pass
        worker.join(1)
        service._final_executor.shutdown(wait=True)


def test_dispatcher_initialization_selects_existing_orchestrator(monkeypatch):
    from assistant.core.dispatcher import Dispatcher
    from assistant import orchestration

    bus = Bus()
    bus.shutdown.set()  # Only initialize; no consumer loop or generation.
    instance = Mock()
    factory = Mock(return_value=instance)
    monkeypatch.setattr(config, "LLM_ROUTING_ENABLED", True)
    monkeypatch.setattr(orchestration, "get_orchestrator", factory)
    dispatcher = Dispatcher(bus, Mock())
    dispatcher.run()
    factory.assert_called_once_with()
    assert dispatcher.llm is instance
    instance.stream.assert_not_called()
