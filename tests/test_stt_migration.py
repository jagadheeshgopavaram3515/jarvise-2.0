"""Gemini-only STT integration contracts. No network or local model loading."""
import builtins
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from assistant import config
from assistant.core.events import Bus, Utterance
from assistant.core.perf import Perf
from assistant.stt import recognizer
from assistant.stt.recognizer import STTService, _lang_from_script


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(config.FLAGS, "demo_mode", False)
    monkeypatch.setattr(config, "STT_VAD_PREFILTER", False)
    svc = STTService(Bus())
    svc._cloud = Mock(in_flight=False)
    svc._cloud.transcribe.return_value = "hello"
    # Trap any attempt to use the retired provider, including lazy init.
    monkeypatch.setattr(recognizer, "FasterWhisperSTT", Mock(side_effect=AssertionError("Whisper forbidden")))
    monkeypatch.setattr(recognizer, "GeminiAudioSTT", Mock(return_value=svc._cloud))
    yield svc
    svc.bus.shutdown.set()
    if svc.is_alive():
        svc.join(2)
    svc._final_executor.shutdown(wait=True)


def utterance(**kwargs):
    return Utterance(audio=np.ones(1600, dtype=np.float32) * 0.1,
                     utterance_id=kwargs.pop("utterance_id", "final-1"), **kwargs)


@pytest.mark.parametrize("text", ["stop", "a", "7", "打开", "నువ్వు ఎలా ఉన్నావు", "तुम कैसे हो", "வணக்கம்"])
def test_every_valid_final_uses_gemini_and_preserves_contract(service, text):
    service._cloud.transcribe.return_value = text
    perf = Perf(t_start=time.perf_counter())
    utt = utterance(during_speech=True, sequence=4, perf=perf)
    service._transcribe_and_emit(utt)
    tr = service.bus.transcript_q.get_nowait()
    assert (tr.text, tr.utterance_id, tr.sequence, tr.during_speech, tr.is_partial, tr.kind) == (
        text, "final-1", 4, True, False, "speech")
    assert tr.perf is perf
    assert tr.language == _lang_from_script(text)
    service._cloud.transcribe.assert_called_once_with(utt.audio, utterance_id="final-1")
    recognizer.FasterWhisperSTT.assert_not_called()


def test_no_whisper_initialization_even_when_old_config_requests_local(service, monkeypatch):
    monkeypatch.setattr(config, "STT_PRIMARY", "local")
    monkeypatch.setattr(config, "HYBRID_STT", False)
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith(("faster_whisper", "ctranslate2", "silero_vad")):
            raise AssertionError("local STT import forbidden")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    service._initialize()
    assert service.initialized.is_set() and service.bus.stt_ready.is_set()
    assert service.transcribe_blocking(utterance().audio) == ("hello", "en")
    recognizer.FasterWhisperSTT.assert_not_called()


def test_partial_does_not_transcribe_or_invent_interrupt_text(service):
    service._transcribe_and_emit(utterance(is_partial=True, during_speech=True))
    service._cloud.transcribe.assert_not_called()
    assert service.bus.transcript_q.empty()


@pytest.mark.parametrize("result", [None, "", "   "])
def test_failure_emits_nothing_and_later_final_recovers(service, result):
    service._cloud.transcribe.side_effect = [result, "recovered"]
    service._transcribe_and_emit(utterance())
    assert service.bus.transcript_q.empty() and service.bus.tts_q.empty()
    service._transcribe_and_emit(utterance(utterance_id="final-2"))
    tr = service.bus.transcript_q.get_nowait()
    assert (tr.text, tr.utterance_id) == ("recovered", "final-2")
    assert service._cloud.transcribe.call_count == 2


def test_exception_is_contained_and_recovers(service):
    service._cloud.transcribe.side_effect = [RuntimeError("SDK error"), "recovered"]
    service._transcribe_and_emit(utterance())
    assert service.bus.transcript_q.empty()
    service._transcribe_and_emit(utterance(utterance_id="next"))
    assert service.bus.transcript_q.get_nowait().utterance_id == "next"


def test_duplicate_final_never_submits_twice(service):
    utt = utterance()
    service._transcribe_and_emit(utt)
    service._transcribe_and_emit(utt)
    service._cloud.transcribe.assert_called_once()
    assert service.bus.transcript_q.qsize() == 1


def test_stale_final_is_dropped_before_sdk(service):
    perf = Perf(t_start=time.perf_counter())
    perf.stt_queued = time.perf_counter() - config.STT_MAX_STALE_S - 1
    service._transcribe_and_emit(utterance(perf=perf))
    service._cloud.transcribe.assert_not_called()
    assert service.bus.transcript_q.empty()


def test_shutdown_never_delivers_late_result(service):
    def result(*args, **kwargs):
        service.bus.shutdown.set()
        return "late"
    service._cloud.transcribe.side_effect = result
    service._transcribe_and_emit(utterance())
    assert service.bus.transcript_q.empty()


def test_blocking_compatibility_no_result(service):
    service._cloud.transcribe.return_value = None
    assert service.transcribe_blocking(utterance().audio) == ("", None)


def test_initialization_failure_signals_completion_without_false_readiness(service, monkeypatch):
    service._cloud = None
    monkeypatch.setattr(recognizer, "GeminiAudioSTT", Mock(side_effect=RuntimeError("init failed")))
    service._initialize()
    assert service.initialized.is_set()
    assert not service.bus.stt_ready.is_set()
    service._transcribe_and_emit(utterance())
    assert service.bus.transcript_q.empty()
    assert service.transcribe_blocking(utterance().audio) == ("", None)
    recognizer.GeminiAudioSTT.assert_called_once()


def test_filler_signal_preserved(service):
    service._forward_filler(utterance(kind="filler", during_speech=True))
    tr = service.bus.transcript_q.get_nowait()
    assert (tr.kind, tr.text, tr.utterance_id, tr.during_speech) == ("filler", "", "final-1", True)
    service._cloud.transcribe.assert_not_called()


def test_running_service_queues_overlap_without_executor_queue(service, monkeypatch):
    started, release, queued = threading.Event(), threading.Event(), threading.Event()
    def request(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return "first"
    service._cloud.transcribe.side_effect = request
    original = service._diagnostic
    def diagnostic(utt, category, **kwargs):
        original(utt, category, **kwargs)
        if category == "queued" and utt.utterance_id == "overlap":
            queued.set()
    monkeypatch.setattr(service, "_diagnostic", diagnostic)
    service.start()
    try:
        assert service.initialized.wait(1)
        service.bus.stt_in_q.put(utterance())
        assert started.wait(1)
        service.bus.stt_in_q.put(utterance(utterance_id="overlap"))
        assert queued.wait(1)
        assert service._final_executor._work_queue.qsize() == 0
        service._cloud.transcribe.assert_called_once()
        release.set()
        assert service.bus.transcript_q.get(timeout=1).utterance_id == "final-1"
        assert service.bus.transcript_q.get(timeout=1).utterance_id == "overlap"
        # Wait for the final lane to release without timing-based sleeps.
        assert service._final_slot.acquire(timeout=1)
        service._final_slot.release()
        service.bus.stt_in_q.put(utterance(utterance_id="next"))
        assert service.bus.transcript_q.get(timeout=1).utterance_id == "next"
        assert service.is_alive()
    finally:
        release.set()
        service.bus.shutdown.set()
        service.join(2)


def test_endpoint_to_gemini_to_dispatcher_interruption(service, monkeypatch):
    from assistant.audio.capture import AudioInputService
    from assistant.core.dispatcher import Dispatcher
    monkeypatch.setattr(config, "FULL_DUPLEX", True)
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINTING", False)
    monkeypatch.setattr(config, "VAD_SILENCE_MS", 60)
    monkeypatch.setattr(config, "VAD_SILENCE_GROW_MS_PER_SEC", 0)
    capture = AudioInputService(service.bus)
    capture.aec.process_audio = lambda frame, **kwargs: frame
    capture.vad.is_speech = Mock(side_effect=[True] * 12 + [False] * 2)
    pcm = (np.ones(480, dtype=np.int16) * 3000).tobytes()
    for _ in range(14):
        capture._ring.put(pcm)
    emit = capture._emit
    def endpoint(*args, **kwargs):
        emit(*args, **kwargs)
        service.bus.shutdown.set()
    monkeypatch.setattr(capture, "_emit", endpoint)
    service.bus.state.set_speaking(True)
    capture._segment_loop()
    service.bus.shutdown.clear()
    utt = service.bus.stt_in_q.get_nowait()
    assert not utt.is_partial and utt.during_speech and utt.utterance_id
    service._cloud.transcribe.return_value = "stop"
    service._transcribe_and_emit(utt)
    tr = service.bus.transcript_q.get_nowait()
    assert tr.utterance_id == utt.utterance_id
    dispatcher = Dispatcher(service.bus, Mock())
    dispatcher._route(tr)
    assert service.bus.interrupt.is_set()
    assert service.bus.transcript_q.empty()


def test_short_command_dispatch_uses_existing_handler(service, monkeypatch):
    from assistant.core.dispatcher import Dispatcher
    from assistant.memory import store
    monkeypatch.setattr(store, "remember_mood", lambda text: None)
    service._cloud.transcribe.return_value = "time please"
    service._transcribe_and_emit(utterance())
    dispatcher = Dispatcher(service.bus, Mock())
    dispatcher._route(service.bus.transcript_q.get_nowait())
    assert any("time" in item.text.lower() for item in list(service.bus.tts_q.queue))
    assert service.bus.tts_q.queue[-1].is_end
