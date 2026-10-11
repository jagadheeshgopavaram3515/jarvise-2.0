"""Bounded final admission; mocked providers and synthetic audio only."""
import threading
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from assistant import config
from assistant.core.events import Bus, Utterance
from assistant.stt import recognizer
from assistant.stt.recognizer import STTService, _has_real_speech
from assistant.stt.cloud import GeminiAudioSTT


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(config.FLAGS, "demo_mode", False)
    monkeypatch.setattr(config, "STT_VAD_PREFILTER", False)
    svc = STTService(Bus())
    svc._cloud = Mock(in_flight=False)
    svc._cloud.transcribe.return_value = "hello"
    monkeypatch.setattr(recognizer, "FasterWhisperSTT", Mock(side_effect=AssertionError("Whisper forbidden")))
    monkeypatch.setattr(recognizer, "GeminiAudioSTT", Mock(return_value=svc._cloud))
    yield svc
    svc.bus.shutdown.set()
    if svc.is_alive():
        svc.join(2)
    svc._final_executor.shutdown(wait=True)


def utterance(**kwargs):
    return Utterance(audio=np.full(1600, 0.1, dtype=np.float32), **kwargs)


@pytest.fixture
def cloud(monkeypatch):
    provider = GeminiAudioSTT.__new__(GeminiAudioSTT)
    provider._client = SimpleNamespace(models=SimpleNamespace(generate_content=Mock()))
    provider._types = SimpleNamespace(Part=SimpleNamespace(from_bytes=Mock(return_value="audio")))
    provider.model = "mock-model"
    provider._executor = ThreadPoolExecutor(max_workers=1)
    provider._request_lock = threading.Lock()
    provider._inflight = None
    provider._inflight_id = ""
    provider._closed = False
    monkeypatch.setattr(config, "HYBRID_STT_TIMEOUT_S", 0.05)
    yield provider
    provider._executor.shutdown(wait=True)


def block_request(cloud):
    started, release = threading.Event(), threading.Event()
    def request(**kwargs):
        started.set()
        assert release.wait(3)
        return SimpleNamespace(text="expired")
    cloud._client.models.generate_content.side_effect = request
    return started, release


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline
        threading.Event().wait(0.005)


def start_blocked(service, monkeypatch):
    started, release = threading.Event(), threading.Event()
    def request(audio, *, utterance_id):
        if utterance_id == "first":
            started.set()
            assert release.wait(3)
        return utterance_id
    service._cloud.transcribe.side_effect = request
    service.start()
    assert service.initialized.wait(1)
    service.bus.stt_in_q.put(utterance(utterance_id="first"))
    assert started.wait(1)
    return release


def test_fifo_ids_duplicates_and_bounded_executor(service, monkeypatch):
    release = start_blocked(service, monkeypatch)
    try:
        for name in ("second", "third", "third"):
            service.bus.stt_in_q.put(utterance(utterance_id=name))
        wait_for(lambda: service.diagnostic_counts["queued"] == 4)
        assert service._final_executor._work_queue.qsize() == 0
        release.set()
        results = [service.bus.transcript_q.get(timeout=1) for _ in range(3)]
        assert [(tr.utterance_id, tr.text) for tr in results] == [
            (name, name) for name in ("first", "second", "third")]
        wait_for(lambda: service.diagnostic_counts["duplicate"] == 1)
        assert service._cloud.transcribe.call_count == 3
    finally:
        release.set()


def test_capacity_rejects_newest_with_counter(service, monkeypatch):
    monkeypatch.setattr(config, "STT_FINAL_QUEUE_CAPACITY", 2)
    release = start_blocked(service, monkeypatch)
    try:
        for name in ("second", "third", "overflow"):
            service.bus.stt_in_q.put(utterance(utterance_id=name))
        wait_for(lambda: service.diagnostic_counts["queue_overflow"] == 1)
        assert len(service._pending) == 2
        release.set()
        assert [service.bus.transcript_q.get(timeout=1).utterance_id for _ in range(3)] == [
            "first", "second", "third"]
        assert service._cloud.transcribe.call_count == 3
    finally:
        release.set()


def test_expiry_without_perf_prevents_late_command(service, monkeypatch):
    monkeypatch.setattr(config, "STT_FINAL_QUEUE_MAX_AGE_S", 0.05)
    release = start_blocked(service, monkeypatch)
    try:
        service.bus.stt_in_q.put(utterance(utterance_id="old-command"))
        wait_for(lambda: service.diagnostic_counts["queue_expired"] == 1)
        release.set()
        assert service.bus.transcript_q.get(timeout=1).utterance_id == "first"
        assert service._cloud.transcribe.call_count == 1
    finally:
        release.set()


def test_interrupt_cancels_pending_and_inflight_even_after_event_clear(service, monkeypatch):
    release = start_blocked(service, monkeypatch)
    try:
        service.bus.stt_in_q.put(utterance(utterance_id="cancelled-command"))
        wait_for(lambda: service.diagnostic_counts["queued"] == 2)
        service.bus.request_interrupt()
        service.bus.interrupt.clear()  # TTS does this; cancellation must survive
        release.set()
        wait_for(lambda: service.diagnostic_counts["cancelled"] == 2)
        assert service.bus.transcript_q.empty()
        service.bus.stt_in_q.put(utterance(utterance_id="fresh"))
        assert service.bus.transcript_q.get(timeout=1).utterance_id == "fresh"
        assert service._cloud.transcribe.call_count == 2
    finally:
        release.set()


def test_capture_epoch_cancels_ingress_backlog(service):
    old = utterance(interrupt_epoch=service.bus.interrupt_epoch)
    service.bus.request_interrupt()
    service._enqueue_final(old)
    assert service.diagnostic_counts["cancelled"] == 1
    service._cloud.transcribe.assert_not_called()


def test_shutdown_drains_pending_without_late_delivery(service, monkeypatch):
    release = start_blocked(service, monkeypatch)
    try:
        service.bus.stt_in_q.put(utterance(utterance_id="pending"))
        wait_for(lambda: service.diagnostic_counts["queued"] == 2)
        service.bus.shutdown.set()
        service.join(1)
        assert not service.is_alive()
        assert not service._pending
        release.set()
        wait_for(lambda: service._active.done())
        assert service.bus.transcript_q.empty()
        assert service.diagnostic_counts["shutdown"] == 2
    finally:
        release.set()


def test_timeout_pending_waits_for_sdk_completion_then_recovers(service, cloud, monkeypatch):
    started, release = block_request(cloud)
    service._cloud = cloud
    service.initialized.set()
    service.start()
    try:
        assert service.initialized.wait(1)
        service.bus.stt_in_q.put(utterance(utterance_id="timed-out"))
        assert started.wait(1)
        wait_for(lambda: service.diagnostic_counts["no_result"] == 1)
        service.bus.stt_in_q.put(utterance(utterance_id="waiting"))
        wait_for(lambda: service.diagnostic_counts["queued"] == 2)
        assert cloud._client.models.generate_content.call_count == 1
        assert cloud._executor._work_queue.qsize() == 0
        # The old invocation already captured its side effect.
        cloud._client.models.generate_content.side_effect = None
        cloud._client.models.generate_content.return_value = SimpleNamespace(text="fresh")
        release.set()
        tr = service.bus.transcript_q.get(timeout=1)
        assert (tr.utterance_id, tr.text) == ("waiting", "fresh")
        assert service.bus.transcript_q.empty()
        assert cloud._client.models.generate_content.call_count == 2
    finally:
        release.set()
        service.bus.shutdown.set()
        service.join(1)


@pytest.mark.parametrize("amplitude,accepted", [(0, False), (0.0049, False), (0.0051, True)])
def test_prefilter_boundary_is_unchanged(service, monkeypatch, caplog, amplitude, accepted):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(config, "STT_VAD_PREFILTER", True)
    monkeypatch.setattr(config, "STT_PREFILTER_RMS", 0.005)
    # Synthetic short acknowledgement: amplitude boundary, not a linguistic claim.
    audio = np.full(1600, amplitude, dtype=np.float32)
    assert _has_real_speech(audio, energy_only=True) == accepted
    service._transcribe_and_emit(Utterance(audio=audio, utterance_id="boundary"))
    assert service._cloud.transcribe.call_count == int(accepted)
    if not accepted:
        assert service.diagnostic_counts["no_speech"] == 1
        assert "rms_below_threshold" in caplog.text
        assert "audio_ms" in caplog.text and "threshold" in caplog.text


def test_empty_prefilter_diagnostic(service, monkeypatch):
    monkeypatch.setattr(config, "STT_VAD_PREFILTER", True)
    service._transcribe_and_emit(Utterance(audio=np.array([], dtype=np.float32)))
    assert service.diagnostic_counts["no_speech"] == 1
    service._cloud.transcribe.assert_not_called()


def test_noise_does_not_fill_waiting_queue(service, monkeypatch):
    monkeypatch.setattr(config, "STT_VAD_PREFILTER", True)
    for i in range(10):
        service._enqueue_final(Utterance(audio=np.zeros(1600), utterance_id=str(i)))
    assert not service._pending
    assert service.diagnostic_counts["no_speech"] == 10
    service._cloud.transcribe.assert_not_called()


def test_delivered_but_not_dispatched_result_is_cancelled(service, monkeypatch):
    from assistant.core.dispatcher import Dispatcher
    from assistant.commands import handlers
    service._transcribe_and_emit(utterance(utterance_id="command"))
    tr = service.bus.transcript_q.get_nowait()
    service.bus.request_interrupt()
    service.bus.interrupt.clear()
    handler = Mock(side_effect=AssertionError("Cancelled command must not execute"))
    monkeypatch.setattr(handlers, "handle", handler)
    dispatcher = Dispatcher(service.bus, Mock())
    dispatcher._route(tr)
    handler.assert_not_called()


def test_capture_preserves_start_generation_after_interruption(service):
    from assistant.audio.capture import AudioInputService
    capture = AudioInputService(service.bus)
    epoch = service.bus.interrupt_epoch
    service.bus.request_interrupt()
    pcm = np.full(480, 3000, dtype=np.int16).tobytes()
    capture._emit([pcm] * 10, time.perf_counter(), time.perf_counter(), False,
                  "capture", 1, interrupt_epoch=epoch)
    service._enqueue_final(service.bus.stt_in_q.get_nowait())
    assert service.diagnostic_counts["cancelled"] == 1
    service._cloud.transcribe.assert_not_called()


def test_silence_padding_can_dilute_short_speech_without_threshold_change(service, monkeypatch):
    monkeypatch.setattr(config, "STT_PREFILTER_RMS", 0.005)
    short = np.full(1600, 0.006, dtype=np.float32)
    assert _has_real_speech(short, energy_only=True)
    padded = np.concatenate((short, np.zeros(6400, dtype=np.float32)))
    assert not _has_real_speech(padded, energy_only=True)


@pytest.mark.parametrize("capacity,age,expected_capacity,expected_age", [
    ("bad", "bad", 4, 8), ("0", "0", 1, 8), ("100", "nan", 16, 8),
    ("2", "0.1", 2, 0.1), ("4", "121", 4, 8),
])
def test_queue_configuration_is_bounded(monkeypatch, capacity, age, expected_capacity, expected_age):
    import importlib.util
    import sys
    from pathlib import Path
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("STT_FINAL_QUEUE_CAPACITY", capacity)
    monkeypatch.setenv("STT_FINAL_QUEUE_MAX_AGE_S", age)
    spec = importlib.util.spec_from_file_location("queue_config", Path(config.__file__))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert module.STT_FINAL_QUEUE_CAPACITY == expected_capacity
    assert module.STT_FINAL_QUEUE_MAX_AGE_S == expected_age
