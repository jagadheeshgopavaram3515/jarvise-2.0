import importlib.util
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from assistant import config
from assistant.stt.cloud import GeminiAudioSTT
from assistant.stt.recognizer import STTService
from assistant.core.events import Bus, Utterance


@pytest.fixture
def cloud(monkeypatch):
    # Exercise the real wrapper and encoding without constructing an SDK client.
    provider = GeminiAudioSTT.__new__(GeminiAudioSTT)
    provider._client = SimpleNamespace(models=SimpleNamespace(generate_content=Mock()))
    provider._types = SimpleNamespace(Part=SimpleNamespace(from_bytes=Mock(return_value='audio')))
    provider.model = 'mock-model'
    provider._executor = ThreadPoolExecutor(max_workers=1)
    provider._request_lock = threading.Lock()
    provider._inflight = None
    provider._inflight_id = ""
    provider._closed = False
    monkeypatch.setattr(config, 'HYBRID_STT_TIMEOUT_S', 0.05)
    yield provider
    provider._executor.shutdown(wait=True)


AUDIO = np.ones(160, dtype=np.float32)


def block_request(cloud):
    started, release = threading.Event(), threading.Event()
    def request(**kwargs):
        started.set()
        assert release.wait(3), 'test failed to release SDK mock'
        return SimpleNamespace(text='old utterance')
    cloud._client.models.generate_content.side_effect = request
    return started, release


def test_fast_success_and_interface(cloud):
    cloud._client.models.generate_content.return_value = SimpleNamespace(text='  expected transcript  ')
    assert cloud.transcribe(AUDIO) == 'expected transcript'
    cloud._client.models.generate_content.assert_called_once()


def test_slow_success_within_configured_timeout(cloud, monkeypatch):
    monkeypatch.setattr(config, 'HYBRID_STT_TIMEOUT_S', 2)
    started, release = block_request(cloud)
    with ThreadPoolExecutor(max_workers=1) as callers:
        call = callers.submit(cloud.transcribe, AUDIO)
        try:
            assert started.wait(1)
            assert not call.done()
        finally:
            release.set()
        assert call.result(1) == 'old utterance'


def test_timeout_no_queue_recovery_and_late_isolation(cloud):
    started, release = block_request(cloud)
    try:
        before = time.monotonic()
        assert cloud.transcribe(AUDIO) is None
        assert 0.03 <= time.monotonic() - before < 1
        assert started.is_set()
        old = cloud._inflight
        assert not old.done()
        for _ in range(20):
            assert cloud.transcribe(AUDIO) is None
        assert cloud._client.models.generate_content.call_count == 1
        release.set()
        assert old.result(1) == 'old utterance'
        cloud._client.models.generate_content.side_effect = None
        cloud._client.models.generate_content.return_value = SimpleNamespace(text='new utterance')
        assert cloud.transcribe(AUDIO) == 'new utterance'
        assert cloud._client.models.generate_content.call_count == 2
    finally:
        release.set()


def test_concurrent_calls_only_one_outstanding(cloud, monkeypatch):
    monkeypatch.setattr(config, 'HYBRID_STT_TIMEOUT_S', 2)
    started, release = block_request(cloud)
    barrier = threading.Barrier(9)
    def call():
        barrier.wait(2)
        return cloud.transcribe(AUDIO)
    with ThreadPoolExecutor(max_workers=8) as callers:
        futures = [callers.submit(call) for _ in range(8)]
        barrier.wait(2)
        try:
            assert started.wait(1)
            # Seven occupied callers must return while the owner still waits.
            from concurrent.futures import wait
            done, pending = wait(futures, timeout=0.5)
            assert len(done) == 7
            assert all(f.result() is None for f in done)
            assert len(pending) == 1
            assert cloud._client.models.generate_content.call_count == 1
        finally:
            release.set()
        assert [f.result(1) for f in pending] == ['old utterance']


def test_exception_recovers(cloud):
    cloud._client.models.generate_content.side_effect = [RuntimeError('SDK failure'), SimpleNamespace(text='recovered')]
    assert cloud.transcribe(AUDIO) is None
    assert cloud.transcribe(AUDIO) == 'recovered'


def test_submission_failure_recovers(cloud):
    original = cloud._executor.submit
    cloud._executor.submit = Mock(side_effect=RuntimeError('submission failed'))
    assert cloud.transcribe(AUDIO) is None
    assert cloud._inflight is None
    cloud._executor.submit = original
    cloud._client.models.generate_content.return_value = SimpleNamespace(text='recovered')
    assert cloud.transcribe(AUDIO) == 'recovered'


@pytest.mark.parametrize('text', ['', '   ', None])
def test_empty_response(cloud, text):
    cloud._client.models.generate_content.return_value = SimpleNamespace(text=text)
    assert cloud.transcribe(AUDIO) is None


def make_service(cloud, monkeypatch):
    monkeypatch.setattr(config, 'STT_PRIMARY', 'cloud')
    monkeypatch.setattr(config, 'STT_CLOUD_FALLBACK', True)
    monkeypatch.setattr(config, 'STT_VAD_PREFILTER', False)
    service = STTService(Bus())
    service._cloud = cloud
    service.engine = Mock()
    return service


def test_occupied_cloud_drops_without_whisper(cloud, monkeypatch):
    started, release = block_request(cloud)
    service = make_service(cloud, monkeypatch)
    try:
        assert cloud.transcribe(AUDIO) is None
        before = time.monotonic()
        service._transcribe_and_emit(Utterance(audio=AUDIO, utterance_id='new-id'))
        assert time.monotonic() - before < 1
        assert service.bus.transcript_q.empty()
        service.engine.transcribe.assert_not_called()
        assert cloud._client.models.generate_content.call_count == 1
        release.set()
        cloud._inflight.result(1)
        assert service.bus.transcript_q.empty()
    finally:
        release.set()
        service._final_executor.shutdown(wait=True)


def test_no_second_cloud_attempt_or_whisper_after_empty_result(cloud, monkeypatch):
    service = make_service(cloud, monkeypatch)
    cloud._client.models.generate_content.return_value = SimpleNamespace(text='')
    service.engine.transcribe.return_value = None
    try:
        service._transcribe_and_emit(Utterance(audio=AUDIO, utterance_id='failed'))
        assert cloud._client.models.generate_content.call_count == 1
        service.engine.transcribe.assert_not_called()
        assert service.bus.transcript_q.empty()
    finally:
        service._final_executor.shutdown(wait=True)


def test_cloud_primary_does_not_load_silero(cloud, monkeypatch):
    service = make_service(cloud, monkeypatch)
    monkeypatch.setattr(config, 'STT_VAD_PREFILTER', True)
    monkeypatch.setattr('assistant.stt.recognizer._get_silero',
                        Mock(side_effect=AssertionError('Silero must not block cloud STT')))
    cloud._client.models.generate_content.return_value = SimpleNamespace(text='hello Jarvis')
    try:
        service._transcribe_and_emit(Utterance(audio=AUDIO, utterance_id='cloud-vad'))
        assert service.bus.transcript_q.get_nowait().text == 'hello Jarvis'
        service.engine.transcribe.assert_not_called()
    finally:
        service._final_executor.shutdown(wait=True)


@pytest.mark.parametrize('value, expected', [
    (None, 15.0), ('0.125', 0.125), ('7 # inline comment', 7),
    ('', 15), ('bad', 15), ('0', 15), ('-1', 15),
    ('nan', 15), ('inf', 15), ('-inf', 15), ('1e999', 15), ('121', 15), ('120', 120),
])
def test_timeout_configuration(monkeypatch, value, expected):
    # Load an isolated config module; prevent .env from overriding test inputs.
    monkeypatch.setattr('dotenv.load_dotenv', lambda *a, **k: None)
    monkeypatch.delenv('HYBRID_STT_TIMEOUT_S', raising=False)
    if value is not None:
        monkeypatch.setenv('HYBRID_STT_TIMEOUT_S', value)
    spec = importlib.util.spec_from_file_location('phase1_config', Path(config.__file__))
    module = importlib.util.module_from_spec(spec)
    import sys
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert module.HYBRID_STT_TIMEOUT_S == expected


def test_timed_out_exception_releases_slot(cloud):
    started, release = threading.Event(), threading.Event()
    def fail_late(**kwargs):
        started.set()
        assert release.wait(3)
        raise RuntimeError('late SDK failure')
    cloud._client.models.generate_content.side_effect = fail_late
    try:
        assert cloud.transcribe(AUDIO) is None
        assert started.is_set()
        old = cloud._inflight
        assert cloud.transcribe(AUDIO) is None
        release.set()
        with pytest.raises(RuntimeError, match='late SDK failure'):
            old.result(1)
        cloud._client.models.generate_content.side_effect = None
        cloud._client.models.generate_content.return_value = SimpleNamespace(text='recovered')
        assert cloud.transcribe(AUDIO) == 'recovered'
    finally:
        release.set()


def test_unavailable_types_returns_none(cloud):
    cloud._types = None
    assert cloud.transcribe(AUDIO) is None
    cloud._client.models.generate_content.assert_not_called()


@pytest.mark.parametrize("code, category", [(401, "authentication_error"), (403, "authentication_error"),
                                            (429, "rate_limit"), (400, "api_error"), (503, "api_error")])
def test_api_error_categories_are_safe_and_recover(cloud, caplog, code, category):
    import json
    class APIError(Exception):
        pass
    exc = APIError("SECRET must not appear in diagnostics")
    exc.code = code
    cloud._client.models.generate_content.side_effect = exc
    assert cloud.transcribe(AUDIO, utterance_id="bad-api") is None
    events = [json.loads(r.message.split("STT_EVENT ")[1]) for r in caplog.records if "STT_EVENT " in r.message]
    assert any(e["category"] == category and e["utterance_id"] == "bad-api" for e in events)
    assert "SECRET" not in caplog.text
    cloud._client.models.generate_content.side_effect = None
    cloud._client.models.generate_content.return_value = SimpleNamespace(text="recovered")
    assert cloud.transcribe(AUDIO, utterance_id="next-api") == "recovered"


@pytest.mark.parametrize("exc, category", [(TimeoutError("secret"), "network_timeout"),
                                          (ConnectionError("secret"), "network_error")])
def test_network_error_categories(cloud, caplog, exc, category):
    cloud._client.models.generate_content.side_effect = exc
    assert cloud.transcribe(AUDIO) is None
    assert category in caplog.text and "secret" not in caplog.text


def test_constructor_validates_and_passes_transport_options(monkeypatch):
    from google import genai
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "mock-credential")
    monkeypatch.setattr(config, "HYBRID_STT_MODEL", "configured-audio-model")
    monkeypatch.setattr(config, "HYBRID_STT_TIMEOUT_S", 0.125)
    client = Mock()
    monkeypatch.setattr(genai, "Client", client)
    provider = GeminiAudioSTT()
    try:
        kwargs = client.call_args.kwargs
        assert kwargs["http_options"].timeout == 125
        assert kwargs["http_options"].retry_options.attempts == 1
        assert provider.model == "configured-audio-model"
    finally:
        provider.close()


@pytest.mark.parametrize("key, model", [("", "model"), ("mock", " ")])
def test_missing_configuration_fails_visibly_without_sdk_client(monkeypatch, caplog, key, model):
    from google import genai
    monkeypatch.setattr(config, "GOOGLE_API_KEY", key)
    monkeypatch.setattr(config, "HYBRID_STT_MODEL", model)
    client = Mock()
    monkeypatch.setattr(genai, "Client", client)
    with pytest.raises(RuntimeError, match="initialization failed"):
        GeminiAudioSTT()
    client.assert_not_called()
    assert "initialization_error" in caplog.text


def test_client_initialization_exception_does_not_expose_secrets(monkeypatch, caplog):
    from google import genai
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "mock-credential")
    monkeypatch.setattr(genai, "Client", Mock(side_effect=RuntimeError("SECRET credentials")))
    with pytest.raises(RuntimeError):
        GeminiAudioSTT()
    assert "SECRET" not in caplog.text and "mock-credential" not in caplog.text


def test_wav_encoding_and_sdk_model_and_prompt_preserved(cloud):
    import io
    import wave
    cloud._client.models.generate_content.return_value = SimpleNamespace(text="नमस्ते")
    assert cloud.transcribe(AUDIO) == "नमस्ते"
    kwargs = cloud._types.Part.from_bytes.call_args.kwargs
    assert kwargs["mime_type"] == "audio/wav"
    with wave.open(io.BytesIO(kwargs["data"]), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (
            1, 2, 16000, len(AUDIO))
    call = cloud._client.models.generate_content.call_args.kwargs
    assert call["model"] == cloud.model
    assert "original language" in call["contents"][1]


def test_timeout_busy_late_completion_and_new_delivery_ids(cloud, monkeypatch, caplog):
    service = make_service(cloud, monkeypatch)
    started, release = block_request(cloud)
    try:
        service._transcribe_and_emit(Utterance(audio=AUDIO, utterance_id="expired-id"))
        assert started.is_set() and service.bus.transcript_q.empty()
        old = cloud._inflight
        service._transcribe_and_emit(Utterance(audio=AUDIO, utterance_id="busy-id"))
        assert service.bus.transcript_q.empty()
        assert cloud._client.models.generate_content.call_count == 1
        release.set()
        old.result(1)
        cloud._client.models.generate_content.side_effect = None
        cloud._client.models.generate_content.return_value = SimpleNamespace(text="fresh")
        service._transcribe_and_emit(Utterance(audio=AUDIO, utterance_id="fresh-id"))
        tr = service.bus.transcript_q.get_nowait()
        assert (tr.text, tr.utterance_id) == ("fresh", "fresh-id")
        assert service.bus.transcript_q.empty()
        assert '"category": "timeout"' in caplog.text
        assert '"category": "occupied"' in caplog.text
        assert "SECRET" not in caplog.text
    finally:
        release.set()
        service._final_executor.shutdown(wait=True)


def test_close_rejects_new_requests_without_waiting(cloud):
    started, release = block_request(cloud)
    try:
        assert cloud.transcribe(AUDIO) is None
        before = time.monotonic()
        cloud.close()
        assert time.monotonic() - before < 0.5
        assert cloud.transcribe(AUDIO) is None
        assert cloud._client.models.generate_content.call_count == 1
    finally:
        release.set()
