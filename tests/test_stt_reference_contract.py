"""Reference request/audio contracts; synthetic audio and mocked SDK only."""
import io
import logging
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from assistant import config
from assistant.audio.capture import AudioInputService
from assistant.core.events import Bus, Utterance
from assistant.stt.cloud import GeminiAudioSTT, _encode_wav
from assistant.stt.recognizer import STTService


def test_wav_matches_reference_pcm_and_keeps_silence():
    audio = np.array([0, -2, -1, -0.5, 0.5, 1, 2, 0], dtype=np.float32)
    with wave.open(io.BytesIO(_encode_wav(audio)), 'rb') as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        assert wav.getnframes() == len(audio)
        assert np.frombuffer(wav.readframes(len(audio)), dtype='<i2').tolist() == [
            0, -32767, -32767, -16383, 16383, 32767, 32767, 0]


@pytest.mark.parametrize('text', ['hello', '\u0c28\u0c2e\u0c38\u0c4d\u0c15\u0c3e\u0c30\u0c02',
                                  'nuvvu ela unnava', 'meeting repu schedule cheyyi'])
def test_original_language_request_and_response_are_preserved(text, monkeypatch):
    monkeypatch.setattr(config, 'STT_LANGUAGE', 'en')  # legacy Whisper setting has no effect
    provider = GeminiAudioSTT.__new__(GeminiAudioSTT)
    provider.model = config.HYBRID_STT_MODEL
    provider._event = Mock()
    provider._types = SimpleNamespace(Part=SimpleNamespace(from_bytes=Mock(return_value='wav-part')))
    request = Mock(return_value=SimpleNamespace(text='  ' + text + '  '))
    provider._client = SimpleNamespace(models=SimpleNamespace(generate_content=request))
    audio = np.zeros(960, dtype=np.float32)
    assert provider._transcribe_blocking(audio) == text
    provider._types.Part.from_bytes.assert_called_once_with(data=_encode_wav(audio), mime_type='audio/wav')
    request.assert_called_once_with(model=config.HYBRID_STT_MODEL, contents=[
        'wav-part', 'Transcribe this audio verbatim in its original language '
        '(English, Hindi, Telugu, or a mix). If there is no clearly '
        'intelligible human speech, output an empty string; never infer '
        'words from silence, room noise, music, or background sounds. '
        'Output ONLY the transcript '
        'text — no quotes, labels, or commentary.'])


def test_capture_preserves_every_frame_and_utterance_identity():
    bus = Bus()
    capture = AudioInputService(bus)
    frames = [np.full(480, value, dtype=np.int16).tobytes() for value in [0, 300, -300, 0]]
    capture._emit(frames, 1.0, 2.0, False, 'original', 3)
    utt = bus.stt_in_q.get_nowait()
    np.testing.assert_array_equal(utt.audio, np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768)
    assert (utt.utterance_id, utt.sequence, utt.is_partial) == ('original', 3, False)


def test_rejection_diagnostics_expose_padding_without_changing_gate(monkeypatch, caplog):
    monkeypatch.setattr(config, 'STT_PREFILTER_RMS', 0.005)
    caplog.set_level(logging.INFO)
    service = STTService(Bus())
    try:
        audio = np.concatenate([np.full(1600, 0.006, dtype=np.float32), np.zeros(6400, dtype=np.float32)])
        before = audio.copy()
        assert not service._prefilter(Utterance(audio=audio, utterance_id='measurement'))
        assert "'max_frame_rms': 0.006" in caplog.text
        # The fourth 30ms frame contains 10ms signal and 20ms silence;
        # its RMS falls below threshold, leaving three qualifying frames.
        assert "'above_threshold_ms': 90.0" in caplog.text
        np.testing.assert_array_equal(audio, before)
        assert not service._prefilter(Utterance(audio=np.zeros(1600)))
        noise = np.random.default_rng(0).normal(0, 0.001, 1600).astype(np.float32)
        assert not service._prefilter(Utterance(audio=noise))
        assert service._prefilter(Utterance(audio=np.full(1600, 0.006, dtype=np.float32)))
    finally:
        service._final_executor.shutdown(wait=True)


def test_conversation_reaches_existing_dispatcher_orchestrator(monkeypatch):
    from assistant.core.dispatcher import Dispatcher
    from assistant.commands import handlers
    from assistant.memory import store
    monkeypatch.setattr(config, 'STT_VAD_PREFILTER', False)
    monkeypatch.setattr(config, 'ENABLE_HUMANIZER', False)
    monkeypatch.setattr(store, 'remember_mood', Mock())
    monkeypatch.setattr(handlers, 'handle', Mock(return_value=False))
    bus = Bus()
    service = STTService(bus)
    service._cloud = Mock(in_flight=False)
    service._cloud.transcribe.return_value = 'explain why the sky is blue'
    try:
        service._transcribe_and_emit(Utterance(audio=np.ones(1600), utterance_id='conversation'))
        assert bus.tts_q.empty()
        dispatcher = Dispatcher(bus, Mock())
        dispatcher._min_dispatch_interval = 0
        dispatcher.llm = Mock()
        dispatcher.llm.stream.return_value = iter(['Light scatters in the atmosphere.'])
        dispatcher._route(bus.transcript_q.get_nowait())
        dispatcher.llm.stream.assert_called_once_with('explain why the sky is blue')
        assert bus.tts_q.get_nowait().text == 'Light scatters in the atmosphere.'
        assert bus.tts_q.get_nowait().is_end
    finally:
        service._final_executor.shutdown(wait=True)
