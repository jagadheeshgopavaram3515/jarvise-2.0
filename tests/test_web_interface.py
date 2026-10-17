"""Transport contracts: no microphone, model calls, TTS or memory writes."""
import time
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from assistant.core.perf import PerfTracker
from assistant.web.app import Runtime, create_app


@pytest.fixture
def runtime():
    return Runtime(starter=Mock())


def test_standby_does_not_start_services_and_exports_events(runtime):
    client = TestClient(create_app(runtime))
    runtime.bus.post_gui("status", "Thinking...")
    runtime.bus.post_gui("subtitle", "Hello", role="assistant")
    data = client.get("/api/state").json()
    assert data["phase"] == "thinking"
    assert data["events"][-1]["text"] == "Hello"
    assert not data["ready"]
    runtime._starter.assert_not_called()
    # A second viewer receives the same events, rather than stealing them.
    assert client.get("/api/state").json()["events"] == data["events"]


def test_start_is_idempotent(runtime):
    client = TestClient(create_app(runtime))
    assert client.post("/api/start").status_code == 200
    assert client.post("/api/start").status_code == 200
    runtime._thread.join(1)
    runtime._starter.assert_called_once_with(runtime.bus)
    assert runtime.ready


def test_chat_uses_existing_dispatch_queue_and_preserves_interrupt_metadata(runtime):
    client = TestClient(create_app(runtime))
    assert client.post("/api/chat", json={"text": "hello"}).status_code == 409
    runtime.ready = True
    runtime.bus.state.set_speaking(True)
    runtime.bus.interrupt_epoch = 4
    assert client.post("/api/chat", json={"text": " hello jarvis "}).status_code == 200
    transcript = runtime.bus.transcript_q.get_nowait()
    assert transcript.text == "hello jarvis"
    assert transcript.utterance_id and transcript.perf.t_start <= time.perf_counter()
    assert transcript.during_speech and transcript.interrupt_epoch == 4
    assert runtime.bus.tts_q.empty()
    assert client.post("/api/chat", json={"text": "   "}).status_code == 422


def test_stop_invalidates_old_turn(runtime):
    client = TestClient(create_app(runtime))
    assert client.post("/api/interrupt").status_code == 409
    assert runtime.phase == "offline"
    runtime.ready = True
    runtime.bus.start_turn()
    before = runtime.bus.interrupt_epoch
    assert client.post("/api/interrupt").status_code == 200
    assert runtime.bus.interrupt.is_set()
    assert runtime.bus.current_turn_id is None
    assert runtime.bus.interrupt_epoch == before + 1


def test_rejects_cross_origin_commands_and_socket(runtime):
    client = TestClient(create_app(runtime))
    assert client.post("/api/start", headers={"Origin": "https://example.com"}).status_code == 403
    runtime._starter.assert_not_called()
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"Origin": "https://example.com"}):
            pass
    with client.websocket_connect("/ws", headers={"Origin": "http://127.0.0.1:3000"}) as socket:
        assert socket.receive_json()["phase"] == "offline"


def test_performance_snapshot_is_detached_and_handles_empty_history(tmp_path):
    tracker = PerfTracker(str(tmp_path / "unused.json"))
    assert tracker.snapshot()["averages"]["total"] is None
    tracker._records = [{"stt": 1.0, "total": 3.0}]
    snapshot = tracker.snapshot()
    assert snapshot["averages"]["stt"] == 1.0
    snapshot["latest"]["total"] = 999
    assert tracker.snapshot()["latest"]["total"] == 3.0


def test_response_is_visible_before_playback_and_has_stable_sentence_id(runtime):
    turn = runtime.bus.start_turn()
    runtime.bus.speak("Hello, sir.")
    response = runtime.snapshot()["events"][-1]
    item = runtime.bus.tts_q.get_nowait()
    assert response["kind"] == "response"
    assert response["speech_id"] == item.speech_id
    assert response["turn_id"] == turn
    assert response["text"] == "Hello, sir."


def test_speaking_event_follows_actual_audio_callback(runtime, monkeypatch):
    from assistant import config
    from assistant.tts.service import TTSService
    monkeypatch.setattr("assistant.tts.service.make_tts_engine", Mock())
    service = TTSService(runtime.bus)
    turn = runtime.bus.start_turn()
    service.engine.play_synthesized.side_effect = lambda audio, interrupt, on_play_start: on_play_start()
    try:
        service._speak_measured("Hello", None, b"audio", speech_id="sentence", turn_id=turn)
        events = runtime.snapshot()["events"]
        assert [event["kind"] for event in events] == ["speech_start", "speech_end"]
        assert all(event["speech_id"] == "sentence" for event in events)
    finally:
        service._synth_pool.shutdown(wait=True)


def test_edge_word_boundaries_follow_played_audio_not_generation(runtime, monkeypatch):
    from assistant.tts.engine import TimedAudio
    from assistant.tts.service import TTSService
    monkeypatch.setattr("assistant.tts.service.make_tts_engine", Mock())
    service = TTSService(runtime.bus)
    turn = runtime.bus.start_turn()
    audio = TimedAudio(b"mp3", [(0.0, 6), (0.5, 12)])

    def play(data, interrupt, on_play_start):
        assert not runtime.snapshot()["events"]
        on_play_start()
        service.engine.progress_sink(0.1, .4)
        events = runtime.snapshot()["events"]
        assert events[-1]["text"] == "Hello "
        assert events[-1]["kind"] == "speech_progress"
        service.engine.progress_sink(.6, .3)
        assert runtime.snapshot()["events"][-1]["text"] == "Hello world."

    service.engine.play_synthesized.side_effect = play
    try:
        service._speak_measured("Hello world.", None, audio, speech_id="word-sync", turn_id=turn)
        assert runtime.bus.state.audio_state.output_level == 0
    finally:
        service._synth_pool.shutdown(wait=True)


def test_edge_collection_keeps_word_metadata_with_its_own_audio(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from assistant.tts import engine

    class Communication:
        def __init__(self, *args, **kwargs):
            assert kwargs["boundary"] == "WordBoundary"

        async def stream(self):
            yield {"type": "audio", "data": b"audio"}
            yield {"type": "WordBoundary", "text": "Hello", "offset": 0}
            yield {"type": "WordBoundary", "text": "world", "offset": 5_000_000}

    monkeypatch.setattr(engine, "edge_tts", SimpleNamespace(Communicate=Communication))
    audio = asyncio.run(engine.EdgeStreamingTTS()._collect("Hello world.", "en-GB-RyanNeural", "curious"))
    assert bytes(audio) == b"audio"
    assert audio.boundaries == ((0.0, 6), (.5, 12))
