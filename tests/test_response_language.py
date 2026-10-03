"""Offline response-language selection and existing voice-selection contracts."""
from unittest.mock import Mock

import pytest

from assistant import config
from assistant.llm.language import response_language_instruction


@pytest.mark.parametrize("text", [
    "నువ్వు ఎలా ఉన్నావు?",
    "Jarvis, nen matladedi vinpistunda?",
    "jaarvash matladaadu vinipisthundha em chesthunnav",
    "Neene matladuthunnanu nanu nuvvematladuthunnavu?",
    "Jarvis, nenu ninnu ela better cheyalo aney vishyam",
    "Sigma movie review ento naku cheppu Jarvis.",
    "Telugu lo matladu",
    "Talk to me in Telugu for a moment.",
])
def test_telugu_and_logged_romanized_inputs_get_native_script_guidance(text):
    instruction = response_language_instruction(text)
    assert "Response language for THIS turn: Telugu" in instruction
    assert "native Telugu script" in instruction


@pytest.mark.parametrize("text", [
    "Can you hear me?", "Explain how Telugu grammar works in English.",
    "I want to learn Telugu.", "Tell me about neural networks.",
])
def test_english_is_not_forced_to_telugu(text):
    assert "Response language for THIS turn: Telugu" not in response_language_instruction(text)


def test_explicit_english_request_overrides_telugu_input():
    assert "THIS turn: English" in response_language_instruction("నువ్వు ఎలా ఉన్నావు? Reply in English")
    assert "THIS turn: Telugu" in response_language_instruction("Reply in English. Actually, reply in Telugu")


@pytest.mark.parametrize("provider", ["gemini", "ollama"])
def test_prompt_guidance_is_after_history_and_resets_each_turn(monkeypatch, provider):
    from assistant.memory import store
    from assistant.llm.gemini import GeminiClient
    from assistant.llm.ollama import OllamaClient

    monkeypatch.setattr(store, "load_history", lambda: [{"role": "assistant", "text": "Hindi"}])
    monkeypatch.setattr(store, "format_history", lambda history: "OLD_HISTORY_IN_HINDI")
    monkeypatch.setattr(store, "format_memory_context", lambda *a, **k: "")
    cls = GeminiClient if provider == "gemini" else OllamaClient
    client = cls.__new__(cls)  # No SDK/client initialization or requests.
    telugu = client._build_prompt("Nenu naku cheppu")
    assert telugu.index("OLD_HISTORY_IN_HINDI") < telugu.index("Response language for THIS turn: Telugu")
    english = client._build_prompt("Can you hear me?")
    assert "Response language for THIS turn: Telugu" not in english
    assert "For English, reply in English" in english


def test_existing_voice_selection_uses_native_reply_script(monkeypatch):
    from assistant.tts.engine import detect_voice

    monkeypatch.setattr(config, "VOICE_MAP", {"en": "english", "te": "telugu", "hi": "hindi"})
    assert detect_voice("వినిపిస్తోంది, సర్.") == "telugu"
    assert detect_voice("Yes sir, I can hear you.") == "english"


def test_streaming_sends_guidance_in_the_existing_single_request(monkeypatch):
    from types import SimpleNamespace
    from assistant.llm.gemini import GeminiClient
    from assistant.memory import store

    monkeypatch.setattr(store, "load_history", lambda: [])
    monkeypatch.setattr(store, "format_history", lambda history: "")
    monkeypatch.setattr(store, "format_memory_context", lambda *a, **k: "")
    monkeypatch.setattr(store, "add_exchange", Mock())
    client = GeminiClient.__new__(GeminiClient)
    client.model = "offline-model"
    client.realtime = Mock()
    client.realtime.needs_realtime.return_value = False
    request = Mock(return_value=iter([SimpleNamespace(text="వినిపిస్తోంది, సర్. ")]))
    client._client = SimpleNamespace(models=SimpleNamespace(generate_content_stream=request))
    assert list(client.stream("nen matladedi vinpistunda?")) == ["వినిపిస్తోంది, సర్."]
    request.assert_called_once()
    assert "THIS turn: Telugu" in request.call_args.kwargs["contents"]
    client.realtime.get_realtime_context.assert_not_called()
