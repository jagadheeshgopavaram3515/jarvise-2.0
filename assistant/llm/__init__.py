"""LLM Provider package — exposes GeminiClient, OllamaClient, and get_llm_client."""
from __future__ import annotations

from assistant.llm.gemini import GeminiClient
from assistant.llm.ollama import OllamaClient


def get_llm_client(provider: str | None = None):
    """Return an LLM client instance for the requested or configured provider.

    Defaults strictly to 'gemini' so existing pipeline behavior is 100% preserved.
    """
    from assistant import config

    choice = (provider or config.LLM_PROVIDER).lower()
    if choice == "ollama":
        return OllamaClient()
    return GeminiClient()


__all__ = ["GeminiClient", "OllamaClient", "get_llm_client"]

