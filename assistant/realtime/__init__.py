"""
Real-Time Knowledge Layer (additive).

Jarvis searches the live web only when a query needs *current* information
(news / today / latest / stock / weather / sports / …) and injects a sanitised
briefing into the Gemini prompt, so the reply is spoken in Jarvis's own voice.
Timeless/definitional queries ("what is Python") never trigger a search.

Public surface:

    from assistant.realtime import RealTimeManager
    rt = RealTimeManager()
    rt.needs_realtime("latest AI news")     # -> True
    rt.get_realtime_context("latest AI news")  # -> structured, sanitised context str

Everything else (providers, extractor, summarizer, cache, sanitiser) is an
internal component used by the manager. Optional third-party libs degrade
gracefully: if a provider or extractor isn't installed, the layer falls back or
returns "" so the normal Gemini path is preserved.
"""
from __future__ import annotations

from assistant.realtime.manager import RealTimeManager
from assistant.realtime.providers import SearchProvider, SearchResult

__all__ = ["RealTimeManager", "SearchProvider", "SearchResult"]
