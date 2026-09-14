"""
Step 3 — SearchProvider abstraction.

    class SearchProvider:  search(query) -> list[SearchResult]
    SearchResult:          title, url, snippet, source

Providers:
  * SerpAPIProvider   — primary (Google via SerpAPI; reuses config.SERPAPI_KEY).
  * DuckDuckGoProvider — fallback (no API key; uses the duckduckgo_search lib if
    installed, else the lightweight DuckDuckGo HTML endpoint).

We only ever take the TOP results — no crawling whole sites here. Page-body
extraction is a separate, deliberate step (see extractor.py).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urlparse

from assistant import config
from assistant.core.log import get

log = get("realtime")


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str

    @staticmethod
    def from_url(title: str, url: str, snippet: str) -> "SearchResult":
        try:
            host = urlparse(url).netloc.replace("www.", "")
        except Exception:
            host = ""
        return SearchResult(title=title or "", url=url or "",
                            snippet=snippet or "", source=host)


class SearchProvider(ABC):
    name = "base"

    @abstractmethod
    def search(self, query: str) -> list[SearchResult]:
        """Return up to config.REALTIME_MAX_RESULTS top results (best-effort)."""
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return True


class SerpAPIProvider(SearchProvider):
    name = "serpapi"

    @property
    def available(self) -> bool:
        return bool(config.SERPAPI_KEY)

    def search(self, query: str) -> list[SearchResult]:
        if not self.available:
            return []
        from serpapi import GoogleSearch
        params = {
            "engine": "google", "hl": config.REALTIME_HL, "gl": config.REALTIME_GL,
            "q": query, "api_key": config.SERPAPI_KEY,
            "num": config.REALTIME_MAX_RESULTS,
        }
        data = GoogleSearch(params).get_dict()
        results: list[SearchResult] = []

        # An answer box / knowledge-graph blurb is a high-signal "top result" —
        # surface it first so direct factual answers (stock, scores) lead.
        ab = data.get("answer_box") or {}
        ans = ab.get("answer") or ab.get("snippet")
        if ans:
            results.append(SearchResult.from_url(
                ab.get("title", "Quick answer"), ab.get("link", ""), ans))
        kg = data.get("knowledge_graph") or {}
        if kg.get("description"):
            results.append(SearchResult.from_url(
                kg.get("title", "Overview"), kg.get("source", {}).get("link", ""),
                kg["description"]))

        for r in (data.get("organic_results") or []):
            results.append(SearchResult.from_url(
                r.get("title", ""), r.get("link", ""), r.get("snippet", "")))
            if len(results) >= config.REALTIME_MAX_RESULTS + 2:
                break
        return results[: config.REALTIME_MAX_RESULTS + 2]


class DuckDuckGoProvider(SearchProvider):
    name = "duckduckgo"

    def search(self, query: str) -> list[SearchResult]:
        # Prefer the maintained library; fall back to the HTML endpoint so the
        # fallback still works with zero extra dependencies installed.
        try:
            return self._via_lib(query)
        except Exception as e:
            log.debug("[REALTIME] ddg lib path failed (%s); trying html endpoint", e)
        try:
            return self._via_html(query)
        except Exception as e:
            log.warning("[REALTIME] duckduckgo fallback failed: %s", e)
            return []

    def _via_lib(self, query: str) -> list[SearchResult]:
        from duckduckgo_search import DDGS  # type: ignore
        out: list[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=config.REALTIME_MAX_RESULTS):
                out.append(SearchResult.from_url(
                    r.get("title", ""), r.get("href", ""), r.get("body", "")))
        return out

    def _via_html(self, query: str) -> list[SearchResult]:
        import requests
        from bs4 import BeautifulSoup  # helper only — not full-site scraping
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; JarvisBot/1.0)"},
            timeout=config.REALTIME_FETCH_TIMEOUT_S,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        out: list[SearchResult] = []
        for res in soup.select("div.result")[: config.REALTIME_MAX_RESULTS]:
            a = res.select_one("a.result__a")
            snip = res.select_one(".result__snippet")
            if not a:
                continue
            out.append(SearchResult.from_url(
                a.get_text(" ", strip=True),
                a.get("href", ""),
                snip.get_text(" ", strip=True) if snip else ""))
        return out


def get_provider() -> SearchProvider | None:
    """Pick a provider per config, with SerpAPI→DuckDuckGo auto-fallback."""
    pref = config.REALTIME_PROVIDER
    if pref == "serpapi":
        p = SerpAPIProvider()
        return p if p.available else None
    if pref == "duckduckgo":
        return DuckDuckGoProvider()
    # auto: SerpAPI when a key exists, else DuckDuckGo.
    serp = SerpAPIProvider()
    return serp if serp.available else DuckDuckGoProvider()
