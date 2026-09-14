"""
Step 4 — SourceExtractor.

Given a result URL, fetch the page and return clean, visible article text with
navigation, ads and boilerplate removed. We deliberately use main-content
extractors rather than raw scraping:

    1. trafilatura       (best boilerplate removal, if installed)
    2. readability-lxml  (Arc90 readability port, if installed)
    3. BeautifulSoup     (helper-only fallback: drop script/style/nav/footer/aside)

Each page is fetched once with a short timeout; failures return "" so a single
bad URL never blocks the briefing.
"""
from __future__ import annotations

import re

from assistant import config
from assistant.core.log import get

log = get("realtime")

_WS_RE = re.compile(r"\s+")
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; JarvisBot/1.0; +realtime)"}
# Tags that are never article content.
_STRIP_TAGS = ("script", "style", "nav", "footer", "header", "aside", "form",
               "noscript", "iframe", "svg", "button")


class SourceExtractor:
    def extract(self, url: str) -> str:
        """Return cleaned main-content text for a URL (capped), or ""."""
        if not url:
            return ""
        text = self._via_trafilatura(url)
        if not text:
            html = self._fetch(url)
            if not html:
                return ""
            text = self._via_readability(html) or self._via_bs4(html)
        text = _WS_RE.sub(" ", text or "").strip()
        if len(text) > config.REALTIME_EXTRACT_MAX_CHARS:
            text = text[: config.REALTIME_EXTRACT_MAX_CHARS].rsplit(" ", 1)[0] + "…"
        return text

    # -- fetch ---------------------------------------------------------------
    def _fetch(self, url: str) -> str:
        try:
            import requests
            resp = requests.get(url, headers=_HEADERS,
                                timeout=config.REALTIME_FETCH_TIMEOUT_S)
            ctype = resp.headers.get("Content-Type", "")
            if "html" not in ctype and ctype:
                return ""
            return resp.text or ""
        except Exception as e:
            log.debug("[REALTIME] fetch failed %s: %s", url, e)
            return ""

    # -- extractors (most → least capable) -----------------------------------
    def _via_trafilatura(self, url: str) -> str:
        try:
            import trafilatura  # type: ignore
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return ""
            return trafilatura.extract(
                downloaded, include_comments=False, include_tables=False) or ""
        except Exception as e:
            log.debug("[REALTIME] trafilatura unavailable/failed: %s", e)
            return ""

    def _via_readability(self, html: str) -> str:
        try:
            from readability import Document  # type: ignore
            from bs4 import BeautifulSoup
            summary_html = Document(html).summary(html_partial=True)
            return BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)
        except Exception as e:
            log.debug("[REALTIME] readability unavailable/failed: %s", e)
            return ""

    def _via_bs4(self, html: str) -> str:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(_STRIP_TAGS):
                tag.decompose()
            main = soup.find("article") or soup.find("main") or soup.body or soup
            # Paragraph text only — drops most menu/ad fragments.
            paras = [p.get_text(" ", strip=True) for p in main.find_all("p")]
            text = " ".join(p for p in paras if len(p) > 40)
            return text or main.get_text(" ", strip=True)
        except Exception as e:
            log.debug("[REALTIME] bs4 extract failed: %s", e)
            return ""
