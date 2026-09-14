"""
Browser control (Phase 4) — Playwright, DOM-based, headless by default.

Strictly Playwright. No Selenium, no PyAutoGUI, no screen/image clicking — all
automation is DOM-based. ``read_page`` / ``get_page_title`` run a headless
Chromium, navigate, and read the DOM. The "open" actions (``open_url``,
``google_search``, ``youtube_search``) launch the user's real browser via
``webbrowser`` so the user actually sees the page — that's the desired UX for a
voice assistant ("open YouTube and search …").

Playwright is imported lazily inside each headless function so this module
imports cleanly even when Playwright (or its browser binary) isn't installed;
the tool then returns a clean, speakable error instead of raising at import.
"""
from __future__ import annotations

import urllib.parse
import webbrowser
from typing import Optional

from assistant import config
from assistant.core.log import get
from assistant.tools.schemas import fail, ok

log = get("tools")


def _normalise_url(url: str) -> str:
    """Add a scheme if the user said a bare domain ('github.com')."""
    url = (url or "").strip()
    if not url:
        return url
    if not url.startswith(("http://", "https://", "file://")):
        url = "https://" + url
    return url


# --------------------------------------------------------------------------- #
# "Open in the user's real browser" actions (visible, what the user wants).
# --------------------------------------------------------------------------- #
def open_url(url: str) -> dict:
    """Open a URL in the default browser (visible)."""
    target = _normalise_url(url)
    if not target:
        return fail("Which page should I open, sir?", error="empty_url")
    try:
        webbrowser.open(target)
        log.info("[BROWSER] opened %s", target)
        return ok(f"Opening {target}.", data={"url": target})
    except Exception as e:
        return fail(f"I couldn't open that page, sir: {e}", error=str(e))


def google_search(query: str) -> dict:
    """Open a Google search for ``query`` in the default browser."""
    q = (query or "").strip()
    if not q:
        return fail("What should I search for, sir?", error="empty_query")
    url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(q)
    try:
        webbrowser.open(url)
        log.info("[BROWSER] google search %r", q)
        return ok(f"Searching Google for {q}.", data={"url": url, "query": q})
    except Exception as e:
        return fail(f"I couldn't run that search, sir: {e}", error=str(e))


def youtube_search(query: str) -> dict:
    """Open a YouTube search for ``query`` in the default browser."""
    q = (query or "").strip()
    if not q:
        return fail("What should I search YouTube for, sir?", error="empty_query")
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(q)
    try:
        webbrowser.open(url)
        log.info("[BROWSER] youtube search %r", q)
        return ok(f"Searching YouTube for {q}.", data={"url": url, "query": q})
    except Exception as e:
        return fail(f"I couldn't search YouTube, sir: {e}", error=str(e))


# --------------------------------------------------------------------------- #
# Headless DOM reads (Playwright). Lazy import keeps the module importable.
# --------------------------------------------------------------------------- #
def _fetch_dom(url: str, want: str = "text") -> dict:
    """Run headless Chromium, navigate to ``url`` and read the DOM.

    want="title" → page <title>; want="text" → trimmed innerText snippet.
    Returns a canonical result dict. Never raises.
    """
    target = _normalise_url(url)
    if not target:
        return fail("Which page should I read, sir?", error="empty_url")

    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return fail("Browser reading isn't available — Playwright isn't installed, sir.",
                    error="playwright_missing")

    timeout_ms = int(config.TOOLS_BROWSER_TIMEOUT_S * 1000)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=config.TOOLS_BROWSER_HEADLESS)
            try:
                page = browser.new_page()
                page.goto(target, timeout=timeout_ms, wait_until="domcontentloaded")
                title = page.title()
                if want == "title":
                    result = ok(f"The page title is: {title}",
                                data={"url": target, "title": title})
                else:
                    body = page.inner_text("body") if page.query_selector("body") else ""
                    snippet = _clean_text(body, config.TOOLS_BROWSER_MAX_CHARS)
                    result = ok(snippet or f"The page '{title}' had no readable text.",
                                data={"url": target, "title": title, "text": snippet})
            finally:
                browser.close()
        log.info("[BROWSER] read %s (%s)", target, want)
        return result
    except Exception as e:
        log.warning("[BROWSER] _fetch_dom failed for %s: %s", target, e)
        return fail(f"I couldn't read that page, sir: {e}", error=str(e))


def _clean_text(text: str, max_chars: int) -> str:
    """Collapse whitespace and trim to a speakable snippet."""
    if not text:
        return ""
    collapsed = " ".join(text.split())
    if len(collapsed) > max_chars:
        collapsed = collapsed[:max_chars].rsplit(" ", 1)[0] + "…"
    return collapsed


def read_page(url: str) -> dict:
    """Fetch a page headlessly and return a snippet of its readable text."""
    return _fetch_dom(url, want="text")


def get_page_title(url: str) -> dict:
    """Fetch a page headlessly and return its <title>."""
    return _fetch_dom(url, want="title")
