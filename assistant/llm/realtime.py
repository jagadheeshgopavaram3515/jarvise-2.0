"""
Real-time information lookups (SerpAPI search, Calendarific holidays, IP geo).

Behaviour ported from the original monolith; keys now come from config/.env.
"""
from __future__ import annotations

import re
from datetime import datetime

import requests

from assistant import config

# Word-boundary so "now" doesn't match "know", "age" doesn't match "message", etc.
# time/date are handled deterministically by the command router, not here.
_REALTIME_RE = re.compile(
    r"\b(current|today|now|weather|news|stock|price|holiday|temperature|forecast|"
    r"score|headlines)\b", re.I)


def should_use_realtime(query: str) -> bool:
    return bool(_REALTIME_RE.search(query or ""))


def get_user_country_code() -> str | None:
    try:
        data = requests.get("https://ipinfo.io/json", timeout=5).json()
        return data.get("country")
    except Exception:
        return None


def real_time_search(query: str) -> str:
    if not config.SERPAPI_KEY:
        return "Real-time search is not configured, sir."
    try:
        from serpapi import GoogleSearch
        params = {
            "engine": "google", "hl": "en", "gl": "IN",
            "q": query, "api_key": config.SERPAPI_KEY,
        }
        results = GoogleSearch(params).get_dict()

        if "answer_box" in results:
            ab = results["answer_box"]
            ans = ab.get("answer") or ab.get("snippet") or \
                (ab.get("highlighted_words") or [None])[0]
            if ans:
                return ans
        kg = results.get("knowledge_graph", {})
        if "description" in kg:
            return kg["description"]
        organic = results.get("organic_results") or []
        if organic:
            return organic[0].get("snippet", "Sorry, I couldn't find anything specific.")
    except Exception as e:
        return f"Search failed: {e}"
    return "Sorry, I couldn't find any relevant results."


def get_holidays(country: str, year: int | None = None, month: int | None = None) -> str:
    if not config.CALENDARIFIC_API_KEY:
        return "Holiday lookups are not configured, sir."
    year = year or datetime.now().year
    params = {"api_key": config.CALENDARIFIC_API_KEY, "country": country, "year": year}
    if month:
        params["month"] = month
    try:
        resp = requests.get("https://calendarific.com/api/v2/holidays",
                            params=params, timeout=8)
        if resp.status_code != 200:
            return "Sorry, I couldn't fetch holidays right now."
        holidays = resp.json().get("response", {}).get("holidays", [])
    except Exception:
        return "Sorry, I couldn't fetch holidays right now."

    if not holidays:
        return f"No holidays found for {country} in {month or year}."
    listed = ", ".join(f"{h['name']} ({h['date']['iso']})" for h in holidays)
    return f"Holidays in {country} for {month or year}: {listed}"
