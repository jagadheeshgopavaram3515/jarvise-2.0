"""
Deterministic command router.

Tried BEFORE the LLM. If a phrase matches a known command we run it and
return True (so the dispatcher skips the LLM). All speech goes through
bus.speak so nothing blocks. Feature set is identical to the original
monolith: web search, open/close apps, play music, time/date, quit,
reminders (set/delete) and trip-holiday planning.
"""
from __future__ import annotations

import calendar
import os
import re
import webbrowser
from datetime import datetime

from assistant.core.events import Bus
from assistant.core.log import get
from assistant.llm import realtime
from assistant.reminders.scheduler import ReminderStore

log = get("commands")

COUNTRY_CODES = {
    "india": "IN", "united states": "US", "usa": "US", "canada": "CA",
    "united kingdom": "GB", "uk": "GB", "australia": "AU",
}

# Clock/date INTENT — must NOT fire on "the last time we talked", "update me",
# "sometime", etc. Only genuine "what's the time / date" questions, incl. common
# romanised Telugu/Hindi forms ("time enta", "time kya", "kitne baje").
_TIME_INTENT = re.compile(
    r"\bwhat(?:'s| is)?\s+(?:the\s+)?time\b|\btime\s+is\s+it\b|\bcurrent\s+time\b|"
    r"\btell\s+me\s+(?:the\s+)?time\b|\bthe\s+time\s+(?:now|please)\b|"
    r"\btime\s+(?:now|please|enta|enti|entha|cheppu|kya|kitna)\b|"
    r"\bkitne\s+baje\b|\bsamay\s+kya\b", re.I)
_DATE_INTENT = re.compile(
    r"\bwhat(?:'s| is)?\s+(?:the\s+|today'?s\s+)?date\b|\btoday'?s\s+date\b|"
    r"\bwhat\s+day\s+is\s+it\b|\bcurrent\s+date\b|\bdate\s+(?:enta|enti|kya)\b", re.I)


def handle(text: str, bus: Bus, reminders: ReminderStore) -> bool:
    t = text.lower().strip()

    if "search for" in t:
        query = t.split("search for", 1)[1].strip()
        if query:
            bus.speak(f"Searching for {query}, sir.")
            webbrowser.open(f"https://www.google.com/search?q={query}")
        else:
            bus.speak("What should I search for?")
        return True

    if "open chrome" in t:
        webbrowser.open("https://www.google.com")
        bus.speak("Opening Chrome.")
        return True

    if "open youtube" in t:
        webbrowser.open("https://www.youtube.com")
        bus.speak("Opening YouTube.")
        return True

    if "close chrome" in t or "close youtube" in t:
        os.system("taskkill /f /im chrome.exe")
        bus.speak("Closed Chrome.")
        return True

    if "play" in t and "music" in t:
        song = t.replace("play music", "").strip()
        if song:
            try:
                import pywhatkit
                pywhatkit.playonyt(song)
                bus.speak(f"Playing {song} on YouTube.")
            except Exception:
                webbrowser.open(f"https://www.youtube.com/results?search_query={song}")
                bus.speak(f"Playing {song} on YouTube.")
        else:
            bus.speak("Which song should I play?")
        return True

    if _TIME_INTENT.search(t):
        bus.speak("The time is " + datetime.now().strftime("%I:%M %p"))
        return True

    if _DATE_INTENT.search(t):
        bus.speak("Today is " + datetime.now().strftime("%A, %B %d, %Y"))
        return True

    if re.search(r"\b(?:what(?:'s| is)?\s+my\s+name|tell\s+me\s+my\s+name|who\s+am\s+i)\b", t):
        from assistant.memory import store
        bus.speak(f"Your name is {store.get_user_name()}, sir.")
        return True

    name_decl = re.search(
        r"^(?:jarvis\s*,?\s*)?(?:my\s+name\s+is|call\s+me|change\s+my\s+name\s+to)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\.?$",
        t, re.I
    )
    if name_decl:
        from assistant.memory import store
        new_name = name_decl.group(1).strip().title()
        if new_name.lower() not in {"not", "a", "an", "the", "here", "sorry"}:
            store.set_user_name(new_name)
            bus.speak(f"Understood sir. I have updated your name to {new_name}.")
            return True

    if any(w in t for w in ("quit", "exit", "goodbye", "bye")):
        bus.speak("Alright sir, goodbye.")
        bus.shutdown.set()
        return True

    if "remind me" in t or "set a reminder" in t:
        return _set_reminder(t, bus, reminders)

    if "delete reminder" in t:
        keyword = t.replace("delete reminder", "").strip()
        if not keyword:
            bus.speak("Please tell me which reminder to delete.")
        elif reminders.delete_matching(keyword):
            bus.speak(f"Deleted reminder for {keyword}.")
        else:
            bus.speak("No matching reminder found.")
        return True

    if "planning for a trip" in t or "plan a trip" in t:
        return _plan_trip(t, bus)

    # ---- Desktop Tools layer (Jarvis V2, additive) ----------------------
    # Tried AFTER the legacy deterministic commands above, BEFORE the LLM. If a
    # desktop tool (open/close app, find file, browser, …) satisfies the request
    # it runs and returns True so the dispatcher skips the conversational LLM.
    # Disabled cleanly via TOOLS_ENABLED=false. Never raises into the dispatcher.
    try:
        from assistant.tools.router import try_tools
        if try_tools(text, bus):
            return True
    except Exception:  # the tools layer must never break the voice pipeline
        log.exception("[TOOLS] router error — falling through to LLM")

    return False


def _set_reminder(t: str, bus: Bus, reminders: ReminderStore) -> bool:
    from dateutil import parser
    parts = t.replace("remind me", "").replace("set a reminder", "").strip()
    match = re.search(r"(.+?) at (.+)", parts)
    if not match:
        bus.speak("Sorry, I didn't understand the time. Please try again.")
        return True
    message, time_str = match.group(1).strip(), match.group(2).strip()
    try:
        when = parser.parse(time_str)
        reminders.add(message, when.strftime("%Y-%m-%d %H:%M"))
        bus.speak(f"Okay, reminder set for {message} at {when:%I:%M %p}.")
    except Exception:
        bus.speak("Something went wrong while setting the reminder.")
    return True


def _plan_trip(t: str, bus: Bus) -> bool:
    loc_match = re.search(r"trip (?:to|in|at)?\s*([a-zA-Z\s]+)", t)
    month_match = re.search(
        r"(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)", t, re.I)
    location = loc_match.group(1).strip() if loc_match else None
    month_num = None
    if month_match:
        month_num = list(calendar.month_name).index(month_match.group(1).capitalize())

    code = COUNTRY_CODES.get(location.lower()) if location else None
    if not code:
        code = realtime.get_user_country_code()
        if code:
            bus.speak(f"Using your current location detected as {code} for the trip.")
        else:
            bus.speak("Please specify a valid country for the trip.")
            return True
    bus.speak(realtime.get_holidays(code, month=month_num))
    return True
