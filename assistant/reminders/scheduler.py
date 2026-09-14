"""
ReminderService — background thread that fires due reminders via TTS.

Reminders live in a thread-safe store shared with the commands module.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime

from assistant.core.events import Bus
from assistant.core.log import get

log = get("reminder")


class ReminderStore:
    def __init__(self):
        self._items: list[dict] = []
        self._lock = threading.Lock()

    def add(self, text: str, when: str):
        with self._lock:
            self._items.append({"text": text, "time": when})

    def delete_matching(self, keyword: str) -> int:
        kw = keyword.lower()
        with self._lock:
            matched = [r for r in self._items if kw in r["text"].lower()]
            for r in matched:
                self._items.remove(r)
            return len(matched)

    def pop_due(self, now_str: str) -> list[dict]:
        with self._lock:
            due = [r for r in self._items if r["time"] == now_str]
            for r in due:
                self._items.remove(r)
            return due


class ReminderService(threading.Thread):
    def __init__(self, bus: Bus, store: ReminderStore, interval: int = 30):
        super().__init__(name="Reminder", daemon=True)
        self.bus = bus
        self.store = store
        self.interval = interval

    def run(self):
        while not self.bus.shutdown.is_set():
            # A bad tick (clock/store/TTS hiccup) must never kill the scheduler.
            try:
                now = datetime.now().strftime("%Y-%m-%d %H:%M")
                for r in self.store.pop_due(now):
                    self.bus.say(f"Reminder: {r['text']}")
            except Exception:
                log.exception("reminder tick failed — scheduler stays alive")
            # Sleep in small slices so shutdown is responsive.
            for _ in range(self.interval):
                if self.bus.shutdown.is_set():
                    return
                time.sleep(1)
