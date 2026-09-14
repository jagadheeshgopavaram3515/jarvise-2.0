"""
Pipeline — constructs the bus and starts every service thread.

This is the single place that wires the producer/consumer graph together.
main.py just calls start_services() and owns the GUI/main loop.
"""
from __future__ import annotations

import queue
import threading

from assistant import config
from assistant.audio.capture import AudioInputService
from assistant.core.backchannel import BackchannelEngine
from assistant.core.dispatcher import Dispatcher
from assistant.core.events import Bus, Transcript
from assistant.core.health import HealthMonitor
from assistant.memory import store
from assistant.reminders.scheduler import ReminderService, ReminderStore
from assistant.stt.recognizer import STTService
from assistant.tts.service import TTSService


def start_services(bus: Bus) -> ReminderStore:
    """Construct + start every service, supervised by a HealthMonitor.

    Each service is created by a small ``spawn_*`` factory that builds, starts
    and (where needed) re-wires the thread. The factories double as the monitor's
    restart recipe: if a thread dies, the monitor calls the same factory to bring
    a fresh one up — without restarting Jarvis. Behaviour is identical to before
    when HEALTH_MONITOR_ENABLED=false (services just run unsupervised).
    """
    store.ensure_legacy_migrated()
    reminders = ReminderStore()
    monitor = HealthMonitor(bus)

    # Mutable handle map so factories can re-point cross-service references
    # (the backchannel holds a reference to the live TTS service).
    handles: dict = {}

    def spawn_tts() -> TTSService:
        svc = TTSService(bus)
        svc.start()
        handles["tts"] = svc
        # If the backchannel already exists, re-point it at the fresh TTS.
        if bus.backchannel is not None:
            bus.backchannel.tts = svc
        return svc

    def spawn_backchannel() -> BackchannelEngine:
        # Backchannel listens off the TTS engine but never through the turn queue.
        bc = BackchannelEngine(bus, handles["tts"])
        bc.start()
        bus.backchannel = bc
        return bc

    def spawn_dispatcher() -> Dispatcher:
        svc = Dispatcher(bus, reminders)
        svc.start()
        return svc

    def spawn_reminder() -> ReminderService:
        svc = ReminderService(bus, reminders)
        svc.start()
        return svc

    tts = spawn_tts()
    backchannel = spawn_backchannel()
    dispatcher = spawn_dispatcher()
    reminder_svc = spawn_reminder()

    # Register for heartbeat + auto-recovery. Backchannel is non-critical (it may
    # exit by design when disabled / no LLM) so it's monitored but never force-
    # restarted; the rest are critical and auto-recover.
    monitor.register("TTS", tts, spawn_tts, critical=True)
    monitor.register("Dispatcher", dispatcher, spawn_dispatcher, critical=True)
    monitor.register("Reminder", reminder_svc, spawn_reminder, critical=True)
    monitor.register("Backchannel", backchannel, spawn_backchannel, critical=False)

    # Audio/STT only run with a real microphone (skipped in demo mode).
    if not config.FLAGS.demo_mode:
        def spawn_stt() -> STTService:
            svc = STTService(bus)
            svc.start()
            handles["stt"] = svc
            return svc

        def spawn_audio() -> AudioInputService:
            svc = AudioInputService(bus)
            svc.start()
            return svc

        stt = spawn_stt()
        bus.post_gui("status", "Loading speech model...")
        bus.stt_ready.wait(timeout=config.STT_STARTUP_WAIT_SECONDS)
        audio = spawn_audio()
        monitor.register("STT", stt, spawn_stt, critical=True)
        monitor.register("AudioInput", audio, spawn_audio, critical=True)

    monitor.start()

    # Greeting once everything is up. Phase 7: if a recent mood is on record,
    # acknowledge it warmly instead of a cold hello.
    name = store.get_user_name()
    mood = store.recent_mood()
    if mood:
        bus.say(f"Welcome back, {name}. You seemed {mood} earlier — how are you feeling now, sir?")
    else:
        bus.say(f"Hello {name}, I'm {config.ASSISTANT_NAME}. How can I help you today, sir?")
    return reminders


class ConsoleInputService(threading.Thread):
    """Text-input producer for DEMO_MODE (no microphone)."""

    def __init__(self, bus: Bus):
        super().__init__(name="ConsoleInput", daemon=True)
        self.bus = bus

    def run(self):
        print("\n[DEMO MODE] Type commands ('quit' to exit).")
        while not self.bus.shutdown.is_set():
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                self.bus.shutdown.set()
                break
            if text:
                self.bus.transcript_q.put(Transcript(text=text))
