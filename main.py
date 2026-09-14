"""
Jarvis entry point.

  python main.py            # full GUI + microphone, real-time full-duplex
  DEMO_MODE=true python ... # headless console (no mic), for testing the brain

Threading model (all daemon threads, coordinated only through the Bus):

    ┌──────────────┐   utterance   ┌──────────┐  transcript  ┌─────────────┐
    │ AudioInput   │ ────────────▶ │   STT    │ ───────────▶ │ Dispatcher  │
    │ (mic + VAD)  │   stt_in_q    │ (whisper)│ transcript_q │ (cmd / LLM) │
    └──────────────┘               └──────────┘              └──────┬──────┘
            ▲ full-duplex: keeps recording while speaking            │ tts_q
            │                                                ┌───────▼──────┐
            │           interrupt (barge-in / spacebar)      │     TTS      │
            └────────────────────────────────────────────────│ (edge_tts)  │
                                                             └──────────────┘
    GUI thread (main) drains gui_q;  Reminder thread fires due reminders.
"""
from __future__ import annotations

import os
import tkinter as tk

from assistant import config
from assistant.core.events import Bus
from assistant.core.pipeline import ConsoleInputService, start_services
from assistant.memory import store


def run_gui(bus: Bus):
    from assistant.gui.app import JarvisGUI
    root = tk.Tk()
    JarvisGUI(root, bus)
    start_services(bus)

    def _on_close():
        # Stop audio + signal all services, give them a beat, then tear down
        # cleanly so the process exits 0 instead of aborting the audio backend.
        bus.interrupt.set()
        bus.shutdown.set()
        root.after(150, root.destroy)

    root.protocol("WM_DELETE_WINDOW", _on_close)
    try:
        root.mainloop()
    finally:
        bus.interrupt.set()
        bus.shutdown.set()
        store.flush(timeout=2.0)
        os._exit(0)  # daemon audio threads can hang process exit; force clean code


def run_console(bus: Bus):
    start_services(bus)
    console = ConsoleInputService(bus)
    console.start()
    try:
        bus.shutdown.wait()
    except KeyboardInterrupt:
        bus.shutdown.set()
    finally:
        store.flush(timeout=2.0)


def warmup_local_model():
    """Warm up the local Ollama Qwen model if routing or ollama is enabled."""
    if config.LLM_ROUTING_ENABLED or config.LLM_PROVIDER == "ollama":
        try:
            from assistant.llm.ollama import OllamaClient
            client = OllamaClient()
            client.warmup(timeout=35.0)
        except Exception as e:
            from assistant.core.log import get
            get("main").warning("[WARMUP] Startup warmup failed (non-fatal): %s", e)


def main():
    bus = Bus()
    warmup_local_model()
    if config.FLAGS.demo_mode:
        run_console(bus)
    else:
        run_gui(bus)


if __name__ == "__main__":
    main()
