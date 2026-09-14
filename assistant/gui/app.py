"""
JarvisGUI — Tkinter front-end, now fully queue-driven.

Tkinter MUST run on the main thread, so the GUI never calls into services
directly. Instead it polls gui_q with root.after() and applies events. The
spacebar triggers barge-in (instant stop) via the bus.

Visuals (pulsing rings, system stats, conversation log) are unchanged from
the original gui_module.py.
"""
from __future__ import annotations

import math
import queue
import tkinter as tk
import tkinter.font as tkFont
from datetime import datetime

import psutil

from assistant.core.events import Bus, Mode


class JarvisGUI:
    def __init__(self, root: tk.Tk, bus: Bus):
        self.root = root
        self.bus = bus
        root.title(f"JARVIS Assistant")
        root.configure(bg="black")
        root.geometry("1400x900")

        self.canvas = tk.Canvas(root, width=600, height=600, bg="black", highlightthickness=0)
        self.canvas.pack(pady=50)
        self.subtitle_font = tkFont.Font(family="Helvetica", size=16, weight="bold")

        self.conversation_box = tk.Text(
            root, fg="#00FFFF", bg="black", font=self.subtitle_font,
            height=10, width=60, wrap="word", bd=0, padx=10, pady=10)
        self.conversation_box.place(x=700, y=650)
        self.conversation_box.configure(state="disabled")

        self.status_label = tk.Label(root, text="🟢 Awake", fg="#00FF00",
                                     bg="black", font=("Helvetica", 20, "bold"))
        self.status_label.place(x=1180, y=70)

        self.rings = []
        self.base_radius, self.ring_count, self.pulse_range = 200, 5, 20
        self.colors = ["#00ffff", "#0099cc", "#005577", "#003344", "#001122"]
        for i in range(self.ring_count):
            scale = self.base_radius + i * 15
            self.rings.append(self.canvas.create_oval(
                350 - scale, 350 - scale, 350 + scale, 350 + scale,
                outline=self.colors[i], width=3))
        self.is_sleeping = False

        self.cpu_label = tk.Label(root, text="CPU Speed: --", fg="cyan", bg="black", font=("Helvetica", 18))
        self.cpu_label.place(x=100, y=140)
        self.task_label = tk.Label(root, text="Task Manager: --", fg="cyan", bg="black", font=("Helvetica", 18))
        self.task_label.place(x=100, y=100)
        self.time_label = tk.Label(root, text="Time: --", fg="cyan", bg="black", font=("Helvetica", 18))
        self.time_label.place(x=100, y=800)

        # Spacebar = barge-in (stop talking immediately).
        root.bind("<space>", self._on_space)

        self._angle = 0
        self._animate_rings()
        self._update_info()

        # Start polling the event bus on the main loop.
        self.root.after(50, self._pump)

    # ---- bus pump (main thread) ----------------------------------------
    def _pump(self):
        try:
            for _ in range(64):  # drain a batch per tick
                ev = self.bus.gui_q.get_nowait()
                self._apply(ev)
        except queue.Empty:
            pass
        if not self.bus.shutdown.is_set():
            self.root.after(50, self._pump)
        else:
            self.root.quit()

    def _apply(self, ev):
        if ev.kind == "status":
            self.update_status(ev.text)
        elif ev.kind == "subtitle":
            if ev.text:
                self.update_subtitle(ev.text, role=ev.role)
        elif ev.kind == "notification":
            self.show_notification(ev.text)
        elif ev.kind == "mode":
            self.is_sleeping = ev.text == "sleep"
            self.status_label.config(
                text="🔴 Sleeping" if self.is_sleeping else "🟢 Awake",
                fg="red" if self.is_sleeping else "#00FF00")

    def _on_space(self, _event=None):
        self.bus.request_interrupt()
        self.show_notification("Speech stopped (Spacebar)")

    # ---- visuals -------------------------------------------------------
    def update_status(self, message: str):
        if message.lower() not in ("listening...", "recognizing..."):
            self.status_label.config(text=message, fg="cyan")

    def update_subtitle(self, text: str, role: str = "assistant"):
        self.conversation_box.configure(state="normal")
        if role == "user":
            prefix, color = "You: ", "#00FF00"
        else:
            prefix, color = "Jarvis: ", "#00FFFF"
        self.conversation_box.insert("end", prefix + text + "\n")
        start = f"end-{len(text) + len(prefix) + 1}c"
        self.conversation_box.tag_add(role, start, "end-1c")
        self.conversation_box.tag_config(role, foreground=color)
        self.conversation_box.see("end")
        self.conversation_box.configure(state="disabled")

    def show_notification(self, message: str, duration: int = 3000):
        notif = tk.Toplevel(self.root)
        notif.overrideredirect(True)
        notif.configure(bg="black")
        x = self.root.winfo_screenwidth() - 320
        notif.geometry(f"300x80+{x}+100")
        tk.Label(notif, text=message, fg="white", bg="black",
                 font=("Helvetica", 14, "bold"), wraplength=280,
                 justify="left", anchor="w", padx=10, pady=10).pack(fill="both", expand=True)
        notif.attributes("-topmost", True)
        notif.after(duration, notif.destroy)

    def _animate_rings(self):
        if self.bus.shutdown.is_set() or not self.root.winfo_exists():
            return
        if not self.is_sleeping:
            for i, ring in enumerate(self.rings):
                offset = math.sin(math.radians(self._angle + i * 20)) * self.pulse_range
                scale = self.base_radius + i * 15 + offset
                self.canvas.coords(ring, 350 - scale, 350 - scale, 350 + scale, 350 + scale)
            self._angle = (self._angle + 5) % 360
        self.root.after(50 if not self.is_sleeping else 1000, self._animate_rings)

    def _update_info(self):
        if self.bus.shutdown.is_set() or not self.root.winfo_exists():
            return
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_speed = cpu_freq.current if cpu_freq else 0
            self.cpu_label.config(text=f"CPU Speed: {cpu_speed:.0f} MHz")
            self.task_label.config(
                text=f"Task Manager: CPU: {psutil.cpu_percent()}% | "
                     f"Mem: {psutil.virtual_memory().percent}%")
            self.time_label.config(text=f"Time: {datetime.now():%I:%M:%S %p}")
        except Exception:
            pass
        self.root.after(1000, self._update_info)
