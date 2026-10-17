"""FastAPI transport for the desktop Bus; the brain and speech workers stay intact."""
from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict

import psutil
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from assistant.core.events import Bus, Transcript, new_utterance_id
from assistant.core.perf import Perf, tracker

ALLOWED_ORIGINS = {"http://127.0.0.1:3000", "http://localhost:3000",
                   "http://127.0.0.1:8765", "http://localhost:8765"}


class Runtime:
    def __init__(self, starter=None):
        self.bus = Bus()
        self.session_id = uuid.uuid4().hex
        self.phase = "offline"
        self.ready = False
        self.events = deque(maxlen=200)
        self.sequence = 0
        self.started_at = None
        self._lock = threading.Lock()
        self._starter = starter
        self._thread = None
        self._system = {}
        self._system_at = 0.0

    def start(self):
        with self._lock:
            if self._thread is not None:
                return
            self.phase = "initializing"
            self.started_at = time.monotonic()
            self._thread = threading.Thread(target=self._boot, name="WebPipeline", daemon=True)
            self._thread.start()

    def _boot(self):
        try:
            if self._starter is None:
                from assistant.core.pipeline import start_services
                self._starter = start_services
            self._starter(self.bus)
            self.ready = True
            self.phase = "listening"
        except Exception:
            from assistant.core.log import get
            get("web").exception("Pipeline startup failed")
            self.phase = "error"
            self.bus.post_gui("notification", "Startup failed. Check the JARVIS log and restart the backend.")

    def drain_events(self):
        for _ in range(256):
            try:
                event = self.bus.gui_q.get_nowait()
            except queue.Empty:
                break
            self.sequence += 1
            self.events.append({"id": self.sequence, "timestamp": time.time(), **asdict(event)})
            if event.kind == "status":
                value = event.text.lower()
                for phase in ("listening", "thinking", "speaking", "recognizing", "initializing"):
                    if phase in value:
                        self.phase = phase
                        break
            elif event.kind == "mode":
                self.phase = "sleeping" if event.text == "sleep" else "listening"

    def snapshot(self):
        self.drain_events()
        if time.monotonic() - self._system_at > 1:
            freq = psutil.cpu_freq()
            self._system = {"cpu": psutil.cpu_percent(), "memory": psutil.virtual_memory().percent,
                            "frequency": round(freq.current) if freq else 0}
            self._system_at = time.monotonic()
        return {
            "session_id": self.session_id,
            "phase": self.phase, "ready": self.ready,
            "microphone_ready": self.bus.stt_ready.is_set(),
            "speaking": self.bus.state.speaking,
            "audio": {"mic_level": self.bus.state.audio_state.mic_level,
                      "output_level": self.bus.state.audio_state.output_level,
                      "user_speaking": self.bus.state.audio_state.user_speaking},
            "mode": self.bus.state.mode.name.lower(),
            "events": list(self.events),
            "system": {**self._system,
                       "uptime": round(time.monotonic() - self.started_at) if self.started_at else 0},
            "report": tracker.snapshot(),
        }

    def submit(self, text):
        if not self.ready or self.bus.shutdown.is_set():
            raise HTTPException(409, "Start JARVIS and wait for it to be ready.")
        # Same queue used by console input and STT; never bypass the dispatcher.
        perf = Perf(t_start=time.perf_counter())
        transcript = Transcript(text=text, utterance_id=new_utterance_id(), perf=perf,
                                interrupt_epoch=self.bus.interrupt_epoch,
                                during_speech=self.bus.state.speaking)
        try:
            self.bus.transcript_q.put_nowait(transcript)
        except queue.Full:
            raise HTTPException(429, "JARVIS is busy. Try again shortly.") from None
        self.bus.post_gui("subtitle", text, role="user")

    async def close(self):
        self.bus.interrupt.set()
        self.bus.shutdown.set()
        if self._thread:
            await asyncio.to_thread(self._thread.join, 2)
            from assistant.memory import store
            await asyncio.to_thread(store.flush, timeout=2.0)


class ChatInput(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


def create_app(runtime=None):
    runtime = runtime or Runtime()

    @asynccontextmanager
    async def lifespan(app):
        yield
        await runtime.close()

    app = FastAPI(title="JARVIS Local API", lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        if request.method == "POST" and request.headers.get("origin") not in ALLOWED_ORIGINS | {None}:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Local interface origin required"}, status_code=403)
        return await call_next(request)

    @app.get("/api/state")
    async def state():
        return runtime.snapshot()

    @app.post("/api/start")
    async def start():
        runtime.start()
        return {"accepted": True}

    @app.post("/api/chat")
    async def chat(body: ChatInput):
        text = body.text.strip()
        if not text:
            raise HTTPException(422, "Enter a message.")
        runtime.submit(text)
        return {"accepted": True}

    @app.post("/api/interrupt")
    async def interrupt():
        if not runtime.ready:
            raise HTTPException(409, "JARVIS is not running.")
        runtime.bus.request_interrupt()
        runtime.bus.post_gui("status", "Listening...")
        return {"accepted": True}

    @app.post("/api/shutdown")
    async def shutdown():
        await runtime.close()
        return {"accepted": True}

    @app.get("/api/report")
    async def report():
        return runtime.snapshot()

    @app.websocket("/ws")
    async def stream(socket: WebSocket):
        if socket.headers.get("origin") not in ALLOWED_ORIGINS:
            await socket.close(code=1008)
            return
        await socket.accept()
        try:
            while True:
                await socket.send_json(runtime.snapshot())
                await asyncio.sleep(0.05)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


app = create_app()
