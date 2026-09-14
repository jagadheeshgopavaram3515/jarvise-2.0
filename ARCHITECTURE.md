# Jarvis 2.0 — Real-Time Full-Duplex Voice Assistant

A refactor of the original `jarvice.py` monolith into a non-blocking,
producer/consumer architecture. **Same features**, but low-latency,
interruptible, and multilingual (Telugu / Hindi / English / code-switched).

---

## 1. Architecture diagram

```
                        ┌───────────────────────────────────────────────┐
                        │                   THE BUS                       │
                        │  queues: stt_in_q, transcript_q, tts_q, gui_q   │
                        │  events: shutdown, interrupt                    │
                        │  state : mode(awake/sleep), speaking flag       │
                        └───────────────────────────────────────────────┘

  MIC                                                                   SPEAKER
   │                                                                       ▲
   ▼                                                                       │
┌──────────────┐  stt_in_q   ┌──────────────┐ transcript_q ┌─────────────┐│ tts_q
│ AudioInput   │ ──────────▶ │     STT      │ ───────────▶ │ Dispatcher  ││◀──────┐
│ sounddevice  │             │faster-whisper│              │             ││       │
│  + WebRTC VAD│             │ (local, int8)│              │ cmd? ─▶ run ││  ┌────┴─────┐
└──────┬───────┘             └──────────────┘              │ else ─▶ LLM ││  │   TTS    │
       │ keeps recording                                   │  (stream)   ││  │ edge_tts │
       │ WHILE speaking                                     └──────┬──────┘│  │ in-memory│
       │ (full-duplex)                                             │ sentences  │+ stop   │
       │                                                           └────────────│──────────│
       │                      interrupt event (barge-in: "stop"/"cancel"/name,  │  └─────────┘
       └──────────────────────  spacebar, or any speech) ──────────────────────┘

  ┌──────────────┐         ┌──────────────┐
  │ GUI (main TH)│◀─gui_q──│  Reminder    │  fires due reminders → tts_q
  │ tkinter+after│         │  thread      │
  └──────────────┘         └──────────────┘
```

The Bus ([assistant/core/events.py](assistant/core/events.py)) is the only thing
services share. No service ever calls another directly, so nothing blocks.

---

## 2. Folder structure

```
jarvis/
├── main.py                      # entry point + threading-model docstring
├── .env.example                 # all secrets/config (copy → .env)
├── requirements-assistant.txt
├── ARCHITECTURE.md              # this file
└── assistant/
    ├── config.py                # loads .env, all tunables, voice map
    ├── core/
    │   ├── events.py            # Bus: queues, events, AppState  ← backbone
    │   ├── pipeline.py          # wires + starts all service threads
    │   └── dispatcher.py        # routing brain: barge-in / wake / cmd / LLM
    ├── audio/
    │   ├── capture.py           # AudioInputService (mic + VAD producer)
    │   └── vad.py               # WebRTC VAD (energy-gate fallback)
    ├── stt/
    │   └── recognizer.py        # FasterWhisperSTT + STTService
    ├── tts/
    │   ├── engine.py            # EdgeStreamingTTS (in-memory, interruptible)
    │   └── service.py           # TTSService consumer + speaking flag
    ├── llm/
    │   ├── gemini.py            # streaming, multilingual GeminiClient
    │   └── realtime.py          # serpapi search / calendarific / ip-geo
    ├── memory/
    │   └── store.py             # name, topics, rolling conversation history
    ├── commands/
    │   └── handlers.py          # deterministic command router (fast path)
    ├── reminders/
    │   └── scheduler.py         # ReminderStore + ReminderService thread
    └── gui/
        └── app.py               # JarvisGUI, queue-driven via root.after
```

---

## 3. Threading model

| Thread          | Type   | Role (producer→consumer)                          | Blocking? |
|-----------------|--------|---------------------------------------------------|-----------|
| **AudioInput**  | daemon | mic → VAD segmentation → `stt_in_q`               | only on its own ring buffer |
| **STT**         | daemon | `stt_in_q` → faster-whisper → `transcript_q`      | only on `get()` |
| **Dispatcher**  | daemon | `transcript_q` → command **or** LLM → `tts_q`     | only on `get()` |
| **TTS**         | daemon | `tts_q` → edge_tts synth+play (checks `interrupt`)| only on `get()` |
| **Reminder**    | daemon | timer → `tts_q`                                   | sleeps in 1 s slices |
| **GUI**         | main   | `gui_q` drained via `root.after(50ms)`            | tkinter mainloop |

Rules enforced:
* **No blocking call in any shared path.** Every consumer uses `queue.get(timeout=…)`
  and loops on `shutdown`.
* **Tkinter only touched on the main thread** — services post `GuiEvent`s; the GUI
  polls them. (The original called `gui.root.after` from worker threads, which is racy.)
* `interrupt` (a `threading.Event`) is the single barge-in signal everyone respects.

---

## 4. Queue design

| Queue          | Item            | Producer    | Consumer   | maxsize | Backpressure |
|----------------|-----------------|-------------|------------|---------|--------------|
| `stt_in_q`     | `Utterance`     | AudioInput  | STT        | 64      | drop-newest if full (live audio, stale frames worthless) |
| `transcript_q` | `Transcript`    | STT         | Dispatcher | 64      | block briefly |
| `tts_q`        | `str` (sentence)| Dispatcher/Reminder | TTS | 128 | block briefly |
| `gui_q`        | `GuiEvent`      | everyone    | GUI        | 256     | drop if full (cosmetic) |

* **Sentence-granular `tts_q`** is what makes TTS start fast: the LLM streams
  the first sentence and pushes it while still generating the rest.
* **`request_interrupt()`** sets the event *and drains `tts_q`*, so barge-in
  doesn't keep speaking queued sentences.
* `Utterance.during_speech` is stamped at capture time so the dispatcher knows a
  transcript arrived while talking (barge-in candidate).

---

## 5. Latency budget & how each target is met

| Stage            | Target    | How                                                            |
|------------------|-----------|----------------------------------------------------------------|
| STT              | < 500 ms  | Local faster-whisper `small` int8 on GPU, `beam_size=1`, `vad_filter`, no network (vs. `recognize_google`'s round-trip). VAD emits the utterance the instant a 600 ms silence gap is detected. |
| LLM first token  | < 1500 ms | `generate_content_stream` — we act on the **first sentence**, not the full reply. `gemini-2.0-flash` + concise system prompt + capped 10-turn history. |
| TTS start        | < 200 ms  | First sentence is short → edge_tts returns its first audio quickly; decoded in memory and pushed to `sounddevice` in 1024-frame blocks. No disk write. |
| **Barge-in stop**| ~20 ms    | Playback writes one small block at a time and checks `interrupt` between blocks. |

**Full-duplex:** AudioInput never stops capturing, even while `speaking` is set,
so the user can talk over the assistant.

---

## 6. Performance optimizations applied

1. **Killed disk I/O in TTS** — `edge_tts` streamed to `BytesIO`, decoded with
   `miniaudio`, played via `sounddevice`. No `uuid.mp3` files, no `os.remove`.
2. **Local STT** — removes the network latency and rate limits of
   `recognize_google`; supports Telugu/Hindi/code-switch natively.
3. **Streaming LLM → sentence-chunked TTS** — speak-while-generating.
4. **Greedy decoding + `condition_on_previous_text=False`** for lowest STT latency.
5. **Single VAD-gated capture loop** replaces repeated
   `adjust_for_ambient_noise` (which added ~1 s per listen in the original).
6. **Bounded queues with drop policies** so a slow stage can't unbounded-buffer.
7. **int8 quantization** keeps the `small` model within the GTX 1650's 4 GB.

---

## 7. Migration plan (from `jarvice.py`)

| Old (monolith)                                   | New (module)                                  |
|--------------------------------------------------|-----------------------------------------------|
| Hard-coded `GOOGLE_API_KEY`, `SERPAPI_KEY`, …    | `.env` → `assistant/config.py`                |
| `TTSManager` + `speak()` (disk MP3, blocking)    | `tts/engine.py` + `tts/service.py` (in-mem)   |
| `listen()` (`speech_recognition`+google)         | `audio/capture.py` + `stt/recognizer.py`      |
| `generate_response()` (blocking, full reply)     | `llm/gemini.py` `GeminiClient.stream()`       |
| `should_use_realtime`/`real_time_search`/holidays| `llm/realtime.py`                             |
| `handle_command()`                               | `commands/handlers.py`                        |
| name/memory/history helpers                      | `memory/store.py`                             |
| `reminder_checker()` + global `reminders` list   | `reminders/scheduler.py`                      |
| `assistant_loop()` single while-True             | `core/dispatcher.py` (event-routed)           |
| `gui_module.JarvisGUI` (`after` from threads)    | `gui/app.py` (queue-pumped on main thread)    |
| spacebar `on_spacebar_stop`                       | `gui/app.py._on_space` → `request_interrupt`  |

**Behaviour preserved:** greeting, sleep/awake, `search for`, open/close Chrome &
YouTube, play music, time/date, quit, set/delete reminder, trip-holiday planning,
real-time search, conversation memory, the pulsing-ring GUI.

The old files are left untouched — nothing was deleted. Run the new stack via
`python main.py`; the legacy `jarvice.py` still runs as before.

---

## 8. Step-by-step implementation / bring-up

```bash
# 1. Create .env from the template and add your keys
cp .env.example .env          # then edit GOOGLE_API_KEY (+ optional keys)

# 2. Install deps
pip install -r requirements-assistant.txt
# GPU STT (GTX 1650): also install CUDA torch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Smoke-test the brain WITHOUT a microphone (headless)
#    Set DEMO_MODE=true in .env, then:
python main.py
#    Type: "Fuel entha undi?"  /  "what time is it"  /  "quit"

# 4. Full real-time run (GUI + mic). Set DEMO_MODE=false:
python main.py
#    - Speak normally; interrupt by saying "stop"/"cancel"/"Jarvis" or press SPACE.

# 5. Tuning
#    - Self-interruption (mic hears the speaker)? set BARGE_IN_KEYWORDS_ONLY=true
#    - STT too slow on CPU?  STT_MODEL_SIZE=base  or  tiny
#    - Clipping the start of speech? raise VAD_SILENCE_MS
```

### Known limitations / future work
* **Acoustic echo cancellation (AEC):** true open-air full-duplex with speakers
  (not headphones) can let the mic hear the assistant. Mitigations in place:
  `BARGE_IN_KEYWORDS_ONLY` and VAD gating. For production add an AEC front-end
  (e.g. `speexdsp`/WebRTC APM) before the VAD.
* **Streaming STT:** currently utterance-chunked (VAD endpoint), not token-streaming.
  For sub-300 ms partials, swap in `whisper_streaming` or a cloud streaming API
  behind the same `STTService` interface.
