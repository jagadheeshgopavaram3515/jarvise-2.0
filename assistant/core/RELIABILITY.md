# Jarvis Reliability Layer (V2)

Additive resilience so no single exception, failed API call, dead thread, network
outage, or tool failure can crash the assistant. **No features changed, no
architecture redesigned, no behaviour altered** — happy paths are byte-identical;
this only adds guards, a health monitor, and graceful fallbacks.

---

## 1. Reliability audit (Phase 1)

| Thread / worker | Before | Risk found | Hardened by |
|---|---|---|---|
| **AudioInput** (`capture.py`) | `run()` wrapped once; on mic error the thread **exited** | mic unplug / PortAudio glitch → permanent loss of input | self-healing reopen loop with capped back-off |
| **STT** (`recognizer.py`) | loop body unguarded; transcribe guarded in executor | a malformed item or routing error kills the thread → silence + costly model reload | per-iteration `try/except` in the consume loop |
| **TTS** (`service.py`) | first-item synth call **unguarded** → `NoAudioReceived` killed the thread | one bad sentence silenced Jarvis for the session | engine degrades to silence + loop-body guard (done earlier) |
| **Dispatcher** (`dispatcher.py`) | already wraps `_route` | — | left as-is (already safe) |
| **Backchannel** (`backchannel.py`) | playback + LLM guarded; loop frame unguarded | a state error could kill the worker | whole-iteration guard |
| **Reminder** (`scheduler.py`) | **no** `try/except` | a clock/store/TTS hiccup kills the scheduler | per-tick guard |
| **ToolExecutor workers** | already timeout + catch-all | — | left as-is (already safe) |
| **STT/TTS/BC thread pools** | task exceptions swallowed by futures | silent failures only (pool thread survives) | acceptable; key paths log |

**Not separate threads (audit clarification):** there is **no** standalone
`MemoryWorker` or `Realtime worker` thread — memory consolidation and realtime
fetch run **inline** inside the Dispatcher/Gemini call path and are already wrapped
in `try/except` (memory failures and realtime failures degrade without breaking the
turn). The spec's named workers map to those inline, already-guarded paths.

**Cross-cutting issues addressed:** unbounded restart loops (capped), corrupt JSON
state files (safe load/atomic dump helpers), and thread death with no recovery
(HealthMonitor).

---

## 2. Recovery architecture

```
                         ┌──────────────────────────────────────────┐
                         │            HealthMonitor (daemon)          │
                         │  every HEALTH_INTERVAL_S (10s):            │
                         │   • is_alive() every registered service    │
                         │   • [HEALTH] heartbeat line                 │
                         │   • dead & critical → CRITICAL log          │
                         │                     → factory() restart     │
                         │   • restarts capped at HEALTH_MAX_RESTARTS  │
                         └───────────────┬──────────────────────────┘
              registers + restart factory │ (pipeline.start_services)
        ┌───────────────┬─────────────────┼─────────────────┬───────────────┐
        ▼               ▼                 ▼                 ▼               ▼
   ┌─────────┐    ┌──────────┐      ┌───────────┐     ┌──────────┐   ┌──────────┐
   │AudioInput│   │   STT    │      │ Dispatcher│     │   TTS    │   │ Reminder │
   │self-heal │   │loop guard│      │ route grd │     │synth grd │   │ tick grd │
   └─────────┘    └──────────┘      └───────────┘     └──────────┘   └──────────┘
   each: while running: try: process() except: log; continue   ← survivability

   Service isolation: every box talks ONLY through the Bus queues; a failure in
   one (Gemini / tool / memory / realtime) is caught locally and never propagates.
```

**Restart re-wiring:** factories live in `pipeline.start_services`. They rebuild
**and** re-wire shared references — e.g. restarting TTS re-points
`bus.backchannel.tts` at the fresh engine; restarting Backchannel resets
`bus.backchannel`. The shared `ReminderStore` is reused so reminders survive a
Reminder-thread restart.

---

## 3. Service isolation & graceful degradation (Phase 5 / 9)

| Failure | Result |
|---|---|
| Gemini down / 429 | `GeminiClient` fallback reply; memory, tools, TTS unaffected |
| A tool throws | `ToolExecutor` returns `{success:false}`, Jarvis speaks the error, conversation continues |
| Memory JSON corrupt | `safe_json_load` returns default; conversation continues on history only |
| Embeddings unavailable | existing `auto` backend already falls back (sentence-transformers → gemini → hash) |
| Realtime/web down | realtime context skipped; Gemini answers from base knowledge |
| Playwright missing | browser **read** tools return a clean spoken error; everything else works |
| Mic unplugged | AudioInput reopens the stream with back-off |
| Any service thread dies | HealthMonitor restarts it within one interval, no Jarvis restart |

---

## 4. Structured logging (Phase 7)

Standard levels, used consistently: `INFO` normal · `WARNING` degraded-but-continuing
· `ERROR` an operation failed · **`CRITICAL` a service died → the only level that
triggers auto-recovery**. Heartbeats log as `[HEALTH] …`, recoveries as
`[RECOVERY] …`.

---

## 5. Config (all optional, default-safe)

| Key | Default | Meaning |
|---|---|---|
| `HEALTH_MONITOR_ENABLED` | `true` | master switch (false = run unsupervised, as before) |
| `HEALTH_INTERVAL_S` | `10` | heartbeat cadence |
| `HEALTH_AUTO_RECOVERY` | `true` | restart dead critical threads |
| `HEALTH_MAX_RESTARTS` | `5` | per-service restart cap (anti-flap) |
| `HEALTH_RESTART_BACKOFF_S` | `2.0` | settle time after a restart |
| `HEALTH_LOG_HEARTBEAT` | `true` | emit the `[HEALTH]` line each cycle |

---

## 6. Bluye readiness (Phase 10)

| Target | Stability outlook | Notes / recommendations |
|---|---|---|
| **Desktop** | ✅ strong | Reference platform. HealthMonitor + guards cover transient faults. |
| **Docker** | ✅ good | Headless: set `DEMO_MODE`/no-mic or pass an audio device. Browser tools need `playwright install chromium` in the image. Mount memory/cache/logs as volumes so state + `logs/tools.log` persist. |
| **Raspberry Pi** | ⚠️ tune | CPU-bound STT is the bottleneck. Use `STT_MODEL_SIZE=tiny`/`base`, `STT_DEVICE=cpu`, set `STT_CPU_THREADS=4`. Prefer Piper TTS (`TTS_BACKEND=piper`) to cut Edge network round-trips. Memory embeddings: `MEMORY_EMBED_BACKEND=hash` if sentence-transformers is too heavy. |
| **Bluye (motorcycle)** | ⚠️ plan for offline | Network is intermittent: Gemini/realtime/Edge-TTS will drop — already degrade gracefully, but for true offline use Piper TTS + a local LLM behind the same `GeminiClient` interface. Long-run: HealthMonitor handles thread faults; watch RAM (whisper + embeddings). |

**Long-run concerns & mitigations**
- **Memory pressure:** whisper model(s) + sentence-transformers dominate RAM. On Pi/Bluye pick one small whisper model and the `hash` embed backend. Working memory is already bounded (`MEMORY_WORKING_TURNS`); conversation history is capped.
- **CPU bottlenecks:** STT decode is the hot path; partials add load — disable with `STT_ENABLE_PARTIALS=false` on weak CPUs.
- **Resource leaks:** thread pools are bounded (max_workers=1–4); queues are bounded with drop policies; atomic JSON writes avoid corruption on power loss. The restart cap prevents runaway thread creation.
- **Continuous operation:** daemon threads + supervised loops + capped auto-recovery → designed to run for days. The HealthMonitor itself is supervised (its loop can't die).

---

## 7. Tests

`tests/test_resilience.py` — 13 failure-injection tests (safeguards, supervised
loop survival, health detect/restart/cap, TTS synth failure, STT transcribe
failure, tool failure isolation, corrupt-JSON degradation). Run:

```bash
python tests/test_resilience.py        # or: pytest -q tests/test_resilience.py
```
