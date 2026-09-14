# Jarvis V2 — Desktop Tools Layer

An **additive** tool-calling layer that lets Jarvis control the desktop without
touching the voice pipeline. Nothing in the protected core (STT, TTS,
AudioInput, Dispatcher, Bus, MemoryEngine, Gemini client, turn IDs,
interruption, realtime, GUI) was modified. The layer is reusable for **Bluye**:
only `router.py` touches Jarvis (`bus.speak`), everything else is generic.

## Flow (Phase 5)

```
User ─▶ Dispatcher ─▶ commands.handlers.handle()
                          │  legacy deterministic commands first (unchanged)
                          ▼
                    router.try_tools(text, bus)
                          │  1. deterministic intent match   (no API call)
                          │  2. else Gemini tool selection    → {"tool","args"}
                          ▼
                    ToolExecutor.execute(tool, args)           ← allow-list + timeout + audit log
                          │
                          ▼
                    Gemini natural response ─▶ bus.speak(...) ─▶ TTS
```

`try_tools` returns **True** only when a tool handled the request (Dispatcher
skips the conversational LLM). Otherwise it returns **False** and the
conversation continues exactly as before — full backward compatibility.

## Modules

| File          | Responsibility                                                        |
|---------------|----------------------------------------------------------------------|
| `schemas.py`  | `ToolSpec`, canonical `tool_result`/`ok`/`fail` shapes (dep-free)     |
| `registry.py` | `ToolRegistry` (allow-list) + `build_default_registry()`             |
| `executor.py` | `ToolExecutor.execute()` — timeout, isolation, safe errors, audit log |
| `apps.py`     | open/close/list desktop apps (subprocess + psutil, allow-list)        |
| `files.py`    | find/search/open files + folders, recent files — **read-only**        |
| `browser.py`  | open_url / google_search / youtube_search / read_page / get_page_title (Playwright, DOM-only) |
| `router.py`   | Phase 5 tool-calling layer + the single `try_tools` integration hook  |
| `__init__.py` | lazy `get_registry()` / `get_executor()` singletons                   |

## Safety (Phase 6)

* **Allow-list only.** A tool not registered in `ToolRegistry` cannot run.
* **Apps** are an explicit allow-list; Explorer/Calculator are non-closable.
* **Files** are strictly read-only (no delete/rename/move/write) and confined
  to `TOOLS_FILE_ROOTS` (default: the user's home tree).
* **Browser** is Playwright/DOM-only — no Selenium, PyAutoGUI or image clicking.
* No shell-out to arbitrary commands; no registry edits, shutdown, or format.

## Logging (Phase 7)

Every call is appended to `logs/tools.log`:

```
2026-06-19 07:43:38 [TOOLS] open_app {"app": "chrome"} success=true duration=0.23s
```

## Config (all optional, in `assistant/config.py` / `.env`)

| Key                      | Default                 | Meaning                              |
|--------------------------|-------------------------|--------------------------------------|
| `TOOLS_ENABLED`          | `true`                  | master switch for the whole layer    |
| `TOOLS_TIMEOUT_S`        | `25.0`                  | per-tool execution timeout           |
| `TOOLS_LOG_FILE`         | `<root>/logs/tools.log` | audit log path                       |
| `TOOLS_GEMINI_SELECTION` | `true`                  | use Gemini to pick a tool            |
| `TOOLS_SELECTION_MODEL`  | `gemini-2.5-flash-lite` | small model for selection            |
| `TOOLS_NATURAL_RESPONSE` | `true`                  | phrase the spoken reply with Gemini  |
| `TOOLS_BROWSER_HEADLESS` | `true`                  | headless Playwright                  |
| `TOOLS_FILE_ROOTS`       | `~`                     | folders the file tools may search    |

## Install (Phase 4)

```bash
pip install playwright
playwright install chromium
```

## Tests (Phase 8)

```bash
pytest -q tests/test_tools.py        # or:  python tests/test_tools.py
```
