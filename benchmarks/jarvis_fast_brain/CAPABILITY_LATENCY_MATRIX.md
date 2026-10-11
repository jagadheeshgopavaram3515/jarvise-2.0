# Phase C — Capability and Latency Matrix

This is a measured/source-grounded matrix, not a winner ranking. STT and TTS are excluded.

| Request class | Current path | Qwen involvement | Tool/other | Observed latency/evidence | Reliability evidence |
|---|---|---|---|---|---|
| Greeting | `LOCAL` route / Ollama | conversational Qwen | none | Smoke Base Qwen 22.75 s including benchmark prompt | One synthetic case; JSON action semantics poor |
| Simple raw response | direct Ollama | raw Qwen | none | 2.39–2.60 s for repeated `READY` | 3 samples only |
| Deterministic app open | direct tool router | none or Gemini selector | `open_app` | Not live-measured here | Existing patched unit test |
| App close | direct tool router | none or Gemini selector | `close_app` | Not live-measured here | Existing safety test |
| File search | direct deterministic or Gemini selector | not LocalAgent-advertised | `find_file`/`search_files` | Not live-measured here | Existing tests |
| Project inspection | TASK graph | plan/action/final Qwen | `grep_code`, `read_file`, listing/tests | Controlled fixture only | Loop verified; quality not statistically measured |
| Browser open/search | direct tool router or Gemini selector | not LocalAgent-advertised | browser tools | Not live-measured here | Source/tests; no live browser |
| Weather | TASK graph | plan/action/final Qwen | `location`, `weather` | Not live-measured here | Existing unit tests |
| Multi-step task | TASK graph | multiple Qwen calls | multiple Executor calls | Controlled fixture loop verified | No aggregate dataset yet |
| Ambiguous request | route-dependent | Qwen may decide | possible tool | Smoke `Open it`: 4.04 s | Clarification failure observed |
| Current research | conversational real-time path or Gemini | provider-dependent | RealTimeManager/browser/Gemini | Not benchmarked as a full trajectory | No controlled research fixture yet |
| Tool failure | TASK graph | next Qwen selection | Executor error result | Not statistically measured | Recovery path exists, policy not guaranteed |

## Measured bottleneck observations

The raw local model dominates the simple repeated-call sample at roughly 2.4–2.6 seconds wall time. The two-case Phase B smoke replay ranged from 4.0 to 22.8 seconds because the benchmark prompt and generated JSON trajectory differ; it is too small to decompose reliably. The current architecture can add classifier, plan, action, observation, and synthesis calls, so multi-step total latency can be several local model calls plus tool time.

The current data does not justify a numerical Fast Brain budget. A future budget must be derived after a larger warm p50/p95 sample and direct-path timings. A reasonable design constraint is that a Fast Brain should be materially cheaper than the Qwen call it is intended to avoid; the exact threshold remains unmeasured.

## Coverage gap

There is no adequate reviewed dataset yet for intent accuracy, argument exactness, multilingual behavior, context resolution, failure recovery, or end-to-end task success. The matrix must not be interpreted as a complete capability score.
