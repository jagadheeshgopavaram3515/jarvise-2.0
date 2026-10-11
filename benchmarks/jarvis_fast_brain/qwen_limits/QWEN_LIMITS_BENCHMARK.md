# Qwen Limits Benchmark — Phase D

## Executive summary

The current evidence establishes an operational distinction, not a complete reliability boundary.

**Qwen is currently useful for:** local text generation, JSON-shaped planning/action proposals, bounded project reasoning, and interpreting fixture observations when the correct prompt/interface is supplied.

**Qwen is not an executor:** raw Qwen cannot open Chrome, search Google, read files, or call Python. JARVIS ToolRegistry/Executor performs those actions.

**Current measured weaknesses:** the two-case smoke replay produced a parseable but semantically invalid greeting action and invented `example.txt` for “Open it.” The current LocalAgent prompt exposes six of 19 tools, which is a JARVIS interface limitation as well as a measurement confounder.

**Context:** model metadata says 32K, but the active runtime reported 4096. The 1K–32K probe returned successfully at every requested level, yet larger requests had `prompt_eval_count=2050`, showing truncation/capping rather than proof of usable 32K semantics.

## Environment and configuration

| Field | Observation |
|---|---|
| Model | `qwen2.5-coder:3b` |
| Format | GGUF, Q4_K_M |
| Parameters | 3.1B |
| Ollama | Reachable at `http://localhost:11434`; CLI/server version not collected |
| Model metadata context | 32768 |
| Active `/api/ps` context | 4096 |
| Active VRAM record | 2,155,169,709 bytes |
| JARVIS Ollama timeout | 45 s |
| Classifier timeout | 5 s |
| Keep-alive | 10m |
| Conversation temperature | 0.3 |
| Classifier temperature/output | 0.0 / 5 tokens |
| Agent JSON temperature | 0.1, no explicit cap |
| Final synthesis | 0.2 / 250 tokens |

## Three modes

### Raw Qwen

Direct prompts showed explanatory text for opening Chrome, refusal for one web-search prompt, and valid-looking JSON that did not match JARVIS’s action schema. No execution occurred.

### Qwen + current LocalAgent contract

The agent sends plan/action JSON, validates names through Registry, executes via Executor, formats observations, and can loop up to eight iterations. The six advertised tools are location, weather, read_file, list_directory, grep_code, and run_project_tests.

### Qwen + complete Registry description

Not run as a production change. This is the next controlled experiment needed to separate model limits from capability-exposure limits. It must use fixture execution and identical cases.

## Complexity boundary

| Level | Current evidence |
|---|---|
| 0 conversation | Raw generation works; smoke agent schema behavior is not semantically reliable |
| 1 deterministic request | Direct router/tool paths exist; Qwen is not proven necessary |
| 2 single tool | Project tools are wired through LocalAgent; broad 19-tool Qwen accuracy is unmeasured |
| 3 contextual task | Some bounded history is passed; semantic resolution is unmeasured |
| 4–5 multi-step | Graph loop exists and one controlled project fixture succeeded; useful ceiling unknown |
| 6 recovery | Failure observation is fed back; recovery success rate is unmeasured |
| 7 research | RealTimeManager/Gemini paths exist; controlled Qwen research performance is unmeasured |
| 8 complex agentic | INSUFFICIENT EVIDENCE |

## Direct answers

1. Reliable current envelope: bounded local generation and structured proposals, not independently verified end-to-end task reliability.
2. With the right tools/context: project inspection and observation-driven agent loops are demonstrably possible in fixtures.
3. Frequent observed errors: action-schema mismatch and target invention in the tiny smoke set.
4. Hallucination modes: invented tool names in a greeting response and invented target `example.txt` for an ambiguous request.
5. Practical context limit: JARVIS-active runtime currently reports 4096; semantic useful limit is not established.
6. Is 32K usable? **Not established.** The runtime evidence suggests the current JARVIS/Ollama session is capped or otherwise evaluating only about 4096-context behavior.
7. 1K–32K probes: all returned the marker, but 24K/32K reported prompt evaluation counts of 2050, so they do not prove 24K/32K semantic use.
8. Reliable tool-step ceiling: **INSUFFICIENT EVIDENCE**; only a controlled one-tool trajectory is demonstrated.
9. Recovery: loop support exists; quality/rate is unmeasured.
10. Contextual follow-up: bounded history is available; reliable resolution is unmeasured.
11. Multilingual/code-switching: STT/router support exists, but Qwen multilingual agent reliability is **INSUFFICIENT EVIDENCE**.
12. Deterministic routing is preferable for exact app/browser/file commands when existing direct matchers cover them, because it avoids a roughly multi-second Qwen call.
13. Qwen is useful for project/code interpretation, contextual reasoning, planning, and observation synthesis where direct matching is insufficient.
14. Gemini candidates: unsupported local capability, externally researched/current or complex multi-source work, and evidence-backed local failure—not every error.
15. Clarification: unresolved pronouns/targets, missing required arguments, and contradictory context.
16. Genuine Qwen limits shown: raw text generation does not execute; semantic/schema discipline and ambiguity handling failed in smoke evidence.
17. JARVIS limitations shown: inconsistent six-vs-19 capability exposure, no hard clarification gate, and legacy fallback hiding classifier transport failures.
18. Future Fast Brain should coordinate path/capability/ambiguity/context/escalation signals.
19. It should not duplicate Qwen planning, ToolRegistry, execution, or final response generation.
20. LoRA candidates: valid JARVIS action schema, real capability names, clarification, context resolution, recovery, multilingual behavior, concise grounded responses.
21. Do not teach permissions, execution policy, or tool security; those belong to Registry/Executor.
22. Practical provider boundary: direct tools for deterministic matches, Qwen for bounded local reasoning and agent loops, clarification for missing targets, Gemini for explicit external/unsupported/escalated work. Thresholds remain to be measured.
