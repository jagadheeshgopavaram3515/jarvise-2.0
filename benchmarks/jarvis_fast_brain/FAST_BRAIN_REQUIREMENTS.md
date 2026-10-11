# Phase C — Fast Brain Requirements (Design Only)

## Evidence boundary

Phase C did not implement or integrate a Fast Brain. Current evidence shows: direct deterministic routes already exist for some app/browser phrases; Qwen is relatively expensive per local call; the LocalAgent can plan and loop but advertises only six of 19 Registry tools; and clarification is not enforced. These observations define requirements, not production routing changes.

## Minimum responsibilities

The future Fast Brain should answer only the questions needed before Qwen:

1. Is this conversational, deterministic action, research, multi-step action, or ambiguous?
2. Is there a known direct capability family in the actual Registry/direct router?
3. Is required context missing or resolvable from bounded references?
4. Should Qwen receive the request, and which minimal capability description/context matters?
5. Is clarification indicated?
6. Is escalation indicated because the local path is unsupported or demonstrably insufficient?

It should not execute tools, generate plans, browse, synthesize long answers, validate permissions, or duplicate Orchestrator state.

## Contract

Reuse `ClassifierDecision` from `assistant/orchestration/decision.py`: `intent`, `execution_path`, `request_modes`, `selected_capability`, `context_references`, `missing_information`, `confidence`, `needs_clarification`, and `needs_escalation`. Raw adapter IDs and timing remain internal. No second production schema is required.

## Design-only routing requirements

| Situation | Candidate future path | Evidence/reason |
|---|---|---|
| Exact deterministic app/browser phrase | Direct tool | Existing direct matcher can avoid a Qwen call; must remain protected by Executor |
| Project inspection or weather chain | Qwen agent | Existing LocalAgent already advertises these six capabilities and supports observations/loops |
| Context-heavy follow-up | Qwen with bounded context | No deterministic pronoun resolver exists today |
| Missing target | Clarification | Smoke replay showed Qwen can invent a target; explicit signal is needed |
| Unsupported Registry capability | Gemini or clarification | LocalAgent prompt does not expose all 19 capabilities; no automatic assumption is safe |
| Repeated local failure | Escalation candidate | Current graph only exposes another Qwen iteration; evidence thresholds still need measurement |

These are future design requirements, not enabled routing.

## Latency requirement

No fixed millisecond target is justified yet. The available warm raw sample is only three calls with an approximate mean of 2.48 seconds, and direct tool timings were not collected in this Phase C run. The Fast Brain’s maximum acceptable latency should be calculated after: (a) a statistically adequate warm Qwen p95, (b) deterministic tool p50/p95, and (c) measured avoided-call probability. It should be low enough that adding it does not erase the latency saved by bypassing unnecessary Qwen calls.

## Future LoRA candidates

Only benchmark-confirmed weaknesses should become training candidates. Current preliminary candidates are clarification instead of target invention, use of actual capability names, valid JARVIS action JSON, bounded context resolution, observation-grounded recovery, and concise final responses. These are hypotheses, not a conclusion that LoRA is required.

## Exact next phase

Expand the fixture-backed dataset with grouped reviewed cases across all 19 Registry capabilities, run repeated base trajectories, add Ollama duration/memory capture, and separately benchmark the direct deterministic and Gemini selector paths. Then derive Fast Brain thresholds from measured failure and latency distributions. Do not integrate the design into production yet.
