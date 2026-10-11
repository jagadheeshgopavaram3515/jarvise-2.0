# Qwen Failure Taxonomy

| Category | Current evidence | Ownership |
|---|---|---|
| `MODEL_REASONING_FAILURE` | Not separable at scale yet | Qwen candidate |
| `TOOL_SELECTION_FAILURE` | Broad capability selection unmeasured | Qwen/interface |
| `ARGUMENT_FAILURE` | Smoke invented `example.txt` | Qwen/prompt |
| `CONTEXT_FAILURE` | Not measured with grouped distractors | Unknown |
| `MULTILINGUAL_FAILURE` | No Phase D trajectory data | Unknown |
| `AMBIGUITY_FAILURE` | “Open it” target invention | Qwen/interface |
| `HALLUCINATION` | Invented `greeting_tool`; invented file target | Qwen/prompt |
| `TARGET_INVENTION` | Confirmed in smoke case | Qwen/interface |
| `RECOVERY_FAILURE` | Rate unmeasured | Unknown |
| `LOOP` | Graph caps at 8 iterations | Orchestration guard |
| `TIMEOUT` | No Phase D trajectory timeout observed | Environment/model |
| `LATENCY_FAILURE` | Multi-second local calls observed | Model/runtime |
| `PROMPT_EXPOSURE_FAILURE` | 6 advertised vs 19 Registry tools | JARVIS interface |
| `CAPABILITY_DESCRIPTION_FAILURE` | Full-description comparison not run | JARVIS interface |
| `ORCHESTRATION_FAILURE` | Legacy classifier fallback can hide transport failure | JARVIS routing |
| `TOOL_REGISTRY_LIMITATION` | Registry is execution boundary, not semantic planner | System boundary |
| `ENVIRONMENT_LIMITATION` | Active context 4096 vs metadata 32768; embedding probe fallback | Environment/config |

Do not label a failure as a Qwen limitation when the model was not shown the required capability or the fixture/orchestrator failed first.
