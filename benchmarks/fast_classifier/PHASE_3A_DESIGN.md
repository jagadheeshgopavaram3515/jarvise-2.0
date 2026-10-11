# JARVIS FastClassifier Phase 3A: Failure Analysis and Design

Status: design only. No model was trained, downloaded, integrated, or added to
the runtime. Production routing remains unchanged.

## 1. Evidence and scope

The saved Phase 2 run evaluated 28 reviewed examples through the real
`QwenClassifier.classify(text) -> (route, latency_ms)` interface using installed
`qwen2.5-coder:3b`. It returned 28 results, with no visible exception, timeout,
or invalid legacy route. Five expected `clarify`/`escalate` paths cannot be
represented by the legacy `LOCAL`/`GEMINI`/`TASK` interface and were excluded
from its accuracy denominator. Four of 23 comparable routes were correct
(17.4%). This is a small diagnostic seed, not a population estimate.

Measured latency from `benchmark-results/legacy-qwen-v1.json`:

| Path provenance | N | Mean | p50 | p95 |
|---|---:|---:|---:|---:|
| Existing deterministic fast path | 5 | 0.193 ms | 0.166 ms | 0.298 ms |
| Local Qwen model | 23 | 2463.149 ms | 2457.655 ms | 2545.320 ms |

The first benchmark invocation was 2482.024 ms. Process startup and model-load
time were not measured separately. The legacy implementation catches internal
Ollama errors and returns `GEMINI`, so the public result cannot prove that no
internal failure occurred.

## 2. Complete route failure analysis

### Comparable incorrect predictions

Every comparable error in the saved report is listed below. `Cause` identifies
the directly supported explanation; it does not infer hidden model reasoning.

| ID | Needed behavior | Expected | Actual | Evidence-based cause |
|---|---|---|---|---|
| `en-memory-001` | Recall resolved prior decision | GEMINI | TASK | Prompt has no memory class; model mapped project-like recall to task |
| `en-desktop-001` | Single `open_app` action | TASK | LOCAL | Router never receives capability descriptions; prompt defines TASK narrowly |
| `en-browser-001` | Browser action | TASK | LOCAL | Browser capability/action is absent from TASK policy |
| `en-music-001` | Music search/action | TASK | LOCAL | Music capability/action is absent from TASK policy |
| `en-multistep-001` | Search, read, summarize | TASK | GEMINI | Three-route prompt describes complexity but not general multistep execution |
| `ctx-hologram-001` | Recall resolved project context | GEMINI | LOCAL | Classifier receives text only, not supplied context |
| `ctx-continue-resolved-001` | Continue supplied prior task | TASK | LOCAL | Classifier receives neither context nor capability descriptions |
| `mixed-conversation-action-001` | Conversation plus `open_folder` | TASK | LOCAL | Mixed modes and available capability are invisible to classifier |
| `stt-noisy-action-001` | Noisy `open_app` request | TASK | LOCAL | Wording/STT variation is outside prompt examples |
| `te-native-conversation-001` | Conversation | LOCAL | GEMINI | Native-script deterministic rule forces GEMINI before model inference |
| `te-native-action-001` | `open_app` action | TASK | GEMINI | Native-script deterministic rule overrides action semantics |
| `te-romanized-knowledge-001` | Stable knowledge answer | LOCAL | GEMINI | Romanized-language keyword rule/prompt language priority overrides meaning |
| `te-codeswitch-research-001` | Current research action | TASK | GEMINI | Language/code-switching policy overrides advertised research need |
| `hi-native-conversation-001` | Conversation | LOCAL | GEMINI | Native-script deterministic rule forces GEMINI before model inference |
| `hi-romanized-action-001` | `open_app` action | TASK | LOCAL | Romanized wording is not understood as an available action |
| `ta-native-conversation-001` | Conversation | LOCAL | GEMINI | Native-script deterministic rule forces GEMINI before model inference |
| `ta-romanized-action-001` | `open_app` action | TASK | LOCAL | Romanized wording is not understood as an available action |
| `ta-codeswitch-research-001` | Current research action | TASK | GEMINI | Code-switching is treated as provider selection, not task semantics |
| `ctx-browser-resolved-001` | Open resolved search result | TASK | LOCAL | Context and capability are not inputs to the legacy classifier |

Confusion counts over the 23 comparable examples:

- Correct: `LOCAL->LOCAL` 2, `TASK->TASK` 1, `GEMINI->GEMINI` 1.
- Incorrect: `TASK->LOCAL` 9, `TASK->GEMINI` 4, `LOCAL->GEMINI` 4,
  `GEMINI->TASK` 1, and `GEMINI->LOCAL` 1.
- All eight comparable examples tagged `supported_capability` failed. The
  separate current-information search example was correctly labeled TASK.
- All four native-script examples were routed to GEMINI by the deterministic
  script rule; three were conversations expected LOCAL and one was an action
  expected TASK.
- Romanized/code-switched errors span knowledge, actions, and research, showing
  that language identity is not a useful proxy for execution behavior.

### Legacy-interface limitations

These five predictions are not counted as comparable model errors because the
legacy interface cannot emit their expected path:

| ID | Expected | Why |
|---|---|---|
| `ctx-continue-unresolved-001` | clarify | No prior request resolves “that” |
| `ctx-open-missing-001` | clarify | Target type and identity are missing |
| `cap-browser-missing-001` | clarify | Search query cannot be resolved |
| `cap-unsupported-email-001` | escalate | No advertised email capability |
| `cap-unsupported-home-001` | escalate | No advertised smart-light capability |

The actual `LOCAL` outputs still demonstrate that the live system cannot express
the desired behavior, but route accuracy cannot fairly score a class that is
absent from the output space. `LegacyClassifierAdapter` does not solve this: it
constructs placeholder intent/modes from the route and therefore supplies no
measured semantic evidence.

### Behavioral groups

| Need | Dataset examples | Required decision behavior |
|---|---|---|
| Conversation/knowledge | 6 | Direct response; language alone must not change mode |
| Memory recall | 3 | Resolve prior-conversation/project reference or request clarification |
| Current research | 4 | Select research plus action when an advertised search capability is required |
| Single action | 10 | Match an advertised capability and detect its required information |
| Multistep execution | 1 primary, plus research workflows | Enter bounded task planning; select only the first capability at classification time |
| Clarification | 3 | Do not select or execute a capability while essential information is absent |
| Escalation | 2 | Report that no advertised capability can satisfy the request |

Counts overlap because `request_modes` is intentionally multi-label.

### Attribution boundaries

- Native-script misroutes are directly attributable to `fast_classify`, which
  returns GEMINI before Qwen sees the utterance.
- Other comparable route errors are outputs of the current text-only Qwen
  classifier and its narrow three-route prompt. Context/capability failures are
  also input-interface failures: `classify(text)` cannot inspect either input.
- The five clarification/escalation cases are output-interface failures, not
  fairly measurable legacy class predictions.
- This benchmark calls the classifier directly. It does not measure upstream
  command handling or prove an end-to-end action would fail.
- A correct TASK label does not prove the current TASK agent can execute every
  advertised registry capability. `LocalAgent.select_action` currently presents
  only location, weather, and project-inspection tools in its prompt; desktop
  and browser tools are not among that candidate list. The desired replacement
  needs a later orchestration design for general capabilities, outside Phase 3A.

### Annotation questions; no labels changed

The Phase 2 files remain unchanged. These records deserve adjudication before
training:

1. `en-browser-001`: `open_url` requires a URL, but only a site description is
   supplied. Either advertise `google_search`, provide a resolved canonical URL,
   or label the URL as missing. Current TASK intent is reasonable; exact
   capability readiness is under-specified.
2. `en-music-001`: `youtube_search` can search but does not guarantee playback.
   The current intent `find_and_play_focus_music` overstates the advertised
   capability. Prefer `find_focus_music`, or add a real playback capability.
3. `stt-noisy-action-001`: “calclater” was manually interpreted as calculator,
   but the dataset has no ASR confidence or alternatives. Add those fields or
   allow clarification as an accepted outcome for low-confidence transcripts.
4. `cap-unsupported-email-001`: escalation is correct for unavailable email,
   but a future annotation may also mark the recipient address as missing.
5. Memory examples use GEMINI as a legacy provider-policy label. A replacement
   contract should label semantic execution as memory recall and map that to a
   provider separately.
6. Desktop/browser TASK labels describe desired semantic routing against the
   advertised benchmark capabilities. They do not assert that the current
   inspection-focused LocalAgent is already an appropriate executor.

## 3. Proposed decision contract and flow

### Stable semantics, open-ended intent

Retain the existing `ClassifierDecision` fields and treat `intent` as a compact,
normalized semantic identifier, not a route and not hidden reasoning. Keep the
intent vocabulary compositional and broad (for example, `open:application`,
`recall:prior_decision`, `research:current_information`) rather than adding an
intent for each phrase, language, project, or application.

For the next dataset version, use a small stable execution vocabulary in
`execution_path`:

- `direct_response`: answer or converse without a tool.
- `memory_recall`: retrieve supplied or stored conversational context.
- `research`: gather current/external information before answering.
- `single_action`: one identified capability is sufficient.
- `multi_step_task`: planning or several capability calls are required.
- `clarify`: essential reference or argument is missing.
- `escalate`: no advertised capability/path can satisfy the request safely.

This vocabulary describes execution, unlike `LOCAL` and `GEMINI`, which name
providers. Provider selection should be a separate deterministic policy so model
availability, privacy, cost, and latency can change without retraining semantic
intent. This is a proposed dataset/adapter evolution, not a production change.

### One-pass decision flow

1. Build one `DecisionRequest` containing the utterance, bounded recent context,
   and capability names/descriptions/required arguments. Do not include handlers.
2. Run one compact multilingual decision model. Qwen must not classify first.
3. In that one inference, predict intent, execution path, multi-label request
   modes, capability or `null`, reference resolution state, information
   sufficiency, clarification/escalation, confidence, and evidence pointers.
4. Deterministically serialize the heads into the existing decision shape and
   validate with `ClassifierDecision.from_mapping()`.
5. Apply consistency checks, not semantic phrase rules: clarification requires
   missing information; selected capability must exist in the advertised set;
   action paths require a capability unless clarification/escalation explains
   its absence; confidence must be bounded.
6. Pass the decision to orchestration. The classifier never imports or invokes
   `ToolExecutor`. Existing registry, permission, argument validation, timeout,
   and audit layers remain authoritative.

`evidence` should contain short references such as input spans, context keys, or
capability IDs (`input_span: "the second one"`, `context_key:
"search_results[1]"`, `capability: "open_url"`). It must not contain a rationale
paragraph or chain-of-thought.

## 4. Proposed model and adapter architecture

### Recommended first candidate

Use a single compact multilingual encoder with several task heads and a
deterministic serializer:

- Shared pretrained multilingual text encoder for English, Telugu, Hindi,
  Tamil, native scripts, romanization, and code-switching.
- Execution-path softmax head.
- Four-label sigmoid head for `RequestMode`.
- Dynamic capability matcher that scores the utterance/context representation
  against the advertised capability descriptions, plus `none` and
  `unavailable` outcomes. New tools therefore do not require phrase rules or a
  classifier code change.
- Heads for reference resolved/unresolved, information sufficiency,
  clarification, escalation, and calibrated confidence.
- Span or context-key pointer outputs for evidence. The agent remains
  responsible for detailed arguments and execution planning.
- Deterministic Python construction of `ClassifierDecision`; the model does not
  generate arbitrary JSON. This makes schema validity easier to guarantee than
  unconstrained free-form generation.

Intent should be compositional: combine the predicted semantic operation with
the selected capability or broad object class. If richer free-form intent proves
necessary, compare a small constrained decoder in a later experiment; do not add
a second model call.

The project configuration mentions a GTX 1650, but the Phase 2 report did not
capture hardware utilization, quantization, encoder size, or memory. Therefore
no candidate latency is claimed. A compact encoder is a design choice to test,
not evidence of achieved speed. The only measured comparator is the current
local-model p95 of 2545.320 ms.

### Adapter and compatibility

Implement a future `CompactDecisionAdapter.predict(DecisionRequest)` behind the
existing `DecisionModelAdapter` protocol. It should own tokenization, one model
inference, deterministic decoding, and model/version metadata. `FastClassifier`
continues to validate and provide safe fallback behavior.

For current callers, a pure compatibility projection can map one structured
decision to the existing tuple without another model call:

| Structured path | Legacy route projection |
|---|---|
| `direct_response` | LOCAL by initial provider policy |
| `memory_recall` | GEMINI until a dedicated memory node exists |
| `research`, `single_action`, `multi_step_task` | TASK |
| `clarify`, `escalate` | GEMINI until dedicated graph nodes exist |

That projection is transitional and lossy. Dedicated clarification/escalation
nodes are a later integration task. Unknown or invalid model output must produce
the existing safe fallback decision, be logged as invalid, and never reach tool
execution as an unvalidated action.

## 5. Safety and ownership boundaries

- Classifier: semantic interpretation and a proposed capability name only.
- Orchestrator/agent: planning, provider policy, context retrieval, and deciding
  whether execution should proceed.
- Registry/executor: capability allow-list, permissions, argument validation,
  timeouts, execution, and audit logging.
- Deterministic guards: reject unavailable capabilities, malformed types,
  inconsistent flags, missing required information, and unsafe execution state.
- No classifier confidence level may bypass permission or validation checks.
- Benchmarking must continue to use descriptors only; selected capabilities are
  never executed.

## 6. Next experiment

Run one offline, single-call comparison between the legacy classifier and one
unintegrated compact multilingual encoder adapter.

### Data

1. Freeze the current 28 examples as `seed-v1`; use them only as a diagnostic
   regression set, not training data.
2. Adjudicate the five annotation questions above and publish schema/version
   changes rather than editing labels silently.
3. Add at least 200 manually reviewed examples, with at least 30 each for
   English, Telugu, Hindi, and Tamil, plus at least 30 romanized/code-switched
   examples. Include balanced direct, memory, research, single-action,
   multistep, clarification, escalation, noisy-STT, and unsupported-capability
   cases.
4. Group paraphrases, translations, shared contexts, project variants, and
   capability variants by semantic scenario. Split by group, never by row.
5. Keep a sealed test split whose utterances, contexts, and paraphrase families
   are absent from training and model-selection data. Record synthetic origin;
   require human review before evaluation use.

### Fair comparison

- Run both adapters on the same machine, process conditions, ordered examples,
  context, and capability descriptors.
- Measure classifier inference only with `perf_counter_ns`; report first call,
  warm p50/p95, process/model-load time when separately observable, errors, and
  timeouts. Repeat at least three runs with order rotation.
- Score legacy only on its representable route projection. Score the structured
  candidate before fallback on every applicable field.
- Do not use Qwen before the candidate. Do not execute tools. Do not tune on the
  sealed test split.

### Proposed acceptance gates

These are experiment targets, not achieved results:

- 100% structurally valid output on the sealed test set; fallback-trigger rate
  no more than 1% excluding injected fault tests.
- At least 85% execution-path accuracy overall and no language group below 75%
  when that group has at least 20 examples.
- At least 90% capability accuracy on supported-action examples.
- At least 0.85 macro F1 for clarification and escalation, reported separately.
- At least 85% exact-set request-mode accuracy and 80% intent accuracy after
  annotation adjudication.
- Zero capability executions during evaluation and zero bypasses of registry or
  permission checks in integration tests.
- Warm inference p95 no more than 250 ms and at least 5x faster than the legacy
  local-model p95 under matched conditions. Cold/startup latency and memory use
  must also be reported; failure of the absolute target is informative and must
  not be hidden by deterministic shortcuts.

Advance to runtime integration only if quality, schema reliability, and latency
gates are met together. Otherwise revise data/model design offline.

## 7. Remaining uncertainties and risks

- Hardware and deployment-runtime constraints have not been measured formally.
- Dynamic capability matching may confuse semantically adjacent tools and needs
  hard negative examples.
- Romanized Indic spelling has high variation; coverage must come from data and
  pretrained representation, not word lists.
- Context windows must be bounded without omitting the referent needed by short
  follow-ups.
- Confidence calibration may drift across languages and should be measured per
  group.
- Exact intent labels are currently more granular than an encoder classifier may
  need; annotation adjudication must settle the compositional intent scheme.
- The transitional legacy projection cannot preserve clarification/escalation
  semantics until LangGraph gains dedicated handling in a later phase.
- The current TASK agent's restricted candidate-tool prompt is not yet a general
  executor for every capability used by the benchmark.
