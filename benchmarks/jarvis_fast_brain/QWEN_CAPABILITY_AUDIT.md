# JARVIS Base Qwen Capability Audit

**Audit type:** observation and source inspection only  
**Date:** 2026-09-27  
**Production changes:** none made by this audit

## 1. Executive answer

Raw Qwen is a text-generation model. It does not have OS access, browser access, or Python tool execution by itself. In the raw tests it described steps, returned JSON, and once refused a search request; it did not actually open Chrome or search the web.

The current JARVIS system does provide an agent/tool pathway. In the LangGraph TASK path, Qwen creates a plan and emits JSON containing an action, tool name, and arguments. `LocalAgent` validates the tool name against `ToolRegistry`, `ToolExecutor` executes the registered Python handler, the result is formatted into an observation, and the graph can call Qwen again. Therefore the precise statement is:

> Qwen can propose a tool action through JARVIS; ToolRegistry/ToolExecutor performs the actual operation.

This is not the same as Qwen itself opening Chrome.

## 2. Ollama and model status

Observed from `D:\JARVIS`:

| Item | Result |
|---|---|
| Endpoint | `http://localhost:11434` |
| Ollama reachable | YES; `/api/tags` responded |
| Installed model | `qwen2.5-coder:3b` |
| Model size | 1,929,912,626 bytes in Ollama metadata |
| Parameter/quantization | 3.1B, Q4_K_M, GGUF |
| Model context metadata | 32,768 tokens |
| Ollama capabilities | completion, tools, insert |
| Loaded before raw tests | NO; `/api/ps` returned an empty model list |
| JARVIS timeout | `OLLAMA_TIMEOUT_S=45.0` seconds |
| Classifier timeout | `CLASSIFIER_TIMEOUT_S=5.0` seconds |
| Keep-alive | `10m` |
| Configured model | `qwen2.5-coder:3b` |
| Classifier model | `qwen2.5-coder:3b` |

The configured production conversation client sends temperature `0.3` and does not set an explicit `num_predict`. The classifier sends temperature `0.0`, `num_predict=5`, and stop tokens. Local agent JSON calls send temperature `0.1` without an explicit output cap; final synthesis sends temperature `0.2`, `num_predict=250`.

## 3. Raw Qwen tests

These calls went directly to Ollama `/api/generate`, with `stream:false`, `temperature=0.0`, `keep_alive=10m`, and a benchmark-only `num_predict` cap. No JARVIS tools or ToolRegistry were supplied.

### A — Explain what would be needed to open Chrome

Latency: **5,773.5 ms**; `prompt_eval_count=46`; `eval_count=120`.

Qwen gave a numbered explanation involving Windows, Chrome installation, a browser, internet access, and hardware. It did not claim that Chrome was opened. The answer included some irrelevant requirements, so factual usefulness was mixed, but execution was not falsely claimed.

### B — Open Chrome

Latency: **4,183.8 ms**; `prompt_eval_count=37`; `eval_count=80`.

Qwen returned instructions such as using the Windows Start Menu and clicking Chrome. It did not claim to have opened Chrome and did not produce an executable tool call because no tool contract was supplied.

### C — Search Google for latest AI news

Latency: **2,556.0 ms**; `prompt_eval_count=37`; `eval_count=14`.

Qwen returned: “I'm sorry, but I can't assist with that request.” It did not search, browse, or claim success.

### D — What tools would be needed for the previous request?

Latency: **4,183.3 ms**; `prompt_eval_count=41`; `eval_count=80`.

Qwen described internet access, a browser/application, software, and credentials. It did not identify a JARVIS capability name because none was provided in the raw prompt.

### E — Return JSON for opening Chrome

Latency: **2,635.6 ms**; `prompt_eval_count=41`; `eval_count=21`.

Qwen returned valid-looking JSON:

```json
{"action":"open","application":"chrome"}
```

This is text, not execution. It does not match the current JARVIS agent action schema, which expects `action: "tool"`, `tool_name`, and `tool_args`.

### Raw-Qwen conclusion

Raw Qwen demonstrated language understanding and JSON formatting, but no independent execution capability. It did not hallucinate that Chrome or Google had been used in these tests.

## 4. Actual current JARVIS Qwen architecture

### Qwen call sites

1. `assistant/orchestration/router.py::QwenClassifier.classify`
   - deterministic regex fast path first;
   - otherwise one non-streaming Ollama request for `TASK`, `LOCAL`, or `GEMINI`;
   - invalid output, Ollama error, and timeout fall back to `GEMINI`.

2. `assistant/llm/ollama.py::OllamaClient.stream`
   - conversational local response;
   - builds a prompt with JARVIS system instructions, time/date, selected memory, recent history, and optional real-time briefing;
   - streams response chunks for TTS.

3. `assistant/orchestration/agent.py::LocalAgent.create_plan`
   - non-streaming JSON request;
   - receives goal and up to three recent history turns;
   - asks for a one-to-three-step plan.

4. `LocalAgent.select_action`
   - non-streaming JSON request;
   - receives goal, plan, completed steps, last observation, errors, recent history, and a manually chosen tool description list;
   - accepts `tool` or `finish` JSON.

5. `LocalAgent.synthesize_final_response`
   - non-streaming Qwen request if there is no sufficiently good candidate response;
   - receives goal and bounded formatted observations;
   - uses `num_predict=250`.

6. `assistant/tools/router.py::_Selector`
   - separate Gemini path for desktop/browser tool selection and natural phrasing;
   - deterministic matching is attempted first, then Gemini selection for imperative-looking text.

### Agent graph

```text
TASK request
  -> understand
  -> Qwen create_plan
  -> Qwen select_action
  -> LocalAgent.execute_tool
  -> ToolExecutor / ToolRegistry / Python handler
  -> format_observation
  -> Qwen select_action again
  -> continue until finish or max_iterations=8
  -> Qwen final synthesis if needed
```

`Orchestrator.run_agent` supplies conversation history from memory when no history is passed. `graph.py` owns the loop, state, retry count, observations, and final response. The graph can perform multi-step workflows when Qwen returns successive valid actions.

## 5. What Qwen actually receives for agent decisions

Sanitized shape of the actual `select_action` prompt:

```text
Recent conversation context: [up to 3 supplied turns]
Goal: <user goal>
Plan: <JSON plan>
Completed Steps: <JSON strings>
Last Observation: <formatted bounded observation or None>
Errors Encountered: <JSON list>

Available Tools:
- location(...): description from Registry
- weather(...): description from Registry
- read_file(...): description from Registry
- list_directory(...): description from Registry
- grep_code(...): description from Registry
- run_project_tests(...): description from Registry

Decide the next action.
Return exactly one of:
{"thought":"...","action":"tool","tool_name":"...","tool_args":{...}}
{"thought":"...","action":"finish","final_response":"..."}
```

Important: the prompt does **not** describe `open_app`, `google_search`, `youtube_search`, `open_url`, or the other browser/desktop tools, even though those tools are registered. The code still checks any returned name with `registry.get_tool`, so a model-generated registered name can be accepted, but it is not part of the advertised Qwen capability set in this agent prompt.

The agent system prompt instructs Qwen to use project inspection tools, chain location→weather, return JSON, ground claims in observations, and never invent tools or facts. It contains repository layout guidance. It does not grant OS privileges.

## 6. Actual ToolRegistry capabilities

The default Registry currently contains 19 capabilities:

`location`, `weather`, `open_app`, `close_app`, `list_running_apps`, `find_file`, `search_files`, `open_file`, `open_folder`, `recent_files`, `read_file`, `list_directory`, `grep_code`, `run_project_tests`, `open_url`, `google_search`, `youtube_search`, `read_page`, and `get_page_title`.

Registry membership is only a catalogue/allowlist. `ToolExecutor.execute` performs the actual handler call in a worker pool, applies the configured 25-second tool timeout, normalizes the result, and writes an audit record. The graph calls this executor; Qwen does not.

## 7. Controlled JARVIS agent tests

No real desktop or browser action was allowed. The tests used a fake Qwen JSON response and a fixture executor.

### “Open Chrome”

When the controlled Qwen response was `{"action":"tool","tool_name":"open_app","tool_args":{"app":"chrome"}}`, `LocalAgent.select_action` accepted the name because `open_app` exists in the Registry. However, `open_app` was absent from the advertised six-tool prompt. The fixture executor was not allowed to launch anything.

Result: **the current code can carry a Qwen-selected `open_app` action to the executor, but the standard LocalAgent prompt does not teach/advertise that capability.**

### “Search Google for latest AI news”

The same controlled path accepted a returned `google_search` name at validation time, but `google_search` was not in the Qwen agent prompt. The separate `tools/router.py` Gemini selector does advertise the complete Registry and can select it after deterministic matching fails.

Result: **browser search is available to JARVIS, but not consistently exposed through the LocalAgent Qwen prompt.**

### Multi-step project inspection

A controlled sequence returned a plan, then `grep_code`, then `finish`. The actual graph executed the fixture `grep_code`, formatted its observation as:

```text
Found 1 matches:
assistant/orchestration/router.py:42 -> QwenClassifier
```

Qwen then returned a final response. No real file operation was performed.

Result: **the plan → tool → observation → next Qwen action loop is operational.**

### Failure recovery

The graph passes the formatted failure into the next selection prompt through `Last Observation` and retains an `Errors Encountered` list. It permits another Qwen action until `max_iterations=8`. There is no separate retry policy that guarantees a retry, alternate capability, clarification, or Gemini escalation; the next behavior is Qwen-dependent.

Result: **replanning is possible; recovery quality is not guaranteed by the current orchestration.**

### Clarification

There is no explicit missing-argument clarification gate in `LocalAgent.select_action`. A Qwen response selecting `open_file` with empty arguments is accepted as a known Registry tool. Whether it asks clarification depends on Qwen output and downstream handler behavior. The legacy route also does not have a dedicated clarification state.

Result: **clarification is model/prompt behavior, not an enforced capability today.**

## 8. Routing observations for requested scenarios

| Scenario | Current source-grounded behavior |
|---|---|
| Open Chrome | Legacy deterministic tool router can match app opening and invoke `open_app`; TASK agent Qwen prompt does not advertise it. No real action was executed in this audit. |
| Browser search | `google_search` is registered; Gemini selector can select it; LocalAgent’s six-tool prompt omits it. |
| “Do that again but search for NVIDIA” | Agent receives only caller-provided recent history (or memory history through Orchestrator); no dedicated pronoun resolver exists. It may resolve this through Qwen, but deterministic success was not established without a real, controlled multi-turn model run. |
| “Open it” with no target | No hard clarification gate. Qwen may finish, ask, or emit a tool action; empty required arguments are not rejected by `select_action` itself. |
| “Open Chrome” | The legacy `tools/router.py` has deterministic app matching. This is a separate direct tool path and does not prove raw Qwen execution. |
| “What is the latest Qwen model?” | Qwen itself has no inherent web access. `OllamaClient` may request a real-time briefing through `RealTimeManager` when its heuristic says current information is needed. The LocalAgent tool prompt does not expose browser research tools. |

## 9. Required distinction

| Layer | Meaning | Current result |
|---|---|---|
| Raw Qwen | Text model called directly through Ollama | Can explain, format JSON, and generate proposed actions; cannot execute OS/browser actions |
| JARVIS Qwen Agent | Qwen plus prompts, LangGraph, Registry validation, Executor, observations | Can propose registered tool actions and participate in multi-step loops |
| Tool execution | Python handlers called by `ToolExecutor` | Performs the actual app/browser/file/weather/location operation, subject to its own safety and timeout behavior |

## 10. Capability table

| Capability | Raw Qwen | JARVIS Qwen Agent | ToolRegistry | Verified? |
|---|---:|---:|---:|---:|
| Understand request | YES | YES | NO | YES, raw and source/controlled tests |
| Open Chrome | NO | PARTIAL | YES (`open_app`) | PARTIAL; no real execution |
| Google search | NO | PARTIAL | YES (`google_search`) | PARTIAL; no real network |
| YouTube search | NO | PARTIAL | YES (`youtube_search`) | PARTIAL; no real network |
| File search | NO | YES for advertised project tools; broader file tools are separately registered | YES (`find_file`, `search_files`) | PARTIAL |
| Read project file | NO | YES | YES (`read_file`) | YES via controlled fixture path/source/tests |
| Weather | NO | YES | YES (`weather`) | YES via existing tests/source; no live call here |
| Location | NO | YES | YES (`location`) | YES via existing tests/source; no live call here |
| Multi-step task | NO | YES/PARTIAL | N/A | YES loop; quality model-dependent |
| Context resolution | NO | PARTIAL | NO | NOT fully tested with live Qwen |
| Clarification | NO | PARTIAL | NO | Source says no enforced gate |
| Failure recovery | NO | PARTIAL | Executor returns failures | YES loop path; outcome model-dependent |

## 11. Current limitations

1. The legacy classifier collapses decisions to `TASK`, `LOCAL`, or `GEMINI`; classifier failures become Gemini fallback and can hide the original Qwen failure in production telemetry.
2. The richer `ClassifierDecision` contract exists but is not wired into the production graph.
3. The LocalAgent advertises only six capabilities despite a 19-tool Registry.
4. Tool-name validation checks Registry membership but does not itself enforce required JSON arguments or a clarification policy.
5. Recovery is an opportunity for another Qwen action, not a guaranteed strategy.
6. Context resolution is dependent on bounded history and Qwen interpretation; no explicit reference-resolution component exists.
7. Browser/search behavior can be handled by a separate Gemini selector, so “JARVIS can search” does not mean “Qwen can search.”
8. Current-information prompting is partly handled by `RealTimeManager` in the conversational Ollama client, not by a uniform Qwen agent research workflow.

## 12. What Qwen can and cannot do today

### Qwen can

- classify legacy routes through a constrained prompt;
- produce conversational text and stream it;
- create JSON plans;
- select a known Registry capability when prompted or when it emits a registered name;
- create tool arguments as JSON;
- interpret bounded observations;
- request another action in the graph loop;
- synthesize a grounded final response from observations when prompted.

### Qwen cannot by itself

- launch Chrome or any Windows process;
- open a URL or perform a Google/YouTube search;
- read or modify files;
- obtain location or weather;
- enforce ToolRegistry permissions;
- validate or execute Python handlers;
- guarantee clarification, recovery, or escalation;
- access the internet merely because it is Qwen.

## 13. Fast Brain and LoRA implications

The Fast Brain would add a compact, model-independent semantic decision before Qwen: intent, execution path, capability family, context references, uncertainty, clarification, and escalation signals. It would not execute tools.

LoRA could potentially teach Qwen JARVIS-specific plan format, actual capability names, argument formation, context resolution, observation-driven recovery, clarification, escalation, multilingual/code-switched handling, and concise final responses. It would not replace Registry validation or make raw Qwen an OS executor.

## 14. Audit safety and repository state

No Chrome launch, browser navigation, arbitrary process, live search, file write, ToolRegistry modification, routing change, STT/TTS change, Gemini behavior change, Fast Brain integration, or LoRA training was performed.

The repository already contained unrelated modifications before this audit, including changes under `assistant/config.py`, `assistant/orchestration/router.py`, and `assistant/stt/recognizer.py`, plus untracked benchmark/support artifacts. Those were preserved. This audit added only this report.

## 15. Exact next phase

Use the Phase B replay harness to create reviewed deterministic fixtures for the 19 actual Registry capabilities, then run Base Qwen trajectories with Ollama availability, prompt version, latency, raw output, parse state, and failure stage recorded. After that, implement a non-production Fast Brain adapter behind the existing `ClassifierDecision` contract and replay the same sealed cases. Do not integrate it into routing or train LoRA until those measurements exist.
