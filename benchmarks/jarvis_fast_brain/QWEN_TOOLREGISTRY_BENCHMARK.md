# Phase C — Qwen ↔ ToolRegistry Benchmark

## Actual execution path

```text
Orchestrator.select_route
  -> QwenClassifier / deterministic router
  -> TASK
  -> LocalAgent.create_plan (Qwen JSON)
  -> LocalAgent.select_action (Qwen JSON)
  -> ToolRegistry membership check
  -> ToolExecutor.execute
  -> registered Python handler
  -> bounded formatted observation
  -> LocalAgent.select_action again
  -> final synthesis
```

The separate direct tool path in `assistant/tools/router.py` performs deterministic matching first, then can ask Gemini to select and phrase a tool call. It also uses the same Executor. This means the current system has two tool-selection routes: LocalAgent/Qwen for TASK graph work, and deterministic/Gemini selection in the direct tool layer.

## Registry inventory

| Name | Execution type | Advertised to LocalAgent Qwen | Direct deterministic path | Gemini selector path | Fixture/safe test status |
|---|---|---:|---:|---:|---|
| `location` | location provider | YES | NO | Registry-visible | Existing tests; no live call |
| `weather` | weather provider | YES | NO | Registry-visible | Existing tests; no live call |
| `open_app` | Windows process launch | NO | YES | Registry-visible | Existing patched tests; no live launch |
| `close_app` | Windows process close | NO | YES | Registry-visible | Existing safety tests |
| `list_running_apps` | process inspection | NO | NO | Registry-visible | Existing tests |
| `find_file` | bounded filesystem search | NO | YES | Registry-visible | Existing tests |
| `search_files` | bounded filesystem search | NO | NO | Registry-visible | Existing tests |
| `open_file` | default application launch | NO | NO | Registry-visible | Existing tests/source; no live open |
| `open_folder` | Explorer launch | NO | YES | Registry-visible | Existing tests; no live launch |
| `recent_files` | filesystem inspection | NO | NO | Registry-visible | Source/Registry |
| `read_file` | bounded project read | YES | NO | Registry-visible | Existing tests/fixture path |
| `list_directory` | bounded project listing | YES | NO | Registry-visible | Existing tests/fixture path |
| `grep_code` | bounded project search | YES | NO | Registry-visible | Existing tests and controlled fixture |
| `run_project_tests` | approved subprocess test runner | YES | NO | Registry-visible | Existing tests; controlled carefully |
| `open_url` | default browser launch | NO | NO | Registry-visible | Source only; no live browser |
| `google_search` | browser URL/search | NO | YES for matching phrases | Registry-visible | Source/tests; no live search |
| `youtube_search` | browser URL/search | NO | YES for matching phrases | Registry-visible | Source/tests; no live search |
| `read_page` | Playwright page fetch | NO | NO | Registry-visible | Existing example test only; not used here |
| `get_page_title` | Playwright page fetch | NO | NO | Registry-visible | Existing example test only; not used here |

Exact count: **19 Registry tools**, **6 advertised to LocalAgent Qwen** (`location`, `weather`, `read_file`, `list_directory`, `grep_code`, `run_project_tests`), and **19 available to the Gemini selector through `registry.describe()`**. The source does not state why the LocalAgent list is narrower; it appears to be an intentional hard-coded Phase 1 agent subset, not a Registry permission distinction.

## Controlled observations

The controlled fixture sequence returned a plan, selected `grep_code`, received a synthetic observation, and then returned a final response. The graph executed no real tool and still demonstrated the actual loop shape.

For a controlled Qwen JSON response selecting `open_app` or `google_search`, `LocalAgent.select_action` accepts the name because `registry.get_tool()` finds it, even though the name is absent from the prompt’s advertised list. Thus Registry membership validation is broader than Qwen capability exposure.

The smoke replay showed Qwen selecting `open_file` with invented `example.txt` for “Open it” and the evaluator classified this as a clarification failure. This is evidence against treating clarification as an enforced ToolRegistry behavior.

## Failure/recovery behavior

ToolExecutor returns normalized failure results, and the graph feeds formatted failures into the next Qwen action prompt. Qwen may retry, repair arguments, choose another tool, finish, or fail; no deterministic policy guarantees one of those outcomes. The iteration ceiling is eight by default.

## Safety boundary

No real Chrome, browser page, file mutation, or arbitrary process was executed for this benchmark. Existing unit tests patch process/browser calls where needed. Future Phase C fixture cases should use the Phase B lookup-only fixture executor and record the same observations for every system.
