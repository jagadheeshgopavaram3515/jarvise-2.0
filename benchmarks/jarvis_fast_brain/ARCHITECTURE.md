# Fast Brain + Qwen Architecture

## Design principles

The Fast Brain is a replaceable decision adapter, not a BERT feature. It may later be a compact encoder, rules-plus-model hybrid, distilled model, or another implementation. It proposes a path; it cannot call tools. Qwen is the main local intelligence. Registry and Executor remain the only execution boundary.

## Responsibilities

| Component | Owns | Must not own |
|---|---|---|
| STT | speech to text | intent, tool choice, escalation |
| Fast Brain | semantic mode, capability family, uncertainty, clarification/escalation signal | execution, arguments with side effects, tool calls |
| Qwen | reasoning, plan, arguments, observations, recovery, local final response | bypassing Registry or permissions |
| Orchestrator | state, workflow, context assembly, iteration, provider choice | hidden tool execution |
| ToolRegistry/Executor | allowlist, schema, validation, timeout, audit, execution | semantic planning |
| Gemini | explicit cloud escalation/fallback | default routing for every failure |
| TTS | text to speech | response generation |

## Target flow

```text
STT transcript
  -> Fast Brain decision
  -> clarification? ask user
  -> deterministic/simple? bounded direct path where already supported
  -> Qwen with compact decision envelope
       -> plan/action JSON
       -> Orchestrator validates through Registry/Executor
       -> bounded observation
       -> Qwen continue, recover, finish, or request escalation
  -> Gemini only with recorded escalation reason/evidence
  -> final text -> TTS
```

This is a target architecture only. The current production graph is unchanged by this document.

## Capability vocabulary from the actual Registry

Current registered families include location and weather; app open/close/list; file find/search/open/folder/recent; bounded project read/list/grep/test; and browser open URL, Google search, YouTube search, page read, and page title. A future Fast Brain should classify into these names or stable families, but must not invent an unavailable capability.

## Memory-conscious design

Keep one Qwen resident when possible. Pass a compact decision envelope, only the relevant capability description/schema, bounded working history, selected memory snippets, and task-specific observations. Do not pass the whole Registry, full logs, or all memory by default. STT, browser, GUI, embeddings, and Qwen residency must be measured together before allowing concurrency. Prefer request serialization for Qwen on an 8 GB system.

## Migration order (future, not now)

1. Add offline contract tests around the existing decision adapter.
2. Build an observation-only Fast Brain adapter behind the existing `FastClassifier` contract.
3. Measure shadow decisions against legacy routes without changing routing.
4. Define Qwen envelope and replay tests.
5. Only after benchmark gates pass, propose production integration separately.
