# JARVIS FastClassifier Offline Benchmark

This benchmark is evaluation infrastructure only. It is not imported by the
JARVIS runtime, never executes a selected capability, and does not change the
production router.

## Current interfaces

`QwenClassifier.classify(text)` returns only `(route, latency_ms)`, where route
is `LOCAL`, `GEMINI`, or `TASK`. The legacy evaluation therefore measures only
execution-path accuracy, visible exceptions/timeouts, and latency. It marks
intent, request modes, capability selection, references, missing information,
clarification, and escalation as unavailable. In particular, the placeholder
semantic fields produced by `LegacyClassifierAdapter` are not scored.

A future `DecisionModelAdapter.predict(DecisionRequest)` can be evaluated for
all validated `ClassifierDecision` fields. The structured evaluator validates
raw output before `FastClassifier` fallback so invalid output remains visible.

## Dataset schema 1.0

The reviewed seed set is `v1/seed.jsonl`. Every record contains:

- `schema_version`, stable `id`, and `split`
- `utterance`, optional `context`, and advertised `capabilities`
- `expected` fields matching the Phase 1 decision contract
- `language`, `categories`, annotation status/source, and review notes

Each seed record has one reviewed intent, execution path, and capability label.
Request modes are multi-label and use exact-set scoring: all expected modes must
be present and no extra mode may be returned. References and missing-information
items are compared as ordered, exact contract values. Alternative labels are not
silently accepted; a future dataset version should represent them explicitly if
manual review finds genuinely equivalent labels.

Related context variants stay in the same split. The companion
`v1/invalid_outputs.jsonl` corpus covers missing fields, invalid types, unknown
request modes, and malformed non-object output. These are model outputs, not
utterances, so they are kept separate from accuracy examples.

The structured contract intentionally permits new execution-path strings, so an
unknown structured path is schema-valid and will fail label comparison rather
than validation. Unknown values from the legacy three-route interface are
invalid outputs and are covered by the harness tests.

## Metrics

Legacy reports contain execution-path accuracy only. Structured reports contain
field-level exact-match accuracy, schema validity, and the rate of outcomes that
would trigger `FastClassifier` fallback. Both report errors, watchdog timeouts,
coverage, failure examples, and route accuracy by language/category.

Expected `clarify` and `escalate` paths are not representable by the legacy
three-route interface, so they are excluded from its accuracy denominator and
reported as not applicable. The Qwen CLI labels each timed call as either
`deterministic_fast_path` or `local_model` using the existing pure routing
function; this adds no model invocation and enables separate latency summaries.

Latency uses `time.perf_counter_ns()` and nearest-rank p50/p95. The first
invocation is reported separately as a cold-start approximation; this does not
guarantee that process startup or model loading was observed. Remaining
successful invocations are reported as warm, but model cache state is not
asserted.

## Commands

Validate and run unit tests without a model or network:

```powershell
python -m pytest tests/test_classifier_benchmark.py -q
```

Run the real current classifier against the complete seed set and write JSON:

```powershell
python -m benchmarks.fast_classifier.evaluate --adapter legacy-qwen --output benchmark-results/legacy-qwen-v1.json
```

Select a split or stable subset:

```powershell
python -m benchmarks.fast_classifier.evaluate --split test --ids en-knowledge-001,ta-romanized-action-001 --output benchmark-results/subset.json
```

The legacy command may call the configured local Ollama endpoint for examples
that do not hit its deterministic fast path. It never downloads a model. An
unavailable Ollama endpoint is hidden by the production classifier as a
`GEMINI` fallback, which the report calls out as a legacy-interface limitation.

Future structured adapters are evaluated programmatically with
`evaluate_structured_adapter(adapter, dataset)`. Adding a CLI adapter should
only require constructing that adapter; the dataset, metrics, and report format
remain unchanged.
