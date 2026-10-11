# Existing BERT candidate: wiring audit and offline benchmark

## Verdict and exact target
No BERT classifier is invoked by the inspected production shadow path. The existing trained candidate is runnable offline, and this report evaluates it separately. It is **compact-mbert-frozen-v3**, using `nlptown/bert-base-multilingual-uncased-sentiment` as a frozen feature encoder and trained sklearn logistic-regression heads from `benchmarks/fast_classifier/models/v3/compact_heads.joblib`. The original sentiment-classification head is unused. No deterministic or Qwen classifier was benchmarked.

True accuracy: **NOT MEASURED**. Independently adjudicated gold labels available: **0**. Historical agreement cannot establish semantic correctness or replacement readiness.

## Actual entry points and invocation evidence
Source references are relative to repository root; line numbers refer to this run's hashed source.

- `main.py:78` starts the application; `assistant/core/pipeline.py:24` initializes services and constructs Dispatcher at lines57–60. Nothing here was started for this benchmark.
- `assistant/core/dispatcher.py:52–59` selects `get_orchestrator()` when LLM routing is enabled. Transcript queue is consumed at line65, command handlers run at158, and remaining requests enter `llm.stream(text)` at189.
- `assistant/orchestration/__init__.py:20–27` constructs Orchestrator. `graph.py:323–335` defaults to `QwenClassifier`, not BERT; `stream:407–430` invokes `select_route:337–367`, then `classify_node:274–279` calls `classifier.classify(user_text)`. `router.py:99–139` is the rule-first/legacy Ollama classifier with LOCAL/GEMINI/TASK routing.
- `assistant/orchestration/fast_brain/shadow.py:21–53` defines `observe(adapter, transcript, context, capabilities, existing_route)`, calls `safe_decide` at40 and returns an in-memory ShadowDecision. It does not persist predictions or call the route_callback. No production caller of observe was found by repository call/import searches.
- `CompactMultilingualAdapter` is defined in `compact_adapter.py:105`. Its actual instantiations are offline runners: `run_phase3b_evaluation.py:109`, `run_phase3c_evaluation.py:48`, and `run_phase4_comparison.py:37–48`. There are no production imports or calls connecting it to observe or orchestration. Existing manually run benchmark artifacts are not live shadow logs.
- Separate `fast_brain/encoder_adapter.py:34–44,62–118` defaults to `distilbert-base-multilingual-cased`, requires a supplied local checkpoint directory, and constructs fresh projection/classification layers. It emits raw tensors, not FastBrainDecision. No production caller exists; this foundation is not the trained v3 compact candidate and was not substituted into the benchmark.

This is source-level wiring evidence, not observation of a running service. External/custom injection into Orchestrator is not ruled out; the shipped default execution path does not invoke BERT.

## Exact inference path, preprocessing and heads
The benchmark directly calls the real `CompactMultilingualAdapter.predict(DecisionRequest(...))`, without the FastClassifier safe fallback. Each selected record gets its exact utterance, original v4 conversation/context object and advertised non-executable capability descriptors. Original annotations and human decisions are preserved separately.

`compact_adapter.py:29–49` serializes `User request: {text}\nContext: {sorted JSON truncated to 2000 characters}` and independent capability descriptions. Lines52–102 load tokenizer/AutoModel with local_files_only=True, tokenize padded/truncated to160 tokens, use CPU eval/inference mode, mask-aware mean pooling and L2 normalization. Lines128–145 perform one batched encoder pass per decision. Exact strings presented to the encoder are recorded per row in predictions.jsonl. No artificial language normalization or translation was added.

Lines146–230 apply path and binary request-mode heads, clarification/escalation heads, capability matching, missing-information and reference heads, then compose intent and validate the actual decision. Mode/capability thresholds0.5; clarification/escalation threshold0.5. Capability encoding is independent; selection uses request/capability embeddings and their absolute difference/product. Missing capability can convert an action path to clarify/escalate. Reference resolution can select the first context scalar. These deterministic postprocessing choices are part of the measured model implementation, not benchmark repairs. Confidence is max path-head probability, not measured accuracy.

Actual head classes captured in [model.json](bert_shadow_classifier_20261010_140019/model.json): paths `clarify`, `direct_response`, `escalate`, `memory_recall`, `multi_step_task`, `research`, `single_action`; binary clarification/escalation/reference-resolution/capability-match; missing-information `__none__`, `target`; reference-kind `__none__`, `item`, `prior_conversation`. Four binary mode heads map to conversational/memory/research/action. Intent is composed (`respond`, `recall`, `research`, `use:<selected capability>`, `clarify_request`, `unsupported_request`); capability selection is limited to advertised names, not invented frozen capability families.

## Output contract and validation
`assistant/orchestration/decision.py:19–153` defines the **older** ClassifierDecision interface: intent, execution_path, request_modes, selected_capability, context_references, missing_information, confidence, needs_clarification, needs_escalation, evidence. Request modes are enums conversational/memory/research/action. Intent/path must be nonempty strings; confidence numeric0–1; booleans checked; clarification requires missing information. Reference and missing-information objects are validated by the same parser.

The frozen seven-field contract is a different interface. This model does **not** emit intent_family, capability_family, context_requirements, uncertainty, escalation.reason or frozen clarification fields. No synthetic conversion or seven-field success is reported. Returned raw output mappings are preserved whole, including confidence/evidence. Internal validation occurs before the adapter returns; a second real ClassifierDecision validation was applied. Intermediate raw logits/head probabilities were not instrumented; failures would have retained their exception rather than fabricated a completed output.

## Population and protocol
All359 unique IDs from fast_brain_v1/dataset.jsonl were linked by original_id to the unchanged v4 source. Historical older structured labels—not Qwen proposals—provide comparison values. The 359 population excludes75 source records rejected during previous consolidation; notably its historical labels contain no clarify-path examples. This limits missing-reference evaluation. No independently verified labels were created.

Sequential offline CPU execution; one attempt per record; no retries; no downloads/network/tools/provider calls. Process-only offline environment flags and an audit guard block network, subprocesses and writes outside new evidence directories. An orchestration package namespace bypasses eager provider initialization while importing the unchanged compact adapter and decision implementation. No model logic is mocked in the benchmark; focused tests use their existing fake boundaries.

The initial harness stopped before inference because its write guard blocked a Windows platform check's NUL access; the platform result was then cached before enabling the guard and NUL explicitly exempted as a nonpersistent sink. Its traceback is retained in scratch. This was a harness issue, not a model failure. The successful load used the same existing artifact. The retained load-error.json describes this superseded initial attempt; model.json and final metrics describe the successful run.

## Results
Attempted **359/359**; successful **359**; inference failures **0**; not attempted **0**. Schema-valid returned outputs **359/359 (100%)** under the actual older contract. Exact eight-field historical agreement **164/359 (45.68%)**. Accuracy remains NOT MEASURED.

Field agreement compares all359 outputs, with request_modes compared as sets, JSON object key order ignored, and all other list/structured values preserved. Exact agreement requires all eight historical target fields; confidence/evidence have no historical target and are not scored. Failures would count as nonagreements in the359 denominator.

| Historical field | Agreements /359 | Agreement |
|---|---:|---:|
| intent | 230/359 | 64.07% |
| execution_path | 285/359 | 79.39% |
| request_modes | 309/359 | 86.07% |
| selected_capability | 231/359 | 64.35% |
| context_references | 299/359 | 83.29% |
| missing_information | 332/359 | 92.48% |
| needs_clarification | 332/359 | 92.48% |
| needs_escalation | 336/359 | 93.59% |

### Language breakdown

| Metadata value | Successful / total | Exact historical agreement |
|---|---:|---:|
| code_switch | 77/77 | 49/77 (63.64%) |
| english | 82/82 | 37/82 (45.12%) |
| hindi | 78/78 | 29/78 (37.18%) |
| tamil | 44/44 | 22/44 (50.00%) |
| telugu | 78/78 | 27/78 (34.62%) |

### Script breakdown

| Metadata value | Successful / total | Exact historical agreement |
|---|---:|---:|
| devanagari | 30/30 | 14/30 (46.67%) |
| latin | 209/209 | 75/209 (35.89%) |
| mixed | 77/77 | 49/77 (63.64%) |
| tamil | 13/13 | 11/13 (84.62%) |
| telugu | 30/30 | 15/30 (50.00%) |

### Split breakdown

| Metadata value | Successful / total | Exact historical agreement |
|---|---:|---:|
| development | 57/57 | 11/57 (19.30%) |
| sealed_candidate | 63/63 | 35/63 (55.56%) |
| train | 239/239 | 118/239 (49.37%) |

### Prediction distribution

- intent: use:google_search=54, clarify_request=27, research=77, recall=40, unsupported_request=60, use:open_folder=1, use:find_file=9, respond=56, use:grep_code=2, use:open_app=22, use:close_app=6, use:search_files=2, use:open_file=3.
- execution_path: single_action=99, clarify=27, multi_step_task=43, memory_recall=40, research=34, escalate=60, direct_response=56.
- selected_capability: google_search=119, None=183, open_app=29, open_folder=1, find_file=9, weather=1, grep_code=3, run_project_tests=1, close_app=6, search_files=4, open_file=3.

Median prediction latency 179.58ms; p95 349.08ms across359 measured sequential requests; includes serialization and postprocessing, excludes model loading. p95 uses sorted index floor(0.95*n), zero based. Model load45.94s. Timings are diagnostic, not a controlled production performance comparison.

## Agreement limitations and examples

188/359 IDs overlap the head-training list. This is not a held-out accuracy benchmark; group/near-duplicate overlap may additionally exist.

- Training-ID overlap=True: 115/188 historical exact agreements.
- Training-ID overlap=False: 49/171 historical exact agreements.

Representative structural disagreements, not adjudicated model errors:

| ID | Utterance | Historical path/capability | Predicted path/capability | Differing fields |
|---|---|---|---|---|
| hist-e6dbad9de51f072d | Find my resume file. | single_action/find_file | single_action/google_search | intent, selected_capability |
| hist-bcdc4351b4033198 | naa resume file vetuku | single_action/find_file | clarify/None | intent, execution_path, selected_capability, missing_information, needs_clarification |
| hist-0e52654409ec9398 | mera resume file dhoondo | single_action/find_file | clarify/None | intent, execution_path, selected_capability, context_references, missing_information, needs_clarification |
| hist-89dfef657c756913 | en resume file-ai thedu | single_action/find_file | single_action/google_search | intent, selected_capability |
| hist-cd06c5b65cc6a037 | Locate the Kubernetes overview and extract its main ideas. | multi_step_task/google_search | multi_step_task/open_app | selected_capability |
| hist-b73949baa37fdf13 | Search for the official SQLite architecture page and brief me. | multi_step_task/google_search | multi_step_task/open_app | selected_capability |
| hist-1bceebae602d3c6b | official WebAssembly guide find chesi read chesi summarize cheyyi | multi_step_task/google_search | multi_step_task/open_app | selected_capability |
| hist-fc9c05a57e24554b | Kubernetes overview vetiki main ideas cheppu | multi_step_task/google_search | multi_step_task/open_app | selected_capability |

The lowest historical agreement fields are intent and selected_capability. Strict reference/missing-information agreement includes descriptions and spans, so disagreements need semantic adjudication before being called errors. Request-mode set order is immaterial. Historical multilingual annotations are not independently verified; per-language figures cannot be interpreted as language accuracy. Full source context/capability descriptors, predictions and comparisons are available in the companion artifacts.

## Tests, integrity and reproducibility
`python -B scratch/bert_shadow_audit_20261010_140019/test_runner.py`: **23 passed in10.86s**, covering tests/test_compact_classifier.py and tests/test_fast_brain_encoder_adapter.py. Earlier harness attempts blocked tempfile writes or omitted logger.get; two resulting import-test failures were corrected solely in the temporary harness. Existing tests were unchanged. Tests establish adapter mechanics and boundaries, not accuracy.

`git diff --check`: exit0, existing LF/CRLF notices only. 249 preexisting files in assistant, benchmarks, reports, tests and logs were hashed before/after and remained byte-identical. Full original Git modifications remain; no reset, discard, commit, dataset/history edit, production change, training or model change occurred. Reference projects were not accessed. No Qwen/Gemini inference or network call was made.

New paths only: this report; reports/bert_shadow_classifier_20261010_140019/ with protocol.json, model.json, predictions.jsonl, record-comparison.csv, metrics.json, training-overlap.json, reproducibility.json, integrity.json, manifest.json and retained initial load-error.json; scratch/bert_shadow_audit_20261010_140019/ with run.py, test_runner.py, finalize.py, initial-load-error.json, integrity-before.json, status-before.txt and isolated pytest temporary files.

Reproduce using the run.py harness in a **new timestamped output directory**; it deliberately reads local model files and blocks external boundaries. Runtime package versions, dataset/source/model hashes, local checkpoint cache inventory and command list are in reproducibility.json. Model head hash7f4dd399b7238c7bf0c166454c3ff5aa09d6e3bf1afb82666b526d43cf452f05. The adapter does not pin an explicit Hugging Face revision; the cache inventory records available snapshot hashes but is not an exact file-open trace. That revision-selection limitation should be eliminated in a future controlled benchmark without altering this run's model configuration.

## Recommendation
The existing compact mBERT candidate is locally runnable and suitable for **subsequent independently adjudicated, held-out evaluation**. This run does not support production shadow wiring, automation, training readiness or replacement. First establish gold labels for its actual older output interface (or explicitly design and validate a separate frozen-contract adapter), include clarification cases excluded from this population, and exclude training IDs plus related semantic groups from the held-out set. Do not treat low historical agreement as measured inaccuracy or high schema validity as semantic correctness.

## Artifacts
- [Complete predictions, inputs and historical labels](bert_shadow_classifier_20261010_140019/predictions.jsonl)
- [Full record comparison CSV](bert_shadow_classifier_20261010_140019/record-comparison.csv)
- [Aggregate metrics](bert_shadow_classifier_20261010_140019/metrics.json)
- [Model and head classes](bert_shadow_classifier_20261010_140019/model.json)
- [Reproducibility and hashes](bert_shadow_classifier_20261010_140019/reproducibility.json)
- [Integrity evidence](bert_shadow_classifier_20261010_140019/integrity.json)
