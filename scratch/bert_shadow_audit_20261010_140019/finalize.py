import json,pathlib,hashlib,csv,collections,importlib.metadata,subprocess,sys
R=pathlib.Path('D:/JARVIS');W=R/'scratch/bert_shadow_audit_20261010_140019';O=R/'reports/bert_shadow_classifier_20261010_140019';report=R/'reports/bert_shadow_classifier_audit_20261010_140019.md'
assert not report.exists()
rows=[json.loads(x) for x in (O/'predictions.jsonl').read_text(encoding='utf-8').splitlines()];m=json.loads((O/'metrics.json').read_text());assert len(rows)==359 and len({r['record_id'] for r in rows})==359
assert m['successful']==sum(r['status']=='success' for r in rows)==359
assert m['historical_exact_agreements']==sum(r['historical_exact_agreement'] for r in rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
sourcepaths=['assistant/orchestration/compact_adapter.py','assistant/orchestration/decision.py','assistant/orchestration/fast_brain/shadow.py','assistant/orchestration/fast_brain/encoder_adapter.py','assistant/orchestration/graph.py','assistant/orchestration/router.py','assistant/core/dispatcher.py','benchmarks/fast_classifier/fast_brain_v1/dataset.jsonl','benchmarks/fast_classifier/v4/dataset.jsonl','benchmarks/fast_classifier/models/v3/compact_heads.joblib','benchmarks/fast_classifier/models/v3/compact_heads.json','benchmarks/fast_classifier/fast_brain_v1/review_decisions.jsonl']
checkpoint=[]
cache=pathlib.Path.home()/'.cache/huggingface/hub/models--nlptown--bert-base-multilingual-uncased-sentiment'
if cache.exists():
    for p in cache.rglob('*'):
        if p.is_file() and 'snapshots' in p.parts:checkpoint.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
dump(O/'reproducibility.json',dict(python=sys.version,versions={x:importlib.metadata.version(x) for x in ['torch','transformers','scikit-learn','joblib','numpy']},source_hashes={p:sha(R/p) for p in sourcepaths},cached_checkpoint_files=checkpoint,checkpoint_note='Read-only cache inventory; no downloads. All cached snapshots listed, not proof each was opened.',commands=['python -B scratch/bert_shadow_audit_20261010_140019/run.py','python -B scratch/bert_shadow_audit_20261010_140019/test_runner.py','git diff --check'],tests='23 passed in 10.86s'))
with (O/'record-comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['record_id','source_id','utterance','language','script','training_overlap','status','historical_label','prediction','historical_exact_agreement','disagreeing_fields','latency_ms'])
    for r in rows:writer.writerow([r['record_id'],r['source_id'],r['utterance'],r['language'],r['script'],r['training_overlap'],r['status'],json.dumps(r['historical_label'],ensure_ascii=False),json.dumps(r['prediction'],ensure_ascii=False),r['historical_exact_agreement'],','.join(k for k,v in r['historical_field_agreement'].items() if not v),r['latency_ms']])
overlap={str(v):dict(total=sum(r['training_overlap']==v for r in rows),agreements=sum(r['training_overlap']==v and r['historical_exact_agreement'] for r in rows)) for v in [True,False]}
dump(O/'training-overlap.json',overlap)
text='''# Existing BERT candidate: wiring audit and offline benchmark

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

`compact_adapter.py:29–49` serializes `User request: {text}\\nContext: {sorted JSON truncated to 2000 characters}` and independent capability descriptions. Lines52–102 load tokenizer/AutoModel with local_files_only=True, tokenize padded/truncated to160 tokens, use CPU eval/inference mode, mask-aware mean pooling and L2 normalization. Lines128–145 perform one batched encoder pass per decision. Exact strings presented to the encoder are recorded per row in predictions.jsonl. No artificial language normalization or translation was added.

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
'''
text+=f'Attempted **{m["attempted"]}/359**; successful **{m["successful"]}**; inference failures **{m["failed"]}**; not attempted **{m["not_attempted"]}**. Schema-valid returned outputs **359/359 (100%)** under the actual older contract. Exact eight-field historical agreement **{m["historical_exact_agreements"]}/359 ({100*m["historical_exact_agreements"]/359:.2f}%)**. Accuracy remains NOT MEASURED.\n\n'
text+='Field agreement compares all359 outputs, with request_modes compared as sets, JSON object key order ignored, and all other list/structured values preserved. Exact agreement requires all eight historical target fields; confidence/evidence have no historical target and are not scored. Failures would count as nonagreements in the359 denominator.\n\n| Historical field | Agreements /359 | Agreement |\n|---|---:|---:|\n'
for k,v in m['historical_field_agreements'].items():text+=f'| {k} | {v}/359 | {100*v/359:.2f}% |\n'
for dim in ['language','script','split']:
    text+=f'\n### {dim.title()} breakdown\n\n| Metadata value | Successful / total | Exact historical agreement |\n|---|---:|---:|\n'
    for k,v in m['groups'][dim].items():text+=f'| {k} | {v["successful"]}/{v["total"]} | {v["historical_exact_agreements"]}/{v["total"]} ({100*v["historical_exact_agreements"]/v["total"]:.2f}%) |\n'
text+='\n### Prediction distribution\n\n'
for dim,counts in m['distribution'].items():text+=f'- {dim}: '+', '.join(f'{k}={v}' for k,v in counts.items())+'.\n'
text+=f'\nMedian prediction latency {m["median_ms"]:.2f}ms; p95 {m["p95_ms"]:.2f}ms across359 measured sequential requests; includes serialization and postprocessing, excludes model loading. p95 uses sorted index floor(0.95*n), zero based. Model load45.94s. Timings are diagnostic, not a controlled production performance comparison.\n\n'
text+='## Agreement limitations and examples\n\n188/359 IDs overlap the head-training list. This is not a held-out accuracy benchmark; group/near-duplicate overlap may additionally exist.\n\n'
for k,v in overlap.items():text+=f'- Training-ID overlap={k}: {v["agreements"]}/{v["total"]} historical exact agreements.\n'
text+='\nRepresentative structural disagreements, not adjudicated model errors:\n\n| ID | Utterance | Historical path/capability | Predicted path/capability | Differing fields |\n|---|---|---|---|---|\n'
for r in [r for r in rows if not r['historical_exact_agreement']][:8]:
    e=r['historical_label'];p=r['prediction'];text+=f'| {r["record_id"]} | {r["utterance"].replace("|","/")} | {e["execution_path"]}/{e["selected_capability"]} | {p["execution_path"]}/{p["selected_capability"]} | '+', '.join(k for k,v in r['historical_field_agreement'].items() if not v)+' |\n'
text+='''
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
'''
report.write_text(text,encoding='utf-8')
before=json.loads((W/'integrity-before.json').read_text());changed=[p for p,h in before.items() if not (R/p).exists() or sha(R/p)!=h];assert not changed,changed
dump(O/'manifest.json',dict(report=str(report),records=359,attempted=359,successful=359,independent_gold=0,accuracy='NOT MEASURED',artifacts={p.name:dict(path=str(p),sha256=sha(p)) for p in O.iterdir() if p.is_file()},preserved_preexisting_files=len(before)))
status=subprocess.check_output(['git','status','--short'],cwd=R,text=True);(W/'status-after.txt').write_text(status,encoding='utf-8')
initial=(W/'status-before.txt').read_text(encoding='utf-8').splitlines();missing=set(initial)-set(status.splitlines());assert not missing,missing
print('Report and manifest complete; 359 records verified; '+str(len(before))+' preexisting files unchanged; all original status entries retained.')
