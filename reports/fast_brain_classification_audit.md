# Fast Classification: complete offline input/output audit

## 1. Executive summary

**Scope changed at the user’s request:** stop Qwen and focus on Fast Classification. This is a full 359-record offline audit, not a completed Qwen benchmark. No further model inference was performed after stopping the Qwen harness.

**The prepared structured labels are not semantically correct for all inputs.** The actual normalization and queue-building code reproduces all 359 labels, but 46/359 (12.81%) lose a clearly required capability domain: 39 file/folder/recent-file requests and seven project-code searches. All 359 labels pass schema validation. Valid shape therefore does not establish correct interpretation.

Another 319/359 (88.86%) empty-source-context records still get context_requirements=[active_task] because the queue builder tests a boolean for non-nullness. This establishes a construction defect/misleading metadata, not a measured semantic accuracy loss. The requested 353 does not match the current snapshot: 359 total, 328 with no decision, 31 historically decided; pending plus 13 skips is 341.

No complete independent seven-field gold set exists. The 46 count is a component-level finding, **not a full-classifier error rate**. 193 records have a provisionally supported primary domain and 120 have explicit semantic/context concerns; neither category proves full-label correctness. Source annotations are historical expected labels, not current live predictions. This evidence does not support replacing or automatically trusting classification.

## 2. Actual taxonomy and classifier interfaces

### Frozen seven-field taxonomy

This inventory is created before record inference. Enum spelling is authoritative; explanatory meanings below are source-derived operational descriptions, not additional schema constraints.

### IntentFamily

`conversation`, `information`, `deterministic_action`, `local_task`, `research`, `memory`, `multi_step_task`, `clarification`, `unsupported`

### CapabilityFamily

`none`, `application`, `browser`, `file`, `project`, `weather`, `location`, `reminder`, `memory`, `research`, `system`, `unknown`

### ExecutionPath

`fast_path`, `respond`, `qwen_agent`, `memory`, `research`, `clarify`, `escalate`

### ContextRequirement

`conversation`, `memory`, `project`, `current_information`, `observations`, `active_task`, `none`

### MissingInformation

`target`, `file`, `application`, `url`, `query`, `location`, `time`, `scope`, `permission`, `reference`

### EscalationReason

`current_information_required`, `unsupported_capability`, `outside_qwen_envelope`, `repeated_local_failure`, `context_overload`, `explicit_cloud_request`

### Uncertainty

`high`, `medium`, `low`, `ambiguous`, `outside_envelope`

### Required decision fields and constraints

All seven top-level fields are required by FastBrainDecision.from_mapping; the Python dataclass's convenience defaults do not make serialized fields optional. clarification requires required and missing; escalation requires candidate and reason (reason may be null). Booleans are intended; current validator does not strictly enforce their JSON types. Extra keys are not rejected. All decisions are advisory, not executable routes.

Source: [contract.py](../assistant/orchestration/fast_brain/contract.py) lines 13–87 (enums), 90–160 (models/parser); [validator.py](../assistant/orchestration/fast_brain/validator.py) lines 16–38. Nonempty context list; none cannot coexist with another requirement; required clarification iff missing list nonempty; escalation candidate iff reason non-null; clarify path/clarification intent require clarification; fast_path cannot have unknown capability. No general intent/path/capability compatibility table is enforced.

### Meanings and computer actions

Intent: conversation=dialogue; information=informational response; deterministic_action=direct action category; local_task=local capability task; research=current/web information task; memory=memory retrieval/use; multi_step_task=multi-operation goal; clarification=incomplete request; unsupported=outside supported capabilities.

Execution: fast_path=deterministic handling; respond=direct response; qwen_agent=agent task; memory=memory workflow; research=research workflow; clarify=clarification; escalate=escalation. These are advisory enums; production's LOCAL/TASK/GEMINI categories are a different interface.

Capability names designate domains: none=no capability; application=application launch/control; browser=browser operations; file=file operations; project=project/code inspection; weather/location/reminder/memory/research/system=their named domains; unknown=unresolved capability.

Context names designate conversation/memory/project/current_information/observations/active_task dependencies; none means no such dependency. Missing-information names identify the required target/file/application/url/query/location/time/scope/permission/reference. Escalation reasons identify current-information need, unsupported capability, outside local envelope, repeated failure, context overload or explicit cloud request. Uncertainty high/medium/low/ambiguous/outside_envelope is a descriptive enum, not measured confidence or accuracy; source does not calibrate it.

Actual builder mappings ([build_fast_brain_dataset.py](../benchmarks/fast_classifier/build_fast_brain_dataset.py) lines 38–97; [queue builder](../benchmarks/fast_classifier/review_fast_brain_dataset.py) lines 39–49): open/close_app or application category → application, usually local_task/qwen_agent; find/read/open_file/list_directory → file (project when categorized project/code), usually local_task/qwen_agent; web search/open_url/read_page → browser or research, research/research when research categorized; multi_step path → multi_step_task/qwen_agent; conversation → conversation/respond. open_folder is not specifically recognized by capability mapper: lacking a file category, it can fall through to none/information/respond. That is a data-construction limitation, not permission to invent computer_control. A supported open-folder interpretation uses file with local_task/qwen_agent (or deterministic_action/fast_path for a genuinely deterministic handler); exact choice needs evidence about execution context.

[Deterministic baseline](../assistant/orchestration/fast_brain/deterministic.py) lines 22–48 recognizes stop/time/date/open Chrome/YouTube as deterministic_action/fast_path; defaults other utterances to information/respond. It is not the provenance of all queue annotations.

Current-information retrieval can use research/research/research with current_information context; explicit multi-step research can use multi_step_task/qwen_agent with browser/research capability depending supported operations. Schema permits both capability families and does not enforce a unique combination. File-versus-application distinction matters for opening folders versus launchable applications.

### Reviewer and persistence

[Reviewer](../benchmarks/fast_classifier/review_fast_brain.py): AI_MODES at line 20; proposal validation 126–169; request/parser/policy 179–225; prompt 228 onwards. AUTO_CONFIDENT and AI_CORRECTION require complete valid labels; HUMAN_REVIEW/SOURCE_CONFLICT require null label and human attention. Equality versus correction is not mechanically enforced. Confident labels requiring clarification are downgraded to HUMAN_REVIEW. Unsupported/escalation candidates retain human review. Source-conflict policy checks limited explicit source intent mappings and legacy aliases. Human decisions append only after explicit input; the benchmark does not invoke that persistence path.

Decision fields are the seven-concept label. Utterance, language/script/response language, provenance, ids/groups/split, priority/reasons, model confidence/rationale/source_evidence, timestamps and review status are descriptive/evaluation metadata. Historical original expected labels use a different schema and must not be validated as frozen seven-field decisions or called live predictions.

### Valid examples

Captured fixtures in [regression data](../tests/fixtures/shadow_review_regression.json) provide research/research/research with no clarification, local_task/qwen_agent/application for Open Notepad, and clarification/clarify/application with missing target for Open it. [Focused tests](../tests/test_shadow_review_regression.py) demonstrate full schema validation, malformed outputs, corrections, explicit conflicts and retained human approval.

### Required fields and valid combinations

| Field | Accepted form / meaning | Requiredness |
| --- | --- | --- |
| intent_family | Nine IntentFamily values above; semantic request family | Required string |
| capability_family | Twelve CapabilityFamily values; required capability domain | Required string |
| execution_path | Seven ExecutionPath values; advisory handling path | Required string |
| context_requirements | Nonempty list of ContextRequirement values; none is exclusive | Required list |
| clarification.required | Intended true/false; required iff missing nonempty | Required nested field |
| clarification.missing | List of ten MissingInformation values; empty when not required | Required nested field |
| escalation.candidate | Intended true/false; true iff reason nonnull | Required nested field |
| escalation.reason | Six EscalationReason values or null | Required nested field, nullable |
| uncertainty | Five Uncertainty values; not calibrated correctness | Required string |

Contract enums/models/parser: [contract.py](../assistant/orchestration/fast_brain/contract.py), lines 13–87, 90–160. Cross-field validator: [validator.py](../assistant/orchestration/fast_brain/validator.py), lines 16–38. clarify path and clarification intent require clarification.required; fast_path cannot have unknown capability. There is no general intent/path/capability semantic compatibility validator, strict boolean type validation, or rejection of additional keys.

Three interfaces must not be conflated:

1. The queue’s frozen seven-field labels come from historical annotation normalization, not live inference.
2. Production [router.py](../assistant/orchestration/router.py):60–75 `fast_classify(text)` returns TASK/GEMINI/None; None defers to QwenClassifier.classify at lines 113–138. Only the pure deterministic function was run, never the model branch. Those provider categories are not IntentFamily values.
3. [decision.py](../assistant/orchestration/decision.py):190–253 `FastClassifier` validates an adapter’s older structured contract. Its intent/execution_path are nonempty strings rather than these frozen enums; RequestMode is conversational/memory/research/action. The compact mBERT adapter is an offline candidate using that older contract, not the source of these queue labels. No new encoder inference was started. [encoder_adapter.py](../assistant/orchestration/fast_brain/encoder_adapter.py) is raw-logit foundation code with newly initialized heads, not a trained seven-field decision model.

### Computer actions represented without invented enums

| Input meaning | Supported existing mapping | Evidence / caveat |
| --- | --- | --- |
| Open folder | file; local_task/qwen_agent for a supported agent operation | Stored human folder corrections and schema. Actual historical mapper omits open_folder. |
| Launch app / browser application | application + local_task/qwen_agent; deterministic_action/fast_path for actual direct handlers | Builder recognizes open_app/close_app; tiny baseline recognizes Chrome/YouTube. |
| Open a URL | browser + local_task/qwen_agent | Actual open_url mapping; different from launching a browser app. |
| Find/read file | file + local_task/qwen_agent, or project when explicitly project/code | read_file/find_file/open_file/list_directory mapper. |
| Search web/current information | research/research with browser/research or domain capability | Current-information retrieval; exact route depends task. |
| Multi-step web research | multi_step_task/qwen_agent with browser/research | Historical multi-step examples. |
| Inspect project code | multi_step_task/qwen_agent with project/file domain | grep_code mapper omission is a defect. |
| Conversational/factual response | conversation/respond or information/respond with none | Boundary not fully enforced; full gold needed. |

`computer_control` and execution_path=`local` are invalid in the frozen contract. The Archive request supports a file-action domain; no guessed full replacement was saved. Decision fields are the seven-field label; language/script, provenance, IDs, priority, history and rationale are metadata.

### Complete valid fixture example

Open Notepad fixture (existing test evidence, not a new live prediction):

```json
{
  "capability_family": "application",
  "clarification": {
    "missing": [],
    "required": false
  },
  "context_requirements": [
    "active_task"
  ],
  "escalation": {
    "candidate": false,
    "reason": null
  },
  "execution_path": "qwen_agent",
  "intent_family": "local_task",
  "uncertainty": "medium"
}
```

## 3. Dataset and human-decision inventory

| Quantity | Count |
| --- | --- |
| Queue/dataset | 359 |
| Pending with no decision | 328 |
| accept | 15 |
| correct | 3 |
| skip | 13 |
| reject | 0 |
| ambiguous | 0 |
| Pending or skipped | 341 |
| Missing labels / invalid canonical labels / duplicate IDs | 0 / 0 / 0 |
| Duplicate decision IDs / invalid persisted decisions | 0 / 0 |
| Repeated text | Open my Downloads folder. has two distinct IDs |
| Stored queue status | 359 pending_review; derived history separate |
| Train / development / sealed_candidate | 239 / 57 / 63 |
| Provenance | 359 historical_synthetic from current v4 source |

Decision categories are mutually exclusive; pending-or-skipped overlaps pending by definition. ReviewStore treats a skip as a persisted decision and excludes it from pending. The historical queue status is not rewritten by decision persistence. Auditing all splits means these examples should not be treated as an untouched holdout after tuning against this report.

## 4. Methodology: actual input → actual output

For every record, source v4 utterance/context/capabilities/expected annotation were recovered by v1 original_id. The real `build_fast_brain_dataset.normalize(source, original_dataset)` was run and checked against the stored record. The **real queue-builder main** was run with an isolated temporary output directory; every rebuilt canonical label exactly matches its stored counterpart. No hand-written copy of its label-generation logic was substituted and no original queue was regenerated.

The real production `fast_classify` function was run on each exact utterance with logging/config initialization isolated; no QwenClassifier object or HTTP call was needed. The real `DeterministicFastBrain.decide`/`safe_decide` was additionally run with each source context and advertised capability names. This tiny foundation component is explicitly distinguished from production routing and from stored annotations. All frozen outputs were validated via FastBrainDecision.from_mapping.

Independent notes record each utterance’s primary meaning, all seven fields and remaining context/semantic uncertainties. Domain errors are counted where a file operation has capability none or a project-code operation loses project/file. The file/folder texts were inspected individually, including colloquial Telugu/Hindi/Tamil and code switching. No uncertainty/context/escalation gold was fabricated; no human label was accepted automatically. Structural equality preserves all meaningful values and array order.

## 5. Aggregate results and metric definitions

| Metric | Definition | Result |
| --- | --- | --- |
| Offline record coverage | Unique audited IDs / all current IDs | 359/359 (100%) |
| Schema validity | Labels accepted by actual frozen validator / all labels | 359/359 (100%) |
| Source/output reproducibility | Exact normalizer + queue label reproduction / all records | 359/359 (100%) |
| Capability-domain loss | Individually supported file/project operations with incompatible capability / all records | 46/359 (12.81%) |
| Empty-context active_task mapping | False context_resolvable still mapped to active_task / all records | 319/359 (88.86%) |
| Primary-domain support | No primary domain concern; full fields still unresolved | 193/359 (provisional) |
| Semantic/context concerns | No confirmed capability loss, but other unresolved concerns | 120/359 |
| Full semantic accuracy | Exact independently adjudicated seven-field correctness | Unavailable |
| New model inference after scope change | Local/external model calls in fast-only audit | 0 |

The 46 domain losses partition into 39 file operations and seven project searches. They are not all errors in the original legacy annotation: many legacy annotations correctly name open_folder/recent_files/grep_code and the **conversion** loses that information. This does not measure the production model’s accuracy.

## 6. Results by language, intent and capability

### Language/script

| Value | Records | Schema valid | Capability-domain loss | Model branch not run |
| --- | --- | --- | --- | --- |
| code_switch/mixed | 77 | 77 | 6 | 24 |
| english/latin | 82 | 82 | 10 | 80 |
| hindi/devanagari | 30 | 30 | 3 | 0 |
| hindi/latin | 48 | 48 | 9 | 20 |
| tamil/latin | 31 | 31 | 6 | 31 |
| tamil/tamil | 13 | 13 | 0 | 0 |
| telugu/latin | 48 | 48 | 9 | 17 |
| telugu/telugu | 30 | 30 | 3 | 0 |

### Original canonical intent

| Value | Records | Schema valid | Capability-domain loss | Model branch not run |
| --- | --- | --- | --- | --- |
| conversation | 42 | 42 | 0 | 11 |
| deterministic_action | 7 | 7 | 0 | 1 |
| information | 39 | 39 | 39 | 26 |
| local_task | 92 | 92 | 0 | 42 |
| memory | 38 | 38 | 0 | 20 |
| multi_step_task | 43 | 43 | 7 | 20 |
| research | 43 | 43 | 0 | 23 |
| unsupported | 55 | 55 | 0 | 29 |

### Original canonical capability

| Value | Records | Schema valid | Capability-domain loss | Model branch not run |
| --- | --- | --- | --- | --- |
| application | 42 | 42 | 0 | 11 |
| browser | 55 | 55 | 0 | 28 |
| file | 29 | 29 | 0 | 22 |
| location | 7 | 7 | 0 | 1 |
| memory | 38 | 38 | 0 | 20 |
| none | 143 | 143 | 46 | 67 |
| project | 2 | 2 | 0 | 0 |
| research | 36 | 36 | 0 | 23 |
| weather | 7 | 7 | 0 | 0 |

Metadata categories are actual stored values. code_switch/mixed can include Latin-only transliteration; it is not proof of mixed Unicode scripts. Native and romanized Telugu/Hindi are distinguished by their actual language/script pair. Full-language accuracy is unavailable.

## 7. Full record-level input/output audit

[Complete JSONL](fast_brain_classification_records.jsonl) contains all source inputs, original expected labels, normalized output, stored/rebuilt seven-field labels, pure production fast-path results, separate offline foundation outputs, validation and per-record semantic concerns. [CSV](fast_brain_classification_records.csv) is the compact view; [metrics](fast_brain_classification_metrics.json) includes exact counts. No proposal replaces an original or human label.

| Record ID | Utterance | Canonical intent/path/capability | Schema | Assessment | Production fast-path |
| --- | --- | --- | --- | --- | --- |
| hist-008429183b0de313 | abhi Pune mein weather kaisa hai? | research/research/weather | True | primary_domain_supported_full_label_unresolved | TASK |
| hist-01de1cb0cfcf916c | browser lo https://docs.python.org open cheyyi. | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-02e8f6d6dc03fb5f | kitchen lights band karo | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-05c7a0ac2d55d803 | ivala chala tiring ga undi, konchem matladadama | conversation/respond/none | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-0677a7e5dd3b8149 | team outing ka plan recall karo | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-06d329132f24d014 | నా డౌన్‌లోడ్స్ ఫోల్డర్ తెరువు. | information/respond/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-077b7471a9b6c016 | customer ki chat lo notify cheyyi | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-083d8a0781cf4e8d | నేను ఏ నగరంలో ఉన్నానో చెప్పు. | deterministic_action/respond/location | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-0981ce2da9857e8a | archive directory saamne lao | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-0ad4098dc092e6c8 | डिज़ाइन मीटिंग में हमने क्या तय किया था? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-0bf3c24d92a36351 | Schedule a calendar meeting with the supplier. | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-0e52654409ec9398 | mera resume file dhoondo | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-0f5a36065b4b4f40 | ప్రాజెక్ట్‌లో TODO వ్యాఖ్యలు వెతికి, చూడాల్సిన ఫైళ్లను సంక్షిప్తంగా చెప్పు. | multi_step_task/qwen_agent/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-106e0dba6c2df3db | Rain के बाद rainbow क्यों दिखता है? | conversation/respond/none | True | semantic_or_context_uncertainty | TASK |
| hist-12f3b891f117e30a | Kubernetes overview se main ideas nikalo | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-135af44978a663cd | varsham tarvata rainbow enduku kanipistundi? | conversation/respond/none | True | semantic_or_context_uncertainty | None |
| hist-14b4023aa37a6d05 | supplier-ku chat message anuppu | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-184946cff44a2ae0 | office thermostat-ai kurai | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-188c526533ffa9a7 | ఇప్పుడు పుణెలో వాతావరణం ఎలా ఉంది? | research/research/weather | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-196e919245b653c6 | do current AI stories dhoondh kar dono sources padho aur differences summarize karo. | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-19cc402ffa01eded | spending spreadsheet locate karo | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-1aca000b7d468adf | latest browser security update choodu | research/research/research | True | primary_domain_supported_full_label_unresolved | None |
| hist-1aec025eb71f9c74 | accessibility standard find chesi key points cheppu | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-1b3779922ce65193 | newest Node.js LTS version thedu | research/research/research | True | primary_domain_supported_full_label_unresolved | None |
| hist-1bc454a377585b0a | Post this update to our team channel. | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-1bceebae602d3c6b | official WebAssembly guide find chesi read chesi summarize cheyyi | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-1bd9413c15d79cc2 | today rupee dollar exchange rate entha check cheyyi | research/research/research | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-1c41b233e2339116 | current Rust stable release check karo | research/research/research | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-1c7023d487ce1d97 | ప్రస్తుత ఏఐ వార్తల్లో రెండు కనుగొని, రెండు మూలాలను చదివి, తేడాలు చెప్పు. | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-1cba8cae8c802a5d | Rain తర్వాత rainbow ఎందుకు కనిపిస్తుంది? | conversation/respond/none | True | semantic_or_context_uncertainty | TASK |
| hist-1cdf6ec820439b01 | today yen exchange rate entha | research/research/research | True | semantic_or_context_uncertainty | GEMINI |
| hist-1d0c61727690d563 | archive directory paiki teesuku ra | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-1ece178b121d7a6a | छह बजे दाँत के डॉक्टर को फोन करने की याद दिलाना। | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-1f2c662819b3459f | Notepad band karo | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-227703fcbebfc674 | bedroom thermostat ko twenty degrees par set karo | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-2389e054bcaf97e4 | Today's latest AI news search करो. | research/research/research | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-238c430f0e8e5a8c | రిలీజ్ సమయం గురించి మనం ఏం నిర్ణయించాం? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-24acb98352fb7d2f | Kal architecture ke baare mein humne kya decide kiya tha? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-2663e4567d48674a | current Rust aur Go releases dhoondh kar highlights compare karo | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-26a4f5e3de580fab | 6 PM కి dentist call reminder పెట్టు. | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-27b0849202fff741 | Kubernetes overview-la main ideas sollu | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-2b7a7faa07a9c4c8 | newest Node.js LTS version dhoondo | research/research/research | True | primary_domain_supported_full_label_unresolved | None |
| hist-306d71b9cf620444 | current metro service status vetuku | research/research/research | True | semantic_or_context_uncertainty | None |
| hist-314adf41b0f7b905 | current train schedule kandupidi | research/research/research | True | semantic_or_context_uncertainty | None |
| hist-327c430593a51509 | पत्तों का रंग क्यों बदलता है? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-3321474913885de9 | Notepad ని close చెయ్యి. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-336fc086af5ef8af | landlord-ku message pannu | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-3422cecce46e1d71 | supplier-oda calendar meeting schedule pannu | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-3495ef26749bf241 | मेरा डाउनलोड्स फ़ोल्डर खोलो। | information/respond/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-36728d03f4b87dd4 | డిజైన్ మీటింగ్‌లో మనం ఏం నిర్ణయించాం? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-36a1693ef2fbf34b | Downloads folder open cheyyi | information/respond/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-36faac423da39c1f | नोटपैड बंद कर दो। | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-371b58a61da51017 | do official accessibility guides dhoondh kar differences summarize karo | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-375ae9c4955ed6eb | बारिश के बाद इंद्रधनुष क्यों दिखाई देता है? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-3ae9743492669d4b | budget workbook kahan hai | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-3b70bc749f7864ad | office thermostat tagginchu | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-3ca4207d030ceaf6 | నాకు టెక్స్ట్ ఎడిటర్ ఓపెన్ చేయి. | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-3e5bb2f78cd1dd0e | My current city ఏదో చెప్పు. | deterministic_action/respond/location | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-3fcd2f0b4e19ddd6 | old records folder kholo | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-416d4a931348a344 | project lo TODO comments search chesi attention kavalsina files summarize cheyyi. | multi_step_task/qwen_agent/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-435d789acfbcdd3f | team trip gurinchi em decide chesam | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-4401bc6d09416582 | current train schedule kanipettu | research/research/research | True | semantic_or_context_uncertainty | None |
| hist-441e40c6fcc80ade | current metro service status thedu | research/research/research | True | semantic_or_context_uncertainty | None |
| hist-449744e5a7801a85 | ఈ వారం చాలా కష్టంగా గడిచింది. కాసేపు మాట్లాడుకుందామా? | conversation/respond/none | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-462595b9a7911279 | project mein TODO comments search karke attention wali files summarize karo. | multi_step_task/qwen_agent/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-467c0a4ed94673bd | latest browser security update paaru | research/research/research | True | primary_domain_supported_full_label_unresolved | None |
| hist-4682d9b183375f71 | trip ki eppudu vellali anukunnam | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-47191b87720b3091 | मैंने हाल में जो फ़ाइलें बदली हैं वे दिखाओ। | information/respond/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-478f18ce2d54f9d3 | क्रोम बंद कर दो। | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-489c05eacd816c6d | मेरा टेक्स्ट एडिटर खोलो। | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-49c3fd1eba78e6f8 | batao main kis city mein hoon. | deterministic_action/respond/location | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-4a5f670d2a722c3b | official Git docs research karke workflow summarize karo | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-4ada7aa856247d49 | క్రోమ్‌ను మూసివేయి. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-4b797b494d7bab11 | బడ్జెట్ షీట్‌ను ప్రియకు ఈమెయిల్ చేయి. | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-4be156b9055707ea | archive folder par le chalo | information/respond/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-4db891188ee03e40 | Send a chat message to the supplier. | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-4dd3d16135b73da2 | Lower the office thermostat. | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-4f304ee9acbae39c | YouTube lo focus music vetuku. | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-4f8fdcb874454333 | Downloads folder kholo | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-502341f972c342b9 | current Rust stable release enna check pannu | research/research/research | True | primary_domain_supported_full_label_unresolved | None |
| hist-5184f3e5814d911a | ivala main AI news kosam search cheyyi. | research/research/research | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-525cc654872d829b | team channel par update post karo | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-526f032ce6e5bc68 | ఎడిటర్‌ని ప్రారంభించు. | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-528b12a42c6388af | ee week chala kashtanga undi, kasepu matladadama? | conversation/respond/none | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-52ebdbd7df61455b | kitchen lights off cheyyi | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-53622e30826e0ba4 | Email the budget sheet to Priya. | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-54658abb906cb901 | ఈ రోజు ముఖ్యమైన ఏఐ వార్తల కోసం వెతుకు. | research/research/research | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-54674783aae5c05e | six ki dentist ki call cheyamani remind cheyyi. | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-55b6ac70330f5de4 | team outing plan recall cheyyi | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-55dbb0fcbc2562dd | travel decision yaad dilao | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-5771c52e64ae28c7 | invoice accountant ko email karo | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-5819f55a9913b475 | naa Downloads folder open cheyyi. | information/respond/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-5819fe4bed2259fa | barish ke baad rainbow kyon dikhai deta hai? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-58b773c7f07da60e | Calculator को open करो. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-590b9367f67077f8 | Project में TODO comments grep करके files summarize करो. | multi_step_task/qwen_agent/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-5bd56535c3385742 | ఎడిటర్ తెరవగలవా? | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-5c446364b6333adf | photo messaging app-la anuppu | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-5cbdf6c46dc04292 | Two current AI stories find చేసి, sources చదివి differences summarize చెయ్యి. | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-5d0011dacf129a45 | team trip ke baare mein kya decide kiya tha | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-5d0cd5cd730417ed | supplier tho calendar meeting schedule cheyyi | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-5dfff40af77711b8 | Set the bedroom thermostat to twenty degrees. | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-5f147e21513ccce4 | Recently changed files मुझे दिखाओ. | information/respond/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-5f173c694ac9ed2d | ఆరు గంటలకు దంత వైద్యుడికి ఫోన్ చేయమని గుర్తు చేయి. | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-5f6ae34e3b88424b | budget sheet Priya ki email cheyyi. | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-605bcb224c8f942c | వర్షం తర్వాత ఇంద్రధనస్సు ఎందుకు కనిపిస్తుంది? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-61a0fe1a73ae2e0d | एडिटर शुरू कर दो। | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-624dda5eea2e80a7 | official Git docs research chesi workflow summarize cheyyi | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-627154af6f328049 | invoice-july.pdf అనే ఫైల్‌ను కనుగొను. | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-62722a28e17d961f | aaj ka rupee dollar exchange rate check karo | research/research/research | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-656a862113910d46 | invoice-july.pdf ane file ni vetuku. | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-658fcfbf4bb6b4ac | newest Node.js LTS version vetuku | research/research/research | True | primary_domain_supported_full_label_unresolved | None |
| hist-65f7683a669ce5a0 | ippudu Pune lo weather ela undi? | research/research/weather | True | primary_domain_supported_full_label_unresolved | TASK |
| hist-67b55ed3dedd1fa3 | supplier ki chat message pampu | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-6838e9232f82f9ee | అయస్కాంతం లోహాన్ని ఎలా ఆకర్షిస్తుంది? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-6883dedf6b0800c5 | बजट शीट प्रिया को ईमेल करो। | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-691be5eaf1338525 | YouTube पर focus music search करो. | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-69f6f3121369a1ca | mera budget spreadsheet dhoondo | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-6a719f36b04b04f2 | aaj yen exchange rate kya hai | research/research/research | True | semantic_or_context_uncertainty | GEMINI |
| hist-6bb10d7ab5ec6e5d | Notepad को close कर दो. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-6d02f178d61b7502 | This week बहुत tough था, थोड़ी देर talk करें? | conversation/respond/none | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-6d26fda7b387cbf0 | current Rust stable release enti check cheyyi | research/research/research | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-6d7c401d84c89de9 | bedroom thermostat twenty degrees ki set cheyyi | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-6e1d9b66c80c8167 | चुंबक धातु को क्यों खींचता है? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-6e791d1ec3a8c7d2 | My current city कौन सा है बताओ. | deterministic_action/respond/location | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-6f2f8975bd351cff | ఆకాశం నీలంగా ఎందుకు కనిపిస్తుంది? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-6fc72ec0734459ec | files mein budget sheet dhoondo | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-738955a1c884eb95 | calculator kholo. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | None |
| hist-73ff50faf5b29b17 | current Rust Go releases find chesi highlights compare cheyyi | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-75560bed27bdbf0e | यह हफ्ता बहुत मुश्किल था। थोड़ी देर बात करें? | conversation/respond/none | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-756ccd7f4537ff5a | एआई की दो मौजूदा खबरें खोजो, दोनों स्रोत पढ़ो और अंतर बताओ। | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-7759cddb59f5261a | current metro service status dhoondo | research/research/research | True | semantic_or_context_uncertainty | None |
| hist-7778d19bf8c08e18 | Remind me at six to call the dentist. | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-781775a4ab56397b | क्या एडिटर खोल सकते हो? | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-7b4d0d960e40c232 | two official accessibility guides find chesi differences summarize cheyyi | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-7cdb2dfb673ab704 | official SQLite architecture page search panni brief kudu | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-7d3b374f9386cb3b | budget workbook ekkada undo choodu | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-7dbd3c977978cff5 | https://docs.python.org browser lo open cheyyi | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-7e4a72d37dcd063c | YouTube par focus music dhoondo. | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-7e774bed40d5eacc | Today's latest AI news search చెయ్యి. | research/research/research | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-7f84fdd3cc547124 | accessibility standard padichu key points sollu | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-7fc4d6c128901e49 | official Git docs research panni workflow summarize pannu | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-80c531918cfa8825 | team channel lo update post cheyyi | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-80c54f4904a8d65b | टेक्स्ट एडिटर चला दो। | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-843676de39b218d4 | Notepad close cheyyi. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-84657f948f9bb482 | Release window के बारे में हमारा decision क्या था? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-85caebba7ee5699f | Browser में https://docs.python.org open करो. | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-86bbe72d3cf2068b | यूट्यूब पर काम के लिए संगीत खोजो। | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-86c944a1d73fa8c0 | रिलीज़ के समय के बारे में हमने क्या तय किया था? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-870a5c93a7477871 | ఆకుల రంగు ఎందుకు మారుతుంది? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-870b18fc0efea95d | nenu recent ga change chesina files chupinchu. | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-8a5e38df1fb1da78 | invoice-july.pdf file ని find చెయ్యి. | local_task/qwen_agent/project | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-8becbcb8a787fb68 | landlord ko message karo | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-8ef0e62720297850 | team channel-la update post pannu | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-90b662cc3b60f294 | supplier ke saath calendar meeting schedule karo | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-90ea1ad2912f2ac9 | official SQLite architecture page search karke brief do | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-91cebbf8cdc23cde | Notify the customer by chat. | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-92b148db50812bff | supplier ko chat message bhejo | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-9342b186021415dd | Recently changed files నాకు చూపించు. | information/respond/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-9397ece3b49563dd | accessibility standard padh kar key points batao | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-94e80af1a58cc2e8 | ఇంద్రధనస్సు ఎలా ఏర్పడుతుంది? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-95cdb48c5a876784 | నోట్‌ప్యాడ్ మూసివేయి. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-981144e03297e05d | Two current AI stories find करके, sources पढ़ो और differences summarize करो. | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-999d532b4ba88571 | बर्फ पानी पर क्यों तैरती है? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-99ea362c071fb8e9 | budget file search karo | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-9a03ab20e1224772 | Notepad close cheyyi | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-9b522b0cfa536fbb | trip ke liye kab nikalna tha | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-9bec38cf7188d0c5 | aaj ki main AI news search karo. | research/research/research | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-9c2419bbaf0f5a8c | बताओ मैं किस शहर में हूँ। | deterministic_action/respond/location | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-9ce0edb2ab896d08 | https://docs.python.org browser mein kholo | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-9dc8e7eac186e600 | six baje dentist ko call karne ka reminder laga do. | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-9e5db5008828e742 | invoice-ai accountant-ku email pannu | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-9fa708933644fc90 | My Downloads folder खोलो. | information/respond/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-9fc76391f3931925 | archive folder kholo | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-a079981e52205fc9 | Calculator ని open చెయ్యి. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-a1e427b1c342fb6c | టెక్స్ట్ ఎడిటర్ తెరువు. | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-a2196a5877d6a1f7 | official WebAssembly guide dhoondh kar padh ke summarize karo | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-a3caf59c0df53070 | invoice-july.pdf file को find करो. | local_task/qwen_agent/project | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-a48165e55d3512bd | naa budget spreadsheet kanipettu | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-a64fd40be96c452c | ye hafta bahut mushkil tha, thodi der baat karein? | conversation/respond/none | True | primary_domain_supported_full_label_unresolved | None |
| hist-a672eb269847b583 | calculator open cheyyi. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-a6bef85f1460635d | Budget sheet को Priya को email करो. | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-a727ccca0966aa98 | archive folder open cheyyi | information/respond/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-a885aff1a7467f03 | official SQLite architecture page search chesi brief ivvu | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-a9d2a1fec7ae0fb9 | customer ko chat par notify karo | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-ac51b18856f1e8eb | अभी पुणे में मौसम कैसा है? | research/research/weather | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-acb81c3a12d86f63 | Browser లో https://docs.python.org open చెయ్యి. | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-ad2fecc042abacb2 | Turn off the kitchen lights. | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-b03ef1d458722d2b | invoice-july.pdf नाम की फ़ाइल ढूँढो। | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-b0e19fb196d7977f | mera Downloads folder kholo. | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-b18ea356bfc4769f | two official accessibility guides thedi differences summarize pannu | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-b2f3b731c99e7d0e | My Downloads folder తెరువు. | information/respond/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-b38fb0b7c2ab38c8 | budget file vetuku | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-b392082d77813f05 | Email the invoice to the accountant. | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-b50d57dfa8815884 | 6 PM पर dentist call reminder लगा दो. | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-bba589eb50161f78 | budget sheet Priya ko email karo. | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-bcda86eb773892c7 | today rupee dollar exchange rate enna check pannu | research/research/research | True | primary_domain_supported_full_label_unresolved | None |
| hist-bcdc4351b4033198 | naa resume file vetuku | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-be43bbffebcd406a | Budget sheet ని Priya కి email చెయ్యి. | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-beffb472e5239d9d | నేను ఇటీవల మార్చిన ఫైళ్లను చూపించు. | information/respond/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-bfb433d9b7ee212d | files lo budget sheet search cheyyi | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-c0083090edda9dd6 | old records folder open cheyyi | information/respond/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-c0298b736337c818 | This week చాలా tough గా ఉంది, కాసేపు talk చేద్దామా? | conversation/respond/none | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-c09b400d2169cfed | office thermostat kam karo | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-c115df669d828467 | browser mein https://docs.python.org kholo. | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-c13105da38ca5149 | today yen exchange rate enna | research/research/research | True | semantic_or_context_uncertainty | None |
| hist-c19b4d776598baab | archived files directory dikhao | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-c1c9b9a526598ab7 | archive folder ki teesuku vellu | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-c1ef53f94bfd63d3 | టెక్స్ట్ ఎడిటర్ రన్ చేయి. | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-c27331f554a6655c | current Rust Go releases thedi highlights compare pannu | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-c70b0f0139db41ac | ब्राउज़र में https://docs.python.org खोलो। | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-c78bf3edcb9c9090 | maine recently change ki files dikhao. | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-c914535c39b27f11 | Message the landlord for me. | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-ccd6cdb094b6e4fa | Project లో TODO comments grep చేసి files summarize చెయ్యి. | multi_step_task/qwen_agent/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-cd8511b1b33c8f12 | two current AI stories find chesi, rendu sources chadivi differences summarize cheyyi. | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-cee8cdf172cf3d21 | गूंज कैसे बनती है? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-cfe651389cdd4a91 | ప్రతిధ్వని ఎలా వస్తుంది? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-d1422a8b64138f34 | nenu ye city lo unnano cheppu. | deterministic_action/respond/location | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-d219754118143aea | release time gurinchi manam em decide chesam? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-d2efe1810ef782a6 | latest browser security update dekho | research/research/research | True | primary_domain_supported_full_label_unresolved | None |
| hist-d3a5cde146e999d6 | Orion project ke liye kya chuna tha yaad dilao | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-d3ca9ee15500dff7 | YouTube లో focus music search చెయ్యి. | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-d45a42ef46ef1df1 | spending spreadsheet dorikinchu | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-d4f271d8813da992 | Notepad band karo. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-d7d041838858a4ac | Orion project ki em choose chesamo gurthu cheyyi | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-d8148ebd2827ae2b | archived files directory chupinchu | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-d8c9d0226aabde4d | customer-ku chat-la notify pannu | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-d9af420dbd534d83 | Could you put on some instrumental focus music? | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-da2923cc8285dce3 | landlord ki message cheyyi | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-daada2b39fd85e9d | bedroom thermostat-ai twenty degrees-ku set pannu | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-db622d208ae0cb51 | Send the photo through a messaging app. | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-dc1a50837063e98d | Release window గురించి మన decision ఏంటి? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-dc7963537a27a433 | photo messaging app lo pampu | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-dcfa5af61fc34283 | kitchen lights-ai off pannu | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-ddb4e670e07977e9 | travel gurinchi mana conclusion enti | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-e0c4844e927565c0 | mana travel decision gurthu cheyyi | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-e249ec2111974dd9 | टेक्स्ट एडिटर खोलो। | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-e4551ea90381ab25 | బ్రౌజర్‌లో https://docs.python.org తెరువు. | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-e56e8b9439062ba4 | యూట్యూబ్‌లో పని చేసేటప్పుడు వినే సంగీతం వెతుకు. | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-e804749891acb51e | Pune में current weather बताओ. | research/research/weather | True | primary_domain_supported_full_label_unresolved | TASK |
| hist-e832590b5ae99c60 | travel par hamara conclusion kya tha | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-ed03632a3e69c7db | మంచు నీటిపై ఎందుకు తేలుతుంది? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-ef808ac680537dee | आज की प्रमुख एआई खबरें खोजो। | research/research/research | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-f15da07032e76d47 | आसमान नीला क्यों दिखता है? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-f18ea96eda0bcc63 | release time ke baare mein humne kya decide kiya tha? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-f2c0d48c82476f5f | photo messaging app se bhejo | unsupported/respond/none | True | semantic_or_context_uncertainty | None |
| hist-f55992c3591e5c22 | current train schedule batao | research/research/research | True | semantic_or_context_uncertainty | GEMINI |
| hist-f5ae834286032ac4 | प्रोजेक्ट में TODO टिप्पणियाँ खोजो और जिन फ़ाइलों पर ध्यान चाहिए उनका सार बताओ। | multi_step_task/qwen_agent/none | True | confirmed_capability_domain_loss | GEMINI |
| hist-f6ae186d9c5726cc | invoice accountant ki email cheyyi | unsupported/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-f7de047b62fe51c4 | कैलकुलेटर खोलो। | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-f817dc1f64f1be8d | కాలిక్యులేటర్ తెరువు. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-fab81e4e51010aad | Pune లో current weather చెప్పు. | research/research/weather | True | primary_domain_supported_full_label_unresolved | TASK |
| hist-fb29bd5958b2516f | इंद्रधनुष कैसे बनता है? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-fb5204f607b5f3ff | invoice-july.pdf naam ki file dhoondo. | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-fc2f6f9cc281cf70 | official WebAssembly guide thedi padichu summarize pannu | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-fc9c05a57e24554b | Kubernetes overview vetiki main ideas cheppu | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-fcaf68bb165708ae | aaj din lamba tha, thodi baat karte hain | conversation/respond/none | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-069dcde66a8f9794 | Launch Calculator. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | None |
| hist-0809d6467b777f81 | Check today's rupee to dollar exchange rate. | research/research/research | True | primary_domain_supported_full_label_unresolved | None |
| hist-0a5f2eebf38ce2e2 | Bring up a text editor. | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | None |
| hist-0a5fdc8884e2e50b | Show me the archived files directory. | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-0a8e23fe284bd78f | Take me to the Python documentation site. | research/research/browser | True | semantic_or_context_uncertainty | None |
| hist-0ac2e53728036b5e | archive folder-ai open pannu | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-0cbd61671f49eccb | Open my old-records folder. | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-0f4e3e0a4e42b3c2 | Remind me of our travel decision. | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-14bca63058beb4a8 | Locate two official accessibility guides, read them, and summarize the differences. | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-19196a26a48d162e | Find today's exchange rate for yen. | research/research/research | True | semantic_or_context_uncertainty | None |
| hist-1ce6a2d1218448e8 | Open https://docs.python.org in my browser. | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-1db8987fb7c7a9a6 | team trip pathi enna decide pannom | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-1df8d3ccd087e273 | Research the official Git documentation and summarize the workflow. | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-20b07a7ae81c71e3 | டெக்ஸ்ட் எடிட்டரை திற. | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-25808a137c379c4d | Find the current train schedule. | research/research/research | True | semantic_or_context_uncertainty | None |
| hist-28721da13b5abd0d | https://docs.python.org browser-la open pannu | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-29bd86a063b8e8be | spending spreadsheet kandupidi | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-2ab4a4ab05600583 | Open my Downloads folder. | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-2c584b6189ec55f3 | Help me find the spending spreadsheet. | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-2f2fae954b058e41 | Look up the latest browser security update. | research/research/research | True | primary_domain_supported_full_label_unresolved | None |
| hist-35138bdc0c6c8e35 | Open my Downloads folder. | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-3557f9a7869e5418 | Tell me which city I am in. | deterministic_action/respond/location | True | primary_domain_supported_full_label_unresolved | None |
| hist-395b50a21bf06f39 | How do rainbows form? | conversation/respond/none | True | semantic_or_context_uncertainty | None |
| hist-39614adaeaf35569 | Orion project-ku enna choose pannom nyabagam pannu | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-42ccc48aa7aba6e3 | Please close Notepad. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | None |
| hist-450397fbd0b31ce0 | Show me the files I changed recently. | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-463d72f30d08ad1e | Search for today's major AI news. | research/research/research | True | primary_domain_supported_full_label_unresolved | None |
| hist-487b7aa07812922f | I had a long day and could use a quick chat. | conversation/respond/none | True | primary_domain_supported_full_label_unresolved | None |
| hist-494e20abed197856 | I need the text editor running. | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | None |
| hist-49aa7a18656b7944 | Where is the budget file? | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-4b3d7fc39ce0dafc | Recall the plan we made for the team outing. | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-51331eb15fd7fcc3 | எடிட்டரை திறக்க முடியுமா? | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-51c17b2b3a3f4ef2 | Why does a rainbow appear after rain? | conversation/respond/none | True | semantic_or_context_uncertainty | TASK |
| hist-52d9d3a2ef14e93d | Find the current metro service status. | research/research/research | True | semantic_or_context_uncertainty | None |
| hist-5418ac4a5b5a539b | டெக்ஸ்ட் எடிட்டரை இயக்கவும். | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-5b7be800025e08e4 | files-la budget sheet thedu | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-5bf7e22c9fcd9522 | When did we agree to leave for the trip? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-5e9f7f9912bd12b9 | Locate the budget workbook. | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-60e3a4c7ff350c83 | Open https://docs.python.org in the browser. | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-6231b669658583b6 | What did we decide about the team trip? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-6243b39c230929ee | Find two current AI stories, read both sources, and summarize the differences. | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-62ad725dcffe0744 | Why does ice float? | conversation/respond/none | True | semantic_or_context_uncertainty | None |
| hist-62e85c47d8d9ee81 | Find focus music on YouTube. | local_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-635576bfbc6907f7 | Open the second one. | local_task/qwen_agent/browser | True | semantic_or_context_uncertainty | None |
| hist-649a9ad12b47ab8b | travel decision nyabagam pannu | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-6535112d2152c331 | travel pathi namma conclusion enna | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-6c1d3caff712a040 | I'm doing better today; open my notes folder while we talk. | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-6cf710b14ce77f95 | Check the current stable Rust release. | research/research/research | True | primary_domain_supported_full_label_unresolved | None |
| hist-6e02c49c27ef9655 | Search my files for the budget sheet. | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-73dbdefdeb1f1654 | Find the file named invoice-july.pdf. | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-75fc31c49c351aaf | இலைகளின் நிறம் ஏன் மாறுகிறது? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-777f0f17f062b311 | Close Notepad. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | None |
| hist-779575fec4c0f0aa | Find my budget spreadsheet. | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-7a716953cc443231 | Do you remember our hologram project? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-7c3c403b662c82af | Find the official WebAssembly guide, read it, and summarize it. | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-7c8f9d31af3a5e6c | inniku romba tiring-a irundhuchu, konjam pesalama | conversation/respond/none | True | primary_domain_supported_full_label_unresolved | None |
| hist-806b3ca192c2facd | archive directory kondu vaa | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-833fbb9af9ec69c9 | Notepad-ai close pannu | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | None |
| hist-89dfef657c756913 | en resume file-ai thedu | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-8a11c5a6b1d96132 | Take me to the archive folder. | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-8a1c279fd51731d9 | வடிவமைப்பு கூட்டத்தில் நாம் என்ன முடிவு செய்தோம்? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-8f10b6d09744ec29 | Search the project for TODO comments and summarize the files that need attention. | multi_step_task/qwen_agent/none | True | confirmed_capability_domain_loss | None |
| hist-90de7abdd58d3113 | Explain why the sky looks blue. | conversation/respond/none | True | semantic_or_context_uncertainty | None |
| hist-927e31a1619df89d | Why do leaves change colour? | conversation/respond/none | True | semantic_or_context_uncertainty | None |
| hist-935f20befd6ed677 | budget file search pannu | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-959feeef7a84d16f | குரோமை மூடு. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | GEMINI |
| hist-95ebea2001f5d2b7 | What did we settle on in the design meeting? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-985058a3ba7eb4aa | Start my editor. | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | None |
| hist-9ae1b045a7bcd06f | Look up the official LangGraph overview, read it, and give me the key design ideas. | multi_step_task/qwen_agent/research | True | primary_domain_supported_full_label_unresolved | None |
| hist-9be37e1508c71545 | I had a rough week. Can we talk for a bit? | conversation/respond/none | True | primary_domain_supported_full_label_unresolved | None |
| hist-9c67fdb0f511d25d | archived files directory kaatu | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-a31ff199ab73512b | Find the accessibility standard, read it, then list key points. | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-a97951651cf8a4cd | காந்தம் உலோகத்தை எப்படி ஈர்க்கிறது? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-b15cca071bf00015 | old records folder-ai open pannu | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-b21b6dbb2982fd85 | Downloads folder-ai open pannu | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-b27cfd78e653f624 | trip-ku eppo kelamba mudivu pannom | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-b2892551ab7c2da1 | Research the newest Node.js LTS version. | research/research/research | True | primary_domain_supported_full_label_unresolved | None |
| hist-b37d640402daa329 | Open the archive folder. | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-b73949baa37fdf13 | Search for the official SQLite architecture page and brief me. | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-b94e68f5d4e69943 | Remind me what we chose for the Orion project. | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-ba87774b2cb54cd1 | பனி ஏன் நீரில் மிதக்கிறது? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-bae42cf9971e4a3f | Close Chrome for me. | local_task/qwen_agent/application | True | primary_domain_supported_full_label_unresolved | None |
| hist-c1267f8c07829179 | What did we decide about the database migration last week? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-c49a5e2b6fe12610 | வானவில் எப்படி உருவாகிறது? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-c6e11201e8761710 | என் டெக்ஸ்ட் எடிட்டரை திறக்கவும். | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-cd06c5b65cc6a037 | Locate the Kubernetes overview and extract its main ideas. | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-cd74e36a5634d0e4 | What was our conclusion about travel? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-d0418366cd05ce0a | எதிரொலி எப்படி உருவாகிறது? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-d0cc0d8789c83ffb | budget workbook enga irukku | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-d152c0516bf7828f | Find the current Rust and Go releases, then compare their highlights. | multi_step_task/qwen_agent/browser | True | primary_domain_supported_full_label_unresolved | None |
| hist-da4fc136644b97fe | What causes an echo? | conversation/respond/none | True | semantic_or_context_uncertainty | None |
| hist-ded1b5db93caeaba | en budget spreadsheet-ai kandupidi | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-e0d376a432981c0a | What did we decide about the release window? | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-e2b494cead1d9e79 | How do magnets attract metal? | conversation/respond/none | True | semantic_or_context_uncertainty | None |
| hist-e6dbad9de51f072d | Find my resume file. | local_task/qwen_agent/file | True | primary_domain_supported_full_label_unresolved | None |
| hist-eae36499e26917cd | team outing plan recall pannu | memory/memory/memory | True | primary_domain_supported_full_label_unresolved | None |
| hist-ecfd7b27ed5a44c6 | Launch the text editor. | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | None |
| hist-edca6c4a272343f6 | archive folder-ku kondu po | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-f2a4fd1032c829b5 | வானம் ஏன் நீலமாக தெரிகிறது? | conversation/respond/none | True | semantic_or_context_uncertainty | GEMINI |
| hist-f57a957909f94f29 | எடிட்டரை தொடங்கு. | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | GEMINI |
| hist-f7884fdea2c61c34 | What is the weather in Pune right now? | research/research/weather | True | primary_domain_supported_full_label_unresolved | TASK |
| hist-f8717c13ca31991d | Bring up the archive directory. | information/respond/none | True | confirmed_capability_domain_loss | None |
| hist-fd1606f4cde3cccc | Can you open the editor? | local_task/qwen_agent/application | True | semantic_or_context_uncertainty | None |

## 8. Historical decisions: preserved audit

[All 31 historical decisions](fast_brain_classification_history.json) retain their original records and labels. Fifteen accepts, three corrections and thirteen skips; no rejects or ambiguous decisions. All stored reviewed labels validate. Two corrections materially change folder labels to local_task/qwen_agent/file. The team-channel ‘correct’ decision is identical to canonical and is a structural no-op. Historical raw Qwen proposal provenance cannot be recovered from this store; accepts are not automatically called accepted Qwen proposals.

### accept (15)

| ID | Utterance | Reviewed label identical to canonical | Schema error |
| --- | --- | --- | --- |
| hist-008429183b0de313 | abhi Pune mein weather kaisa hai? | True | None |
| hist-01de1cb0cfcf916c | browser lo https://docs.python.org open cheyyi. | True | None |
| hist-02e8f6d6dc03fb5f | kitchen lights band karo | True | None |
| hist-05c7a0ac2d55d803 | ivala chala tiring ga undi, konchem matladadama | True | None |
| hist-0677a7e5dd3b8149 | team outing ka plan recall karo | True | None |
| hist-077b7471a9b6c016 | customer ki chat lo notify cheyyi | True | None |
| hist-083d8a0781cf4e8d | నేను ఏ నగరంలో ఉన్నానో చెప్పు. | True | None |
| hist-0ad4098dc092e6c8 | डिज़ाइन मीटिंग में हमने क्या तय किया था? | True | None |
| hist-1b3779922ce65193 | newest Node.js LTS version thedu | True | None |
| hist-1bceebae602d3c6b | official WebAssembly guide find chesi read chesi summarize cheyyi | True | None |
| hist-1bd9413c15d79cc2 | today rupee dollar exchange rate entha check cheyyi | True | None |
| hist-1c41b233e2339116 | current Rust stable release check karo | True | None |
| hist-1c7023d487ce1d97 | ప్రస్తుత ఏఐ వార్తల్లో రెండు కనుగొని, రెండు మూలాలను చదివి, తేడాలు చెప్పు. | True | None |
| hist-1cba8cae8c802a5d | Rain తర్వాత rainbow ఎందుకు కనిపిస్తుంది? | True | None |
| hist-1cdf6ec820439b01 | today yen exchange rate entha | True | None |

### correct (3)

| ID | Utterance | Reviewed label identical to canonical | Schema error |
| --- | --- | --- | --- |
| hist-06d329132f24d014 | నా డౌన్‌లోడ్స్ ఫోల్డర్ తెరువు. | False | None |
| hist-0981ce2da9857e8a | archive directory saamne lao | False | None |
| hist-1bc454a377585b0a | Post this update to our team channel. | True | None |

### reject (0)

None persisted.

### ambiguous (0)

None persisted.

### skip (13)

| ID | Utterance | Reviewed label identical to canonical | Schema error |
| --- | --- | --- | --- |
| hist-0bf3c24d92a36351 | Schedule a calendar meeting with the supplier. | None | None |
| hist-0e52654409ec9398 | mera resume file dhoondo | None | None |
| hist-0f5a36065b4b4f40 | ప్రాజెక్ట్‌లో TODO వ్యాఖ్యలు వెతికి, చూడాల్సిన ఫైళ్లను సంక్షిప్తంగా చెప్పు. | None | None |
| hist-106e0dba6c2df3db | Rain के बाद rainbow क्यों दिखता है? | None | None |
| hist-12f3b891f117e30a | Kubernetes overview se main ideas nikalo | None | None |
| hist-135af44978a663cd | varsham tarvata rainbow enduku kanipistundi? | None | None |
| hist-14b4023aa37a6d05 | supplier-ku chat message anuppu | None | None |
| hist-184946cff44a2ae0 | office thermostat-ai kurai | None | None |
| hist-188c526533ffa9a7 | ఇప్పుడు పుణెలో వాతావరణం ఎలా ఉంది? | None | None |
| hist-196e919245b653c6 | do current AI stories dhoondh kar dono sources padho aur differences summarize karo. | None | None |
| hist-19cc402ffa01eded | spending spreadsheet locate karo | None | None |
| hist-1aca000b7d468adf | latest browser security update choodu | None | None |
| hist-1aec025eb71f9c74 | accessibility standard find chesi key points cheppu | None | None |

Accepts involving yen exchange rates retain counter-currency uncertainty; stable knowledge labeled conversation has an unresolved information/conversation boundary. Unsupported-operation labels depend on capability availability. These concerns do not authorize edits to human decisions. Pending IDs are all 328 audit rows without a persisted decision.

## 9. Critical input/output findings

| Group | Records | Lost information |
| --- | --- | --- |
| v4_open_downloads | 7 | file capability and action handling |
| synth_open_archive_folder | 20 | file capability and action handling |
| v4_project_todo_inspection | 7 | project/file capability |
| train_open_downloads | 4 | file capability and action handling |
| v4_recent_files | 7 | file capability and action handling |
| mixed_conversation_action | 1 | file capability and action handling |

### archive directory paiki teesuku ra

ID `hist-1d0c61727690d563`. Source expected: `{"context_references": [], "execution_path": "single_action", "intent": "use:open_folder", "missing_information": [], "needs_clarification": false, "needs_escalation": false, "request_modes": ["action"], "selected_capability": "open_folder"}`. Actual constructed label: `{"capability_family":"none","clarification":{"missing":[],"required":false},"context_requirements":["active_task"],"escalation":{"candidate":false,"reason":null},"execution_path":"respond","intent_family":"information","uncertainty":"medium"}`. Assessment: confirmed_capability_domain_loss. Canonical capability none and respond path do not represent the requested file operation.

### मैंने हाल में जो फ़ाइलें बदली हैं वे दिखाओ।

ID `hist-47191b87720b3091`. Source expected: `{"context_references": [], "execution_path": "single_action", "intent": "use:recent_files", "missing_information": [], "needs_clarification": false, "needs_escalation": false, "request_modes": ["action"], "selected_capability": "recent_files"}`. Actual constructed label: `{"capability_family":"none","clarification":{"missing":[],"required":false},"context_requirements":["active_task"],"escalation":{"candidate":false,"reason":null},"execution_path":"respond","intent_family":"information","uncertainty":"medium"}`. Assessment: confirmed_capability_domain_loss. Canonical capability none and respond path do not represent the requested file operation.

### ప్రాజెక్ట్‌లో TODO వ్యాఖ్యలు వెతికి, చూడాల్సిన ఫైళ్లను సంక్షిప్తంగా చెప్పు.

ID `hist-0f5a36065b4b4f40`. Source expected: `{"context_references": [], "execution_path": "multi_step_task", "intent": "research", "missing_information": [], "needs_clarification": false, "needs_escalation": false, "request_modes": ["research", "action"], "selected_capability": "grep_code"}`. Actual constructed label: `{"capability_family":"none","clarification":{"missing":[],"required":false},"context_requirements":["active_task"],"escalation":{"candidate":false,"reason":null},"execution_path":"qwen_agent","intent_family":"multi_step_task","uncertainty":"medium"}`. Assessment: confirmed_capability_domain_loss. Canonical multi_step_task is supported; capability none omits the source/code domain. grep_code absent from mapper.

### latest browser security update choodu

ID `hist-1aca000b7d468adf`. Source expected: `{"context_references": [], "execution_path": "research", "intent": "research", "missing_information": [], "needs_clarification": false, "needs_escalation": false, "request_modes": ["research", "action"], "selected_capability": "google_search"}`. Actual constructed label: `{"capability_family":"research","clarification":{"missing":[],"required":false},"context_requirements":["active_task"],"escalation":{"candidate":false,"reason":null},"execution_path":"research","intent_family":"research","uncertainty":"medium"}`. Assessment: primary_domain_supported_full_label_unresolved.

**Confirmed mapper omissions:** [build_fast_brain_dataset.py](../benchmarks/fast_classifier/build_fast_brain_dataset.py):38–67 recognizes file/app/browser operations but omits open_folder, recent_files and grep_code. This leads to capability none; intent subsequently falls through to information for file actions. [review_fast_brain_dataset.py](../benchmarks/fast_classifier/review_fast_brain_dataset.py):39–49 then derives respond or qwen_agent from that normalized intent and uses `context_resolvable is not None`, not context truth. No schema violation catches these semantic losses.

**Production deterministic fast path:** 8 TASK, 179 GEMINI, 172 None. None is a deferral, not a completed LOCAL/GEMINI/TASK prediction. Example “Why does a rainbow appear after rain?” returns TASK because of rain, despite being stable factual explanation; the hard keyword rule does not distinguish current weather from explanatory questions. Native Indic text returns GEMINI by explicit language policy, which cannot be scored as a seven-field capability decision. No downstream execution occurred.

**Tiny offline foundation baseline:** information/respond/none for all 359 cases. No stop/time/date/open-Chrome/YouTube patterns matched this corpus. It ignores the available context/capabilities outside its tiny pattern set and cannot independently classify these tasks. It is not the original queue classifier and is not evidence that production returned that label 359 times.

**Ambiguity limitations:** zero queue labels require clarification. Some yen/train/metro/editor requests omit execution-specific details. One “Open the second one” source annotation contains a resolved URL, though raw source search results are not copied into the reviewer queue. These cases cannot justify a global claim of correctly handling missing referents. All 319 empty-context rows require review of active_task semantics; source capability descriptors are omitted from queue metadata.

## 10. Verification and tests

Fast-classifier-focused command: `python -B C:\Users\Jagadheesh\AppData\Local\Temp\jarvis-full-review-benchmark-20261010-121133\run_fast_tests.py` — **42 passed in 2.40 seconds** (FastClassifier contract, classifier benchmark harness, Fast Brain dataset, foundation). Earlier scoped evaluator/reviewer command: `python -B C:\Users\Jagadheesh\AppData\Local\Temp\jarvis-full-review-benchmark-20261010-121133\run_tests.py` — **76 passed in 1.62 seconds**. Counts overlap and are not added as unique tests. No skipped tests. Both runners block network/process/project writes and isolate eager production logging. The temporary fast-test wrapper initially hit an inherited UTF-8 BOM; corrected the temporary reader to utf-8-sig and reran successfully. No production change was needed.

All 359 actual normalizer outputs and rebuilt queue labels match; all actual frozen schemas pass. Real pure-function evaluation completed in 0.156 seconds (not a model latency benchmark). `git diff --check` passed. New evaluator tests cover invalid enums, semantic/structured mismatch, multilingual actions, current-information/multi-step examples, malformed output, timeout distinctions, and preservation of human history; they do not prove a model is accurate. No full app or full test suite ran.

## 11. Recommendations

**Immediate conclusion:** classification data construction needs correction before these labels can be trusted as a training/evaluation target. Preserve source labels; under a separate fix authorization, add the demonstrated file/project mappings and reconcile context requirements, then write a new candidate queue rather than overwrite human history. No such correction was implemented here.

**Human-assisted review:** inspect source utterance, advertised capabilities and complete structured fields. Manually prioritize the 46 capability-loss records and 319 false-context records; these overlap, so do not add counts. Keep uncertainty and legitimate unsupported/clarification behavior.

**Automation/replacement:** not supported. Require an independently reviewed multilingual seven-field gold set, actual trained decision-model outputs, per-field/end-to-end correctness, clarification/source-conflict tests, and a clear bridge between provider routing and semantic task routing. Schema validity and source reproduction alone are insufficient.

## 12. Limits, reproducibility and integrity

All 359 records received the Fast Classification offline audit. Production’s model-dependent results for 172 records are deliberately unavailable after the user stopped Qwen. No live seven-field trained Fast Brain classifier was identified: the encoder foundation exposes raw newly initialized heads, while the separate existing compact adapter uses the older contract. Its historical benchmarks are a different dataset/version/interface and are not represented as fresh predictions here.

Evidence/harness path: `C:\Users\Jagadheesh\AppData\Local\Temp\jarvis-full-review-benchmark-20261010-121133`. Input/output JSONL and manifest link source hashes and all preserved provenance. Temporary queue builder writes only `C:\Users\Jagadheesh\AppData\Local\Temp\jarvis-full-review-benchmark-20261010-121133\offline-queue-builder`. It does not overwrite datasets, original queue, historical evidence, human decisions or logs.

Earlier Qwen collection was explicitly interrupted by the user: 143 new HTTP inference attempts, 1 reused response, 143 completed pipeline records, 70 transport failures, 1 in-flight client interruption, 215 never attempted. Raw requests/responses already captured are retained; the interrupted record has no fabricated final proposal. This is **not a completed full Qwen benchmark**. No Ollama service was stopped or restarted; only this task’s verified temporary Python harness was terminated. No further Qwen requests were made.

Integrity: all 269 pre-existing hashed files remain byte-identical; all initial git-status entries remain. Production classifier/routing/schema/configuration, original data/history/logs and reference projects were untouched. Created repository paths: `benchmarks/fast_classifier/shadow_review_evaluation.py`, `tests/test_shadow_review_evaluation.py`, `reports/fast_brain_taxonomy_inventory.md`, plus the six report companions listed below. Existing user changes were preserved; nothing committed.

Commands: read-only git/source/data inventory; prepare/group/provenance/history/semantic inspections; earlier run_tests.py and run_benchmark.py; two read-only local health GETs; verified stop_qwen.py on this task’s harness; fast_only.py using real normalizer/temporary queue builder/pure classifier; run_fast_tests.py; report_fast_only.py and git diff --check. Failed temporary script checks/path/BOM issues are preserved in tool history; none modified production behavior.

Artifacts: `fast_brain_classification_audit.md`, `fast_brain_classification_records.jsonl`, `fast_brain_classification_records.csv`, `fast_brain_classification_metrics.json`, `fast_brain_classification_history.json`, `fast_brain_classification_manifest.json`.
