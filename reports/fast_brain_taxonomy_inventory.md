# Authoritative Fast Brain taxonomy — pre-benchmark inventory

This inventory is created before record inference. Enum spelling is authoritative; explanatory meanings below are source-derived operational descriptions, not additional schema constraints.

## IntentFamily

`conversation`, `information`, `deterministic_action`, `local_task`, `research`, `memory`, `multi_step_task`, `clarification`, `unsupported`

## CapabilityFamily

`none`, `application`, `browser`, `file`, `project`, `weather`, `location`, `reminder`, `memory`, `research`, `system`, `unknown`

## ExecutionPath

`fast_path`, `respond`, `qwen_agent`, `memory`, `research`, `clarify`, `escalate`

## ContextRequirement

`conversation`, `memory`, `project`, `current_information`, `observations`, `active_task`, `none`

## MissingInformation

`target`, `file`, `application`, `url`, `query`, `location`, `time`, `scope`, `permission`, `reference`

## EscalationReason

`current_information_required`, `unsupported_capability`, `outside_qwen_envelope`, `repeated_local_failure`, `context_overload`, `explicit_cloud_request`

## Uncertainty

`high`, `medium`, `low`, `ambiguous`, `outside_envelope`

## Required decision fields and constraints

All seven top-level fields are required by FastBrainDecision.from_mapping; the Python dataclass's convenience defaults do not make serialized fields optional. clarification requires required and missing; escalation requires candidate and reason (reason may be null). Booleans are intended; current validator does not strictly enforce their JSON types. Extra keys are not rejected. All decisions are advisory, not executable routes.

Source: [contract.py](../assistant/orchestration/fast_brain/contract.py) lines 13–87 (enums), 90–160 (models/parser); [validator.py](../assistant/orchestration/fast_brain/validator.py) lines 16–38. Nonempty context list; none cannot coexist with another requirement; required clarification iff missing list nonempty; escalation candidate iff reason non-null; clarify path/clarification intent require clarification; fast_path cannot have unknown capability. No general intent/path/capability compatibility table is enforced.

## Meanings and computer actions

Intent: conversation=dialogue; information=informational response; deterministic_action=direct action category; local_task=local capability task; research=current/web information task; memory=memory retrieval/use; multi_step_task=multi-operation goal; clarification=incomplete request; unsupported=outside supported capabilities.

Execution: fast_path=deterministic handling; respond=direct response; qwen_agent=agent task; memory=memory workflow; research=research workflow; clarify=clarification; escalate=escalation. These are advisory enums; production's LOCAL/TASK/GEMINI categories are a different interface.

Capability names designate domains: none=no capability; application=application launch/control; browser=browser operations; file=file operations; project=project/code inspection; weather/location/reminder/memory/research/system=their named domains; unknown=unresolved capability.

Context names designate conversation/memory/project/current_information/observations/active_task dependencies; none means no such dependency. Missing-information names identify the required target/file/application/url/query/location/time/scope/permission/reference. Escalation reasons identify current-information need, unsupported capability, outside local envelope, repeated failure, context overload or explicit cloud request. Uncertainty high/medium/low/ambiguous/outside_envelope is a descriptive enum, not measured confidence or accuracy; source does not calibrate it.

Actual builder mappings ([build_fast_brain_dataset.py](../benchmarks/fast_classifier/build_fast_brain_dataset.py) lines 38–97; [queue builder](../benchmarks/fast_classifier/review_fast_brain_dataset.py) lines 39–49): open/close_app or application category → application, usually local_task/qwen_agent; find/read/open_file/list_directory → file (project when categorized project/code), usually local_task/qwen_agent; web search/open_url/read_page → browser or research, research/research when research categorized; multi_step path → multi_step_task/qwen_agent; conversation → conversation/respond. open_folder is not specifically recognized by capability mapper: lacking a file category, it can fall through to none/information/respond. That is a data-construction limitation, not permission to invent computer_control. A supported open-folder interpretation uses file with local_task/qwen_agent (or deterministic_action/fast_path for a genuinely deterministic handler); exact choice needs evidence about execution context.

[Deterministic baseline](../assistant/orchestration/fast_brain/deterministic.py) lines 22–48 recognizes stop/time/date/open Chrome/YouTube as deterministic_action/fast_path; defaults other utterances to information/respond. It is not the provenance of all queue annotations.

Current-information retrieval can use research/research/research with current_information context; explicit multi-step research can use multi_step_task/qwen_agent with browser/research capability depending supported operations. Schema permits both capability families and does not enforce a unique combination. File-versus-application distinction matters for opening folders versus launchable applications.

## Reviewer and persistence

[Reviewer](../benchmarks/fast_classifier/review_fast_brain.py): AI_MODES at line 20; proposal validation 126–169; request/parser/policy 179–225; prompt 228 onwards. AUTO_CONFIDENT and AI_CORRECTION require complete valid labels; HUMAN_REVIEW/SOURCE_CONFLICT require null label and human attention. Equality versus correction is not mechanically enforced. Confident labels requiring clarification are downgraded to HUMAN_REVIEW. Unsupported/escalation candidates retain human review. Source-conflict policy checks limited explicit source intent mappings and legacy aliases. Human decisions append only after explicit input; the benchmark does not invoke that persistence path.

Decision fields are the seven-concept label. Utterance, language/script/response language, provenance, ids/groups/split, priority/reasons, model confidence/rationale/source_evidence, timestamps and review status are descriptive/evaluation metadata. Historical original expected labels use a different schema and must not be validated as frozen seven-field decisions or called live predictions.

## Valid examples

Captured fixtures in [regression data](../tests/fixtures/shadow_review_regression.json) provide research/research/research with no clarification, local_task/qwen_agent/application for Open Notepad, and clarification/clarify/application with missing target for Open it. [Focused tests](../tests/test_shadow_review_regression.py) demonstrate full schema validation, malformed outputs, corrections, explicit conflicts and retained human approval.
