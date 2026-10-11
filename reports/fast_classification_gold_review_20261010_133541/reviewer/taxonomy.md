# Frozen taxonomy for blinded independent review

Source enums and parser: contract.py:13–87,137–160; constraints: validator.py:16–38. The meanings are operational guidance, not additional enum values.

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

## All required decision fields

intent_family, execution_path, capability_family, context_requirements, clarification, uncertainty, escalation are required. clarification has required (boolean) and missing (array); escalation has candidate (boolean) and reason (enum or null). Context is a nonempty array. All enums must use exact spellings above. Gold annotations require exact keys and boolean/array types even where the runtime parser is permissive; this does not change the runtime contract.

## Cross-field constraints

none cannot coexist with another context requirement. clarification.required iff missing is nonempty. escalation.candidate iff reason is nonnull. clarify execution path and clarification intent require clarification. fast_path cannot use unknown capability. The implementation does not enforce all semantic intent/path/capability combinations; legal shape alone is insufficient.

## Operational meaning to adjudicate

conversation is dialogue; information is an explanatory/factual response. deterministic_action denotes a directly handled action, local_task a local capability task, research information gathering, memory stored-memory retrieval, multi_step_task a composed goal, clarification unresolved essential input, unsupported unavailable capability. Capability values designate domains, not tool names. file is for folder/file operations, application for launchable programs, browser for browser actions, project for project/code work. research, weather, location, reminder, memory and system name their domains; none means no capability, unknown means an unresolved domain.

Execution fast_path indicates a documented deterministic handler; respond a direct answer; qwen_agent an agent task (an enum value, not a command to run Qwen); memory/research their workflows; clarify asks for missing information; escalate flags an escalation workflow. Folder opening can use file/local_task/qwen_agent under an appropriate local-task policy; launching an app can use application. Research is distinct from simply opening a known URL; multistep research can need multi_step_task. Do not infer that any schema-valid path is executable in production.

Context requirements indicate needed context, not merely the existence of a context object. conversation/memory/project/current_information/observations/active_task identify needed sources; none excludes other entries. Empty supplied context must not automatically imply active_task. A command can depend on an unresolved referent even when context is empty. Clarification missing entries identify target/file/application/url/query/location/time/scope/permission/reference. Escalation reason names are above; actual capability availability and runtime policy must support a claim of unsupported or escalation.

## Important unresolved rubric issue

Uncertainty high/medium/low/ambiguous/outside_envelope has no calibrated thresholds in the frozen validator. The design proposes calibrated bands but does not implement their boundaries. Do not invent precision or infer the correct band from a classifier output. Reviewers/steward must agree a versioned operational rubric for uncertainty and route/context ownership before full-field labels become defensible gold. If a required field cannot be justified, leave the whole record ambiguous/unresolved; partial judgments are supporting evidence only.

No computer_control IntentFamily or local ExecutionPath exists. Do not create replacement enum values to accommodate a request. Tool names in the input capability list are supporting availability evidence, not capability_family enum values.
