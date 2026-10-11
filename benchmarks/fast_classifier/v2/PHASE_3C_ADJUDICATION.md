# Phase 3C Annotation Adjudication

## Scope and review status

All 228 v2 records were reviewed for semantic consistency, context sufficiency,
capability availability, registry argument compatibility, provenance, and split
placement. This was an assistant review, not an independent human or
native-speaker review. The immutable v2 files were not changed. Corrections are
materialized only in `v3/dataset.jsonl`; every changed record and field is
listed in `v3/split_manifest.json` under `label_changes`.

## Definite corrections

| Example ID | Original | Phase 3C decision | Reason | Certainty |
|---|---|---|---|---|
| `en-music-001` | `single_action`, `youtube_search` | `escalate`, no capability | The request asks to play music; the advertised and real tool only opens search results. | Definite against the current registry |
| `cap-unsupported-email-001` | Escalation plus missing `recipient_address` | Escalation with no missing information | An address would not make unavailable email execution possible and should not trigger clarification. | Definite |

The `find_file` descriptor used required argument `name` in 140 records. The
real `ToolRegistry` requires `filename`. Every affected ID is recorded in the
v3 manifest. This is one repeated metadata correction, not 140 semantic
relabels.

## Ambiguous but unchanged

| Example/group | Concern | Decision |
|---|---|---|
| `en-browser-001` | Reaching a described site may require search then open. | Retained `research`; `multi_step_task` is also defensible and needs product-policy review. |
| `stt-noisy-action-001` | “calclater” could be resolved or clarified depending on ASR confidence. | Retained action because ASR confidence and alternatives are absent. |
| `synth_open_editor` | “text editor” is semantic app identification but not an explicit app alias. | Retained capability selection; argument allow-list enforcement remains outside classification. |
| `ctx-continue-resolved-001` | Context says inspect tests but advertises only `read_file`. | Retained one immediate action because the supplied context resolves a read target. |
| memory groups | Context is supplied directly; live memory retrieval could be a separate capability. | Retained `memory_recall`; provider policy remains separate. |

## Dataset findings

- The 200 v2 synthetic records came from only ten scenario groups. Classes were
  strongly correlated with topics and wording families.
- Dev had no direct-response or escalation examples.
- The sealed test has 18 records, no memory or multistep examples, and no
  language subgroup with 20 examples. It cannot support multilingual claims.
- The seven-path semantic vocabulary remains sufficient; no provider labels or
  new execution paths were added.

## Phase 3C additions

V3 adds 84 targeted synthetic records: 56 train and 28 dev. They add distinct
scenario groups for every path, native scripts, romanization, code-switching,
context resolution, unsupported actions, and underrepresented registry
capabilities. They are marked `synthetic_candidate`, provenance
`phase3c_targeted_synthetic`, and
`assistant_reviewed_pending_human_review`. No test examples were added.

