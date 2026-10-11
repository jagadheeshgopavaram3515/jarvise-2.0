# FastClassifier Dataset v2 Adjudication

Version: 2.0

This report records every intentional change from immutable
`v1/seed.jsonl`. The original IDs are retained. All v2 adjudications were made
during Phase 3B by the coding assistant and remain **pending human review**;
they are not represented as human-reviewed ground truth.

## Global decisions

| Change | v1 | v2 | Reason |
|---|---|---|---|
| Execution semantics | Provider routes `LOCAL`/`GEMINI`/`TASK` | `direct_response`, `memory_recall`, `research`, `single_action`, `multi_step_task`, `clarify`, `escalate` | Separate semantic execution from provider policy |
| Intent labels | Scenario-specific labels | `respond`, `recall`, `research`, `use:<capability>`, `clarify_request`, `unsupported_request` | Keep intent compositional and avoid an expanding phrase taxonomy |
| Split assignment | Row-level dev/test | Scenario-group dev/test | Keep translations, paraphrases, and related contexts together |
| Review metadata | `reviewed` | `synthetic_candidate`; `review_status=pending_human_review` | Semantic label migration has not been approved by a human reviewer |

Every v1 row receives only those global transformations unless listed below.
Traceability is recorded by `derived_from_id`, `group_id`, `provenance`, and the
source-v1 hash in `split_manifest.json`.

## Scenario-specific decisions

| ID | Decision | Evidence/rationale | Remaining uncertainty |
|---|---|---|---|
| `en-browser-001` | Advertise `google_search` before `open_url`; path `research`; selected capability `google_search` | `open_url` requires a URL but the utterance gives only a site description | A future resolver could know canonical sites without search |
| `en-music-001` | Normalize intent to `find:focus_music`, then v2 compositional `use:youtube_search` | Advertised capability can search but cannot guarantee playback | Playback needs a distinct capability |
| `stt-noisy-action-001` | Keep intended calculator action and group with calculator variants | v1 note explicitly records the manually reviewed transcription interpretation | ASR confidence/alternatives are absent; clarification may also be valid |
| `cap-unsupported-email-001` | Keep escalation and add missing `recipient_address` | Email is unavailable and an executable address is absent | Product policy may ask for address only after email capability exists |
| Memory examples | Path becomes `memory_recall` | Memory need is semantic; GEMINI was a provider-policy label | Future memory retrieval ownership remains an orchestration choice |
| Desktop/browser examples | Paths become `single_action`, `research`, or `multi_step_task` | Labels describe desired behavior against advertised capabilities | Current LocalAgent does not advertise every registry tool |

## Added examples

Two hundred examples were generated from ten scenario templates, each with five
phrases in four language/script variants. They are grouped by scenario and are
assigned only to train or development splits. Their provenance is
`template_generated`, annotation status is `synthetic_candidate`, and review
status is `pending_human_review`.

They are suitable for exercising an engineering prototype, not for claiming
general multilingual accuracy. No synthetic example appears in the sealed test
split.
