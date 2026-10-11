# V4 annotation guide

## Semantic labels

- `direct_response`: ordinary conversation or stable knowledge without a tool.
- `memory_recall`: answer requires prior conversation or supplied memory.
- `research`: current/external information can be obtained with one capability.
- `single_action`: one advertised capability completes the request.
- `multi_step_task`: multiple retrieval/action steps or subsequent synthesis.
- `clarify`: an essential argument or reference is unresolved.
- `escalate`: no advertised capability can perform the requested operation.

Provider names such as Qwen and Gemini are not semantic labels. A selected
capability must be present in the example's advertised capability descriptors.
Clarification requires a concrete `missing_information` entry. Escalation means
unavailable capability, not merely missing arguments.

## Review workflow

1. Review the utterance without seeing model predictions.
2. Verify naturalness and meaning, especially native-script and romanized text.
3. Verify context sufficiency and the advertised capability descriptions.
4. Review every expected field, then set `annotation_status` and
   `review_status` to `reviewed`; use `ambiguous` when no single decision is
   defensible and record accepted alternatives in notes.
5. A second reviewer resolves disagreements and marks the row `adjudicated`.

Current status: all v4 records are `pending_review`. Assistant authoring is not
human or native-speaker validation.

