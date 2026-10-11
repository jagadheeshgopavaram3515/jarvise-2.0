# Dataset Status

## Existing cases reused

The Phase B smoke dataset contains:

- `conversation_001`: “Hello Jarvis”
- `clarify_001`: “Open it” with no target

Both are synthetic. The Phase C controlled fixture evidence adds project inspection, tool selection, and failure-loop observations, but it is not a versioned large dataset.

## Required future case schema

Each case should include `id`, `group`, language, script, category, complexity level, user input, bounded context, capabilities, expected intent/capability/tool sequence, clarification/escalation labels, difficulty, context size, review status, and candidate providers. Expected fields remain evaluator-only.

## Coverage gap

The requested 250–400 cases, multilingual human review, grouped train/dev/sealed splits, all 19 capability families, 1–5+ tool trajectories, and repeated recovery cases do not yet exist. This phase does not fabricate them. Any claim about broad multilingual reliability, maximum useful tool steps, or provider accuracy is therefore `INSUFFICIENT EVIDENCE`.
