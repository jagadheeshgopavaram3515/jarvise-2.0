# Base Qwen vs Qwen+LoRA Evaluation Plan

## Frozen task set

Build a versioned, hand-reviewed set using actual Registry capabilities: conversation; open/search browser; “do that again” context resolution; project classifier inspection with `grep_code`/`read_file`; current-information research with browser tools; location→weather chaining; explicit-city weather; file search/open; malformed/missing target clarification; tool timeout/failure recovery; and unsupported capability escalation. Include English, Telugu, Hindi, Romanized forms, and code switching.

Use deterministic fixtures/mocks for tools and web results. Do not call arbitrary live tools for the core score. Keep a separate live smoke set.

## Measurable metrics

Report per-example and aggregate: task completion; correct capability/tool; schema-valid arguments; argument exactness; multi-step completion; recovery after injected failure; clarification precision/recall; context-reference resolution; multilingual task success; final-response rubric score; time to first token and completion latency; peak RAM/VRAM; Qwen request count; and escalation frequency plus escalation appropriateness.

Tool success means the expected registered tool and arguments were selected and the fixture result leads to the expected next state. Final response quality is scored against a rubric (grounded, concise, language-appropriate, no invented action), with two reviewers or an explicit deterministic rubric—not an unmeasurable “looks better” claim.

## Protocol

Run base and adapter with identical prompt envelope, temperature, token caps, seeds where supported, fixtures, and machine state. Repeat latency runs enough to report median and p95. Separate Fast Brain-only accuracy from end-to-end Qwen trajectory success. Record failures by class: model output, validation, tool, timeout, context, or infrastructure.

## Gates before production proposal

No integration proposal until the adapter improves or preserves task success and valid arguments, does not materially worsen clarification or multilingual behavior, stays within memory limits, and has a reproducible serving path. A higher success score with unsafe or ungrounded tool arguments is a failure, not an improvement.
