# Provider Boundary — Design Based on Current Evidence

| Task type | Direct tool | Qwen | Gemini | Clarification | Current evidence |
|---|---|---|---|---|---|
| Exact deterministic app/browser command | Candidate first | Unnecessary unless interpretation is ambiguous | Not needed | If target missing | Existing direct matcher; avoids multi-second Qwen call |
| Bounded project inspection | Not sufficient alone | Candidate primary | Escalation only if local failure/unsupported | If request lacks scope | LocalAgent loop and controlled fixture |
| Location → weather chain | Tools plus Qwen coordination | Candidate primary | Fallback only | If location/intent missing | Existing agent rules/tests |
| Context-heavy follow-up | Not sufficient alone | Candidate primary with bounded context | If local context fails | If unresolved | History passed, outcome unmeasured |
| Unsupported capability | No | No | Candidate | Maybe | Registry inventory and exposure gap |
| Current multi-source research | Browser/research tools | Candidate planner, not freshness authority | Candidate escalation | If scope/source missing | Full controlled research benchmark absent |
| “Open it” without target | No | Should not guess | Not first choice | Required | Smoke showed target invention, so explicit clarification is needed |

This is a design boundary, not production routing. The principle is to use the cheapest demonstrated capable path first and escalate only with evidence.
