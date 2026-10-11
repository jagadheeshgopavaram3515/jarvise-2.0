# Qwen Capability Matrix

| Capability | Raw Qwen | Current LocalAgent | Full Registry exists | Evidence status |
|---|---|---|---|---|
| Conversation | PARTIAL | PARTIAL | NO | Raw prompts and 1 smoke case |
| Project/code explanation | PARTIAL | YES/PARTIAL | `grep_code`, `read_file`, listing | Controlled one-tool fixture |
| Location | NO | YES/PARTIAL | YES | Existing tests/source; no live call |
| Weather | NO | YES/PARTIAL | YES | Existing tests/source; no live call |
| App open/close | NO | PARTIAL, not advertised | YES | Direct patched tests; no live action |
| File search/open | NO | PARTIAL, not advertised | YES | Existing tests/source; no live mutation |
| Browser/search | NO | PARTIAL, not advertised | YES | Source/direct tests; no live browser |
| Tool argument accuracy | NO | INSUFFICIENT EVIDENCE | Executor boundary | Tiny smoke shows invented path |
| Multi-step coordination | NO | PARTIAL | N/A | One controlled trajectory |
| Failure recovery | NO | PARTIAL | Executor returns failures | Loop exists, rate unknown |
| Clarification | NO | PARTIAL/weak | NO | Smoke target invention |
| Context follow-up | NO | INSUFFICIENT EVIDENCE | NO | History is passed, quality unmeasured |
| Multilingual/code-switch | PARTIAL unknown | INSUFFICIENT EVIDENCE | NO | No Phase D Qwen multilingual run |

## Full Registry exposure experiment

The complete Registry contains 19 tools. The current LocalAgent describes only six. A full-description mode must be benchmark-only and must not alter production. The comparison should use identical user/context/fixture cases and report whether failures move from `CAPABILITY_DESCRIPTION_FAILURE` to model reasoning/argument failures.
