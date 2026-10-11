# FastClassifier V4 dataset card

## Purpose

Offline comparison of three decision approaches under a common JARVIS-specific
contract. This dataset must not be used to claim production readiness.

## Composition

- Total: 434 examples.
- Train: 216 retained from v3.
- Development: 78 retained from v3.
- Sealed test: 140 newly authored examples in 20 semantic scenario groups.
- Test slices: 20 each for English, Telugu native script, Hindi Devanagari,
  romanized Telugu, romanized Hindi, English/Telugu code-switching, and
  English/Hindi code-switching.

The test covers all seven execution paths and actual registry capability names
for location, weather, desktop applications, approved file operations, project
inspection, browser/search, and YouTube search. Reminder and email requests are
escalation cases because the current default registry exposes neither action.

## Provenance and review

Train/dev are derived from v3 and preserve their group assignments. The v4 test
was manually authored as synthetic benchmark candidates and was not generated
from model predictions. Every row is marked `pending_review`; zero examples
have independent human or native-speaker review. Results are therefore
diagnostic and must not be used to declare a winner.

## Leakage controls

All language variants of a semantic scenario share one `group_id` and one
split. No v3 sealed-test example was copied into the v4 test. The manifest
records source/dataset hashes and complete group assignments. The sealed test
must not be used for training, prompt selection, thresholds, or model choice.

## Known limitations

- Gold labels and multilingual wording require independent review.
- The seven variants per test scenario are correlated, so 140 rows are not 140
  independent semantic situations.
- Romanization varies widely in real speech and is only sparsely represented.
- This is a transcript benchmark; it does not measure Google STT or audio.
- Current-resource requests can depend on runtime policy even when the semantic
  decision is clear.

