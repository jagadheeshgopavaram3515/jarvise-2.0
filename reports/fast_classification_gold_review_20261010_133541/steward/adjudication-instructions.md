# Steward workflow

Do not share this directory or old reports/predictions before initial judgments. Keep original-labels.jsonl as delayed supporting evidence only. The file split is procedural, not an ACL. Verify human identities/independence and language competence; the code checks attestations and completeness only.

Freeze field semantics (especially uncertainty bands), retain both initial reviews and all disagreements, and use a separate qualified adjudicator when they disagree. Preserve self-review separately. Never overwrite an original queue or decision log. Hash accepted references before revealing predictions.

After reference labels qualify, run from D:\JARVIS:

```powershell
python -B -m benchmarks.fast_classifier.gold_review --reviews <new-reviewed-jsonl> --manifest D:\JARVIS\reports\fast_classification_gold_review_20261010_133541\steward\manifest.json --out <new-benchmark-output-directory>
```

The command refuses an existing output directory, verifies input/rubric hashes, runs only eligible records, and uses the real deterministic baseline with no network/model. Failed/missing/invalid predictions count as incorrect in each eligible denominator. Ambiguous/unresolved/self/AI/historical references are excluded and listed. The initial blank bundle has N=0 and must remain NOT_MEASURED. Do not advertise one qualified single-review label as consensus; retain reviewer count and qualifications.

Future reporting: exact seven-field/nested match; nine per-dimension counts; confusion matrices for enum/boolean fields and exact-set-list diagnostics for context/missing; macro and per-class precision/recall/F1 with zero-support classes disclosed; language/script, intent/capability, known-error and ordinary subgroup sizes. A full census gives descriptive correctness on this fixed set, not sampling generalization. Correlated translations require group-aware uncertainty analyses; do not attach naive iid Wilson intervals to real-world claims. Separately report historical consistency, classifier gold accuracy and Qwen-review quality; Qwen remains out of scope.
