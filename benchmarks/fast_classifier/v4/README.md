# FastClassifier V4 comparison dataset

This dataset supports an offline, same-input comparison of the Phase 3C BERT
candidate, Laya, and the existing Qwen classifier. It is not runtime
configuration and no capability is executed by the benchmark.

## Layout

- `dataset.jsonl`: schema `4.0` examples and gold decision candidates.
- `split_manifest.json`: hashes, group membership, and frozen split assignments.
- `ANNOTATION_GUIDE.md`: decision rules and review workflow.
- `DATASET_CARD.md`: composition, provenance, and limitations.

Generate deterministically with:

```powershell
python benchmarks/fast_classifier/prepare_v4.py
```

The 140-row test split is sealed for Phase 4. Do not use its labels for
training, model selection, prompt tuning, or threshold tuning.

