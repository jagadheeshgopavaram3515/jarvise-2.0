# JARVIS Brain — Slot Contract Forensic Analysis and Alignment Report

**Date:** 2026-10-10  
**Repository:** `D:\JARVIS`  
**Working Directory:** `D:\JARVIS\reports\jarvis_slot_contract_fix_20261010_211600`  
**Role:** Senior ML Systems Engineer / Production ML Architect  

---

## Executive Summary & Core Verdict

This investigation examined the hypothesis: **whether inconsistencies between capability argument schemas, `missing_information` fields, and clarification decisions are responsible for JARVIS Brain's remaining classification errors.**

### Core Forensic Verdict: **Hypothesis Disproved for the Evaluation Corpora**
Across both frozen test corpora (47 errors in Frozen 100, 11 errors in Independent 24, totaling 58 decision mismatches):
- **0 out of 58 errors (0.0%) were caused by slot contract naming mismatches** (e.g. predicting generic `target` when the test expected concrete `path`, `query`, or `folder_path`).
- In both evaluation corpora (`unseen_evaluation_corpus_100.jsonl` and `unseen_evaluation_corpus_v4.jsonl`), the ground truth `expected` labels **exclusively expect `name: "target"`** for 100% of clarification examples (`{'target': 16}` in Frozen 100, `{'target': 4}` in Independent 24).
- The baseline v5 adapter already emits `name: "target"` when clarification is predicted, matching the benchmark ground truth schema exactly.

### Architectural Reality & Separation of Concerns
While slot contract naming does **not** explain the benchmark errors, an architectural schema divergence between system layers was verified:
1. **ToolRegistry (`assistant/tools/registry.py`)**: Defines parameter names `app`, `filename`, `path`, `url`, `query`.
2. **Classifier Evaluation Corpora**: Advertises capability schemas with parameter names `name`, `query`, `path`, `url`.
3. **FastBrain (`fast_brain/contract.py`)**: Uses an abstract `MissingInformation` enum (`TARGET`, `FILE`, `APPLICATION`, `URL`, `QUERY`).
4. **Compact BERT Heads (`compact_heads.joblib`)**: Was trained only on classes `['__none__', 'prior_request', 'target']`.

To solve this architectural inconsistency without degrading benchmark performance or modifying model weights, we designed and implemented a **Canonical Argument Contract and Slot Contract Resolver** (`slot_contract_resolver.py` and `experimental_adapter_v6.py`). It enforces schema validation, capability admissibility, and contract consistency, passing **14/14 regression tests (100%)** with **zero regressions** across both corpora.

---

## 1. Baseline Verification & Hash Manifest

All baseline artifacts, models, and evaluation corpora were verified before analysis:

| Artifact | Path | SHA-256 |
|---|---|---|
| Model Artifact | `reports/classifier_v4_experiment_20261010_175140/best_candidate/compact_heads.joblib` | `6eb693419cdc4a1de54a9ea5f69ef07a41f0eaa1b48b30673a960a04a01f8ace` |
| v5 Adapter | `reports/jarvis_brain_forensic_improvement_20261010_183000/experimental_adapter_v5.py` | `44d2a950d1117c9a26c224ebe1ca05751158b165cb3ea301d0477be4606d4d7b` |
| Frozen 100 Corpus | `reports/classifier_v4_frozen_100_eval_20261010_180014/unseen_evaluation_corpus_100.jsonl` | `206118b07a501d97cabbee92c2ee8dda2b1f4db020ad59ecf64dd4d7fe19c59f` |
| Independent 24 Corpus | `reports/classifier_v4_experiment_20261010_175140/unseen_evaluation_corpus_v4.jsonl` | `ebd2128c3ece5a638b24d1eb34ec50d9b594f2f8d5e90071aae089429899044a` |

### Baseline Reproduction
- **Frozen 100 Corpus:** 53 / 100 (53.0%) strict exact match, 77 / 100 (77.0%) capability top-1.
- **Independent 24 Corpus:** 13 / 24 (54.2%) strict exact match, 20 / 24 (83.3%) capability top-1.
Reproduction matches historical results with 100.0% precision.

---

## 2. Forensic Failure Root-Cause Analysis (58 Mismatches Traced)

Every single mismatch across Frozen 100 (47 records) and Independent 24 (11 records) was traced through:
`classifier output → adapter → capability definition → argument schema → missing-information mapping → clarification decision`

### Aggregate Error Categorization

| Category | Count | Pct (%) | Primary Responsible Component | Forensic Diagnosis |
|---|---|---|---|---|
| **Reference Resolution / Annotation Artifact** | 13 | 22.4% | Benchmark ground truth annotation | The synthetic corpus generator hardcoded `reference: 'it'` across all clarification requests, failing whenever the user said "that", "use", "isse", "idi", or "athai", where the NLP extractor correctly extracted the actual pronoun. |
| **Capability Ranking Error** | 12 | 20.7% | Capability similarity ranker | Semantic confusion in BERT dense space between closely paired candidates (e.g. `delete_file` vs `delete_folder`, `open_file` vs `open_folder`) on non-English utterances. |
| **Semantic Misclassification (False Abstention)** | 8 | 13.8% | Execution path / calibration | The model triggered clarification or escalation on legitimate actions due to low confidence on non-standard phrasing (e.g., "VS Code application open pannu"). |
| **False Escalation** | 7 | 12.1% | Escalation head / domain whitelist | Short colloquial Indic requests without Latin entity names (e.g., "దీన్ని వెంటనే డిలీట్ చెయ్యి", "इसे हटा दो") were escalated because they lacked recognized domain keywords. |
| **Path / Intent Misclassification** | 7 | 12.1% | Multi-step / Direct response head | Confusion between conversational responses vs multi-step actions or direct answers. |
| **Semantic Misclassification (False Action)** | 6 | 10.3% | Capability ranker / threshold | Model committed to an action (e.g., `open_folder`) on ambiguous utterances (e.g., "वह तुरंत खोलो", "అది ఓపెన్ చెయ్యి") where clarification was expected. |
| **False Clarification** | 2 | 3.4% | Clarification head | Conversational utterances wrongly triggering clarification (e.g. "Namaste JARVIS, aap kaise ho?"). |
| **Other Discrepancies** | 3 | 5.2% | Mode / Intent composition | Minor intent format variations. |
| **Slot Contract / Argument Schema Mismatch** | **0** | **0.0%** | **N/A** | **Zero occurrences.** |

### Key Evidence
1. **When clarification was expected and predicted**, `missing_information` had **0 mismatches**. Expected was `[{"name": "target", "description": "The requested target is not identifiable"}]`, and predicted was identical.
2. In all 12 cases where `missing_information` was recorded as mismatched (7 in Frozen 100, 5 in Independent 24), the failure was actually caused by an **upstream path decision failure** (e.g. model predicted `single_action` or `escalate` instead of `clarify`, so `missing_information` was empty `[]`).

---

## 3. Canonical Argument Contract Architecture

Although slot contract naming was not the source of benchmark errors, aligning argument contracts across JARVIS is essential for runtime execution reliability.

We designed a canonical argument contract in `slot_contract_resolver.py`:

### Authoritative Capability Schema Registry (`CANONICAL_CAPABILITY_SCHEMAS`)
```python
CANONICAL_CAPABILITY_SCHEMAS = {
    "open_app": CanonicalCapabilitySchema(
        name="open_app",
        required_arguments=("app",),
        optional_arguments=(),
        argument_aliases={"name": "app"},
        clarification_prompts={"app": "Which application would you like to open?"},
    ),
    "open_folder": CanonicalCapabilitySchema(
        name="open_folder",
        required_arguments=("path",),
        optional_arguments=(),
        argument_aliases={"folder_path": "path"},
        clarification_prompts={"path": "Which folder path would you like to open?"},
    ),
    "find_file": CanonicalCapabilitySchema(
        name="find_file",
        required_arguments=("filename",),
        optional_arguments=(),
        argument_aliases={"query": "filename"},
        clarification_prompts={"filename": "What file name or pattern are you looking for?"},
    ),
    "read_file": CanonicalCapabilitySchema(
        name="read_file",
        required_arguments=("path",),
        optional_arguments=("offset", "limit"),
        argument_aliases={"file_path": "path"},
        clarification_prompts={"path": "Which file should I read?"},
    ),
    "read_page": CanonicalCapabilitySchema(
        name="read_page",
        required_arguments=("url",),
        optional_arguments=(),
        argument_aliases={"web_url": "url"},
        clarification_prompts={"url": "Which webpage URL should I read?"},
    ),
    # ... additional desktop tools
}
```

### Canonical Rules Enforced by `SlotContractResolver`:
1. **Admissibility Enforcement:** If the classifier selects a capability that is not advertised in `request.capabilities`, it safely clears `selected_capability` and escalates to prevent unauthorized tool execution.
2. **Clarification Completeness:** If `execution_path == 'clarify'` or `needs_clarification == True`, `missing_information` is guaranteed to contain at least one valid item, separating machine-readable identifiers (`name`) from human-readable messages (`description`).
3. **Action Consistency:** If `execution_path` is an action path, `missing_information` is strictly cleared (`[]`), preventing invalid intermediate states.
4. **Escalation Preservation:** `needs_escalation == True` strictly suppresses clarification and capability selection.
5. **Context References Preservation:** Deictic and anaphoric references are preserved without corruption across Indic and Latin scripts.

---

## 4. Regression Test Suite

Created `test_slot_contract_alignment.py` covering all contract requirements:

```
collected 14 items
TestCanonicalArgumentContract::test_canonical_schemas_defined PASSED          [ 7%]
TestCanonicalArgumentContract::test_distinguishes_required_from_optional PASSED [14%]
TestCanonicalArgumentContract::test_validate_valid_argument_names PASSED       [21%]
TestCanonicalArgumentContract::test_validate_invalid_argument_names PASSED     [28%]
TestCanonicalArgumentContract::test_unknown_capability_rejection PASSED         [35%]
TestSlotContractResolver::test_clarification_consistency_with_missing_info PASSED [42%]
TestSlotContractResolver::test_clarification_consistency_false_when_empty PASSED [50%]
TestSlotContractResolver::test_multiple_missing_arguments_representation PASSED [57%]
TestSlotContractResolver::test_machine_readable_vs_human_readable_separation PASSED [64%]
TestSlotContractResolver::test_context_references_preserved PASSED             [71%]
TestSlotContractResolver::test_escalation_preserved PASSED                     [78%]
TestSlotContractResolver::test_multilingual_input_stability PASSED             [85%]
TestSlotContractResolver::test_multi_step_request_contract PASSED              [92%]
TestSlotContractResolver::test_safe_rejection_of_unsupported_capability PASSED [100%]
============================== 14 passed in 4.10s ==============================
```

---

## 5. Controlled Comparative Evaluation (v5 Baseline vs v6 Adapter)

Evaluated on identical inputs across both unchanged evaluation corpora:

### Summary Comparison Table

| Metric | Frozen 100 (v5 Baseline) | Frozen 100 (v6 Experimental) | Independent 24 (v5 Baseline) | Independent 24 (v6 Experimental) |
|---|---|---|---|---|
| **Strict 8-Field Exact Match** | **53 / 100 (53.0%)** | **53 / 100 (53.0%)** | **13 / 24 (54.2%)** | **13 / 24 (54.2%)** |
| **Capability Top-1 Match** | **77 / 100 (77.0%)** | **77 / 100 (77.0%)** | **20 / 24 (83.3%)** | **20 / 24 (83.3%)** |
| **Contract Validation Rate** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **Regressions Introduced** | — | **0 (0.0%)** | — | **0 (0.0%)** |
| **Fixes on Benchmarks** | — | **0 (0.0%)** | — | **0 (0.0%)** |
| **Unchanged Decisions** | — | **100 / 100 (100.0%)** | — | **24 / 24 (100.0%)** |
| **Mean Inference Latency** | 141.8 ms | 132.2 ms | 118.5 ms | 111.1 ms |
| **P95 Latency** | 201.5 ms | 185.8 ms | 172.4 ms | 166.6 ms |

### Per-Field Match Rates (Frozen 100)

| Field | v5 Baseline | v6 Experimental | Status |
|---|---|---|---|
| `intent` | 67.0% | 67.0% | Identical |
| `execution_path` | 82.0% | 82.0% | Identical |
| `request_modes` | 93.0% | 93.0% | Identical |
| `selected_capability` | 77.0% | 77.0% | Identical |
| `needs_clarification` | 93.0% | 93.0% | Identical |
| `needs_escalation` | 87.0% | 87.0% | Identical |
| `missing_information` | 93.0% | 93.0% | Identical |
| `context_references` | 81.0% | 81.0% | Identical |

### Clarification Confusion Matrix (Frozen 100)
- **True Positives (TP):** 11 / 16 (68.8% sensitivity)
- **False Positives (FP):** 2 (conversational inputs mistaken for clarification)
- **True Negatives (TN):** 82 / 84 (97.6% specificity)
- **False Negatives (FN):** 5 (ambiguous Indic requests committing to action or escalating)

### Performance by Scenario Group (Frozen 100)

| Scenario Group | Records | Exact Match | Capability Match |
|---|---|---|---|
| `corpus100-folder` | 12 | 10 / 12 (83.3%) | 11 / 12 (91.7%) |
| `corpus100-memory` | 12 | 11 / 12 (91.7%) | 12 / 12 (100.0%) |
| `corpus100-direct` | 10 | 7 / 10 (70.0%) | 10 / 10 (100.0%) |
| `corpus100-app` | 10 | 7 / 10 (70.0%) | 8 / 10 (80.0%) |
| `corpus100-file` | 12 | 6 / 12 (50.0%) | 6 / 12 (50.0%) |
| `corpus100-search` | 12 | 5 / 12 (41.7%) | 5 / 12 (41.7%) |
| `corpus100-webpage` | 8 | 4 / 8 (50.0%) | 8 / 8 (100.0%) |
| `corpus100-escalate` | 6 | 2 / 6 (33.3%) | 2 / 6 (33.3%) |
| `corpus100-clarify` | 12 | 1 / 12 (8.3%) | 10 / 12 (83.3%) |
| `corpus100-multistep` | 6 | 0 / 6 (0.0%) | 5 / 6 (83.3%) |

### Performance by Script (Frozen 100)
- **Latin Script (68 records):** 39 / 68 (57.4%) exact, 59 / 68 (86.8%) capability top-1.
- **Telugu Script (12 records):** 6 / 12 (50.0%) exact, 8 / 12 (66.7%) capability top-1.
- **Devanagari Script (15 records):** 7 / 15 (46.7%) exact, 8 / 15 (53.3%) capability top-1.
- **Tamil Script (5 records):** 1 / 5 (20.0%) exact, 2 / 5 (40.0%) capability top-1.

---

## 6. Real Bottlenecks and Recommendations

Having definitively ruled out slot contract naming as the source of errors, the forensic audit identifies the three true remaining bottlenecks:

1. **Synthetic Annotation Artifacts in Context References (22.4% of errors):**
   - The test generator for `corpus100-clarify` systematically hardcoded `expected['context_references'] = [{'kind': 'item', 'reference': 'it', 'resolved_value': None}]`.
   - In realistic NLP evaluation, when the user says "Delete that", "use turant open karo", or "athai marubadiyum open pannu", extracting `"that"`, `"use"`, or `"athai"` is semantically correct.
   - Aligning reference evaluation with linguistic ground truth will resolve 13/58 mismatches without any model tuning.

2. **Native Indic Dense Space Alignment (20.7% of errors):**
   - Compact multilingual BERT produces slightly lower cosine similarity for native script verb phrases compared to Latin/English.
   - Fine-tuning the lightweight linear projection head on parallel Indic action pairs (rather than rule-based transliteration or slot remapping) is the justified path forward.

3. **Short Indic Utterance Escalation Thresholding (12.1% of errors):**
   - Short Indic action requests (e.g. "దీన్ని వెంటనే డిలీట్ చెయ్యి") lack Latin keywords in `APP_TERMS` or `PAGE_TERMS`, triggering fallback escalation.
   - Expanding the root verb lexicon for Hindi (`हटाओ`, `खोलो`), Telugu (`డిలీట్`, `తెరవు`), and Tamil (`அழி`, `திற`) directly in the domain token validator will prevent false escalation.

---

## 7. Safety and Hygiene Verification

- **Production Code Safety:** Verified via `git status --porcelain assistant/`. Zero modifications made to production files.
- **Artifact Preservation:** All v4, v5, and transliteration experiment reports and artifacts remain unchanged.
- **Corpora Integrity:** `unseen_evaluation_corpus_100.jsonl` and `unseen_evaluation_corpus_v4.jsonl` preserved byte-for-byte with verified SHA-256 hashes.
- **No Weight Retraining:** Model weights were not modified.
- **No Utterance-Specific Overrides:** No request-specific hacks or rules were introduced.

