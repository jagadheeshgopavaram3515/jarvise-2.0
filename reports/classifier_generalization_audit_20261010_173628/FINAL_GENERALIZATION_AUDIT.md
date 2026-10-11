# JARVIS COMPACT BERT v3 — FINAL GENERALIZATION AUDIT

**Audit Date:** 2026-10-10  
**Audit Directory:** `D:\JARVIS\reports\classifier_generalization_audit_20261010_173628\`  
**Target Evaluation:** `D:\JARVIS\reports\classifier_next_experiment_20261010_171102\`  
**Dataset Inspected:** `provisional_validation.jsonl` (40 fixed records) & `provisional_train.jsonl` (232 records)  
**Audit Verdict:** **REPRODUCED BUT GENERALIZATION UNPROVEN**

---

## 1. Executive Verdict & Core Findings

### **VERDICT: REPRODUCED BUT GENERALIZATION UNPROVEN**

The reported **38 / 40 (95.0%) strict exact agreement** in `classifier_next_experiment_20261010_171102` is **fully reproducible** mechanically on the unchanged 40-record validation dataset using the frozen multilingual BERT encoder, saved candidate heads, and experimental adapter. All 23 unit and regression tests pass, and live encoder latency is confirmed at **170.6 ms median inference** (174.2 ms total pipeline).

However, **generalization to unseen distribution remains unproven**. The dramatic leap from 10/40 (25%) to 38/40 (95%) was achieved through a combination of:
1. Genuine architectural fixes (dense cosine similarity, argument feasibility checks, typed context resolution, deterministic schema rendering).
2. **Post-hoc heuristics tailored to the 40 validation examples**, including specific keyword matching for memory phrases (`'database migration'`, `'architecture'`), context-shape heuristics (`'decision' in projects` $\rightarrow$ `'prior discussion'`), language-specific action words (`'kholo'`, `'bring'`) to resolve missing target slots, and an ambiguous-tool heuristic specifically targeting `open_file` vs `open_folder`.
3. **Severe scenario group imbalance**: Exactly **20 of the 40 validation records (50.0%)** belong to a **single synthetic scenario group** (`synth_open_archive_folder`), creating an optimistically inflated evaluation metric.

---

## 2. Claim-by-Claim Forensic Verification

### Claim 1: Strict Exact Reproduction (38/40)
- **Status: VERIFIED (REPRODUCED)**
- **Audit Methodology:** Evaluated saved predictions in `best_metrics.json` against `provisional_validation.jsonl` using the exact frozen eight-field comparison contract:
  `FIELDS = ('intent', 'execution_path', 'request_modes', 'selected_capability', 'needs_clarification', 'needs_escalation', 'missing_information', 'context_references')`.
- **Result:**
  - Strict Exact Matches: **38 / 40 (95.0%)**
  - Mismatches: Exactly 2 records:
    1. `ctx-hologram-001`: expected `['memory', 'conversational']`, got `['memory']` (diff in `request_modes`).
    2. `en-music-001`: expected `escalate` / `None`, got `single_action` / `youtube_search` (diff in `execution_path`, `intent`, `selected_capability`, `needs_escalation`).
  - Scored fields, denominators (40), fallback decisions, and frozenset serialization rules match the baseline evaluation.

---

### Claim 2: Tracing Rules in the Structured Capability Ranker & Adapter
- **Status: OPTIMISTIC BIAS CONFIRMED**
- **Forensic Inspection of `experimental_adapter_v3.py`:**
  
  1. **Archive Folder Entity Weights:**
     ```python
     folder_terms = {'folder', 'directory', 'archive', 'tar', 'zip', 'extract', 'backup', 'folders'}
     file_terms = {'file', 'files', 'document', 'recent', 'code'}
     if any(t in q_tokens for t in folder_terms):
         if 'folder' in cap_name or 'directory' in cap_name:
             score += 0.20
         elif 'file' in cap_name:
             score -= 0.15
     ```
     *Finding:* While conceptually sound as an entity constraint, adding `'archive'` and tuning the $+0.20$ bonus was directly motivated by analyzing the 20 `synth_open_archive_folder` validation errors.

  2. **Ambiguous Tool Filter:**
     ```python
     if 'open_file' in cap_names and 'open_folder' in cap_names:
         if not (q_tokens & folder_terms) and not (q_tokens & file_terms):
             return None, 0.30
     ```
     *Finding:* This rule specifically catches `dev_clarify_unresolved` records where the offered capability candidate list is exactly `['open_file', 'open_folder']`.

  3. **Target Slot Inference Heuristic:**
     ```python
     if name == '__none__' or any(w in request.user_text.lower() for w in ['it', 'kholo', 'open', 'bring', 'target']):
         name = 'target'
     ```
     *Finding:* The words `'kholo'` (Hindi), `'bring'` (English), and `'open'` were added directly to override the learned classifier head for `dev_clarify_unresolved-hi-Latn` (*"use phir se kholo"*) and `dev_clarify_unresolved-en` (*"Bring it up again."*).

  4. **Memorized Reference Span Rules:**
     ```python
     elif kind == 'prior_conversation':
         if any('decision' in p for p in request.context.get('projects', [])):
             span = 'prior discussion'
         elif 'database migration' in text_lower:
             span = 'the database migration last week'
         elif 'architecture' in text_lower:
             span = 'Kal architecture ke baare mein'
     ```
     *Finding:* This is direct dataset memorization. The strings `'database migration'` and `'architecture'` explicitly match validation records `en-memory-001` and `hi-codeswitch-memory-001`. The context heuristic `any('decision' in p for p in projects)` specifically produces the arbitrary annotation string `'prior discussion'` for the four `dev_memory_project-*` records.

---

### Claim 3: Train / Validation Independence Audit
- **Status: PARTIALLY INDEPENDENT (Proven Text/Group Split, High Template Similarity)**
- **Audit Findings:**
  - **Scenario Group Overlap:** **0 groups**. Train has 27 scenario groups; Validation has 7 scenario groups. None overlap.
  - **Normalized Utterance Overlap:** **0 strings**. No validation query appears verbatim in train.
  - **Protected Test Set Leakage:** **0 records**. None of the 40 validation records are in `protected_test_manifest.json`.
  - **Capability Imbalance:**
    - Train distribution: `google_search` (44), `find_file` (22), `open_app` (20), `close_app` (8), `open_url` (5), `open_folder` (5), `None` (128).
    - Validation distribution: `open_folder` (20), `google_search` (4), `None` (16).
  - *Uncertainty:* Although group IDs and string hashes do not cross the boundary, the semantic templates (`synth_open_archive_folder-*`) were synthetically generated using a narrow set of slot substitutions across languages.

---

### Claim 4: Leave-One-Scenario-Group-Out (LOSGO) Performance
- **Status: EXPLORATORY RE-EVALUATION**

Validation contains exactly **7 scenario groups**. Performance evaluated across each intact group:

| Group ID | Record Count | Description | Paths | Strict Exact | Accuracy |
| :--- | :---: | :--- | :--- | :---: | :---: |
| `compact-group-1af1630607e7a3c5f49c` | 20 | Archive folder open queries | `single_action` | 20 / 20 | **100.0%** |
| `compact-group-da242a5143b515385a25` | 4 | Orion project memory recall | `memory_recall` | 4 / 4 | **100.0%** |
| `compact-group-3a2b332a08f1dcc9926e` | 4 | Stable knowledge Q&A | `direct_response` | 4 / 4 | **100.0%** |
| `compact-group-aa9d6f78412adadd951b` | 4 | Multistep guides search | `multi_step_task` | 4 / 4 | **100.0%** |
| `compact-group-3367a99a5702cf6a58aa` | 3 | Historical memory queries | `memory_recall` | 2 / 3 | **66.7%** |
| `compact-group-73d1735100f11ceea56b` | 4 | Unresolved action followups | `clarify` | 4 / 4 | **100.0%** |
| `compact-group-0905a1846b0e754f74c5` | 1 | Music search / play | `escalate` | 0 / 1 | **0.0%** |

*LOSGO Analysis:*
- If Group 1 (Archive Folders) is held out, the remaining validation set is **18 / 20 (90.0%)**.
- If Group 5 (Historical Memory) is held out, the remaining validation set is **36 / 37 (97.3%)**.
- If Group 7 (Music Search) is held out, the remaining validation set is **38 / 39 (97.4%)**.

---

### Claim 5: Archive-Folder Performance vs. All Other Examples
- **Status: SEPARATE BREAKDOWN CONFIRMED**

| Cohort | Number of Records | Distinct Scenario Groups | Strict Exact Agreement | Percentage |
| :--- | :---: | :---: | :---: | :---: |
| **Archive-Folder Queries** | 20 | **1** | 20 / 20 | **100.0%** |
| **All Other Queries** | 20 | **6** | 18 / 20 | **90.0%** |
| **Total Validation Set** | **40** | **7** | **38 / 40** | **95.0%** |

*Crucial Observation:* Half of the entire validation benchmark represents a **single scenario group** of multilingual paraphrases for opening an archive folder. This single group masked the baseline's frequency collapse and now disproportionately boosts the v3 score.

---

### Claim 6: The Four `prior discussion` vs `Orion project` Cases
- **Status: AMBIGUOUS CONTRACT AUDITED (STRICT METRIC PRESERVED)**
- **Inspected Cases:** `dev_memory_project-en`, `dev_memory_project-hi-Latn`, `dev_memory_project-ta-Latn`, `dev_memory_project-te-Latn`.
- **Target Contract:**
  - Utterance mentions: `"Orion project"`
  - Context contains: `{'projects': [{'name': 'Orion', 'decision': 'use a local cache'}]}`
  - Target annotation: `{'reference': 'prior discussion', 'kind': 'prior_conversation', 'resolved_value': 'use a local cache'}`
- *Forensic Finding:* The ground-truth label `'prior discussion'` is **an abstract annotation label**, not an extractive source span from the utterance. In other memory records (`en-memory-001`, `hi-codeswitch-memory-001`), the annotator expected literal extractive spans (`'the database migration last week'`).
- The v3 implementation used a context-shape heuristic (`if any('decision' in p for p in projects): span = 'prior discussion'`) to satisfy this specific annotation convention.
- **Strict metric policy maintained:** No altered labels were introduced to the ground-truth benchmark; the strict exact metric was evaluated against the original file without modification.

---

### Claim 7: Real Local Encoder Latency Breakdown
- **Status: BENCHMARKED & VERIFIED**
- **Hardware Platform:** Windows 10 (Build 26300), GenuineIntel x86_64, PyTorch 2.9.0+cpu, 4 compute threads.
- **Model Path:** `cache['encoder_snapshot']` (`benchmarks/fast_classifier/models/v3/encoder`).
- **Measurements across all 40 validation records (after 5-sample warmup):**

| Pipeline Component | Median (ms) | p95 (ms) | Mean (ms) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Model Cold Startup** | **10,950 ms** | — | — | Weight deserialization from disk |
| **Encoder Forward Pass** | **170.6 ms** | 239.0 ms | 168.2 ms | Frozen Multilingual BERT inference |
| **Structured Ranking & Resolution** | **4.1 ms** | 4.9 ms | 3.9 ms | Cosine sim + lexical + resolvers |
| **Total Pipeline Latency** | **174.2 ms** | **244.0 ms** | **172.2 ms** | End-to-end `adapter.predict` |

The structured candidate ranker and typed resolver add only **~4.1 ms** of overhead over the raw neural encoder forward pass.

---

### Claim 8: Relevant Regression Suite Results
- **Status: ALL 49 RELEVANT TESTS PASSED**

| Test Suite | Total Tests | Passed | Failed | Skipped | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `reports/.../test_next_experiment.py` | 23 | 23 | 0 | 0 | **PASSED** |
| `tests/test_compact_classifier.py` | 18 | 18 | 0 | 0 | **PASSED** |
| `tests/test_fast_classifier.py` | 8 | 8 | 0 | 0 | **PASSED** |
| **TOTAL** | **49** | **49** | **0** | **0** | **100% PASS** |

Zero tests failed, broke, or were skipped.

---

### Claim 9: SHA-256 Hashes and File Integrity Audit
- **Status: BYTE-LEVEL INTEGRITY VERIFIED**

```json
{
  "baseline_model_v3_sha256": "7f4dd399b7238c7bf0c166454c3ff5aa09d6e3bf1afb82666b526d43cf452f05",
  "v2_model_candidate_sha256": "4a75e077e02573b36a911e468e513aff839ab37496883e64ac935122a28a458f",
  "v3_model_candidate_sha256": "546d5aaab563f9a4040ccd0504ff7173c4ed6d2e4bd37b8d9d9f87ec56339f96",
  "provisional_train_sha256": "67d57286b528b62abb04355ecb0933c3906f5aa59805d8ff68df424b1afd4c6d",
  "provisional_validation_sha256": "6f8cf0d9a8b1b7aab65d2fc2918bf26d327af00b4c76bf28f94aaad936a36692",
  "frozen_features_sha256": "8fe6183843885294a1b1ce601d954bcca09323b7a078e6dc911f29c8d4dc1082",
  "adapter_v3_code_sha256": "ea8f63134f070db86211120e1dc75623a83cd7f172b9756efa9cbfa2e198dbbb",
  "runner_v3_code_sha256": "d9fb6b31137cb514b0116d04b8a010a6ea5a27aef1c77ecaf5ec89868d66ef4f",
  "test_v3_code_sha256": "6aa37631059fe1ba1a9d773c5ab0e0dd873d48d8d6c3035fccb99f9b03f3f04d",
  "best_metrics_v3_sha256": "8e72547062d471622ad7ff58e0a72db5df649c32bbf38e1630cfb6f450dcea02"
}
```

- **Production Code Status:** `git status --porcelain=v1 assistant/` matches the pre-session state. Zero production files in `assistant/` were modified.

---

## 3. Audit Conclusion & Recommendations

1. **Conclusion:**
   The v3 architecture successfully demonstrates that **dense multilingual candidate ranking, argument feasibility checks, and structured context resolution** eliminate the severe frequency bias of naive logistic regression heads. The 38/40 score is verified and mathematically exact on the benchmark.
   However, because key reference-extraction and clarification rules were implemented after directly diagnosing the 40 validation failures, the performance cannot be certified as independently generalizable without evaluation on a truly unseen, independently annotated gold test set.

2. **Next Steps for Production Readiness:**
   - **Replace Lexical Keyword Heuristics with General Named Entity Recognition (NER) / Slot Fillers:** Instead of hardcoding `'database migration'` or `'kholo'`, use token-level extractive tagging for spans and slot-filling for arguments.
   - **Unify Annotation Contract Guidelines:** Standardize whether references should be literal source spans (e.g. `"Orion project"`) or canonical semantic labels (`"prior discussion"`).
   - **Evaluate on Sealed Test Set:** When authorized, evaluate the dual ranker against an independent gold-annotated test split that was never inspected during adapter development.

