# JARVIS COMPACT BERT v4 — GENERALIZATION EXPERIMENT REPORT

**Date:** 2026-10-10  
**Experiment ID:** `classifier_v4_experiment_20261010_175140`  
**Location:** `D:\JARVIS\reports\classifier_v4_experiment_20261010_175140\`  
**Predecessor Experiments:**
- Baseline: `benchmarks/fast_classifier/models/v3/` (8 / 40, 20.0%)
- Experimental v2: `reports/classifier_completion_20261010_163425/` (10 / 40, 25.0%)
- Experimental v3: `reports/classifier_next_experiment_20261010_171102/` (38 / 40, 95.0%)
- Generalization Audit: `reports/classifier_generalization_audit_20261010_173628/`  
**Production Status:** Zero production files modified in `assistant/`. Completely isolated, bounded experiment.

---

## 1. Executive Summary & Verdict

### Generalization Status: **90% GENERALIZATION IS UNSUPPORTED ON UNSEEN DATA**

In response to the forensic audit (`classifier_generalization_audit_20261010_173628`), this experimental v4 pipeline **systematically removed all validation-specific memorized rules, keyword triggers, and target-specific heuristics** from the inference adapter.

### Summary of Controlled Findings:

1. **Original 40-Record Validation Set (After Heuristic Removal):**
   - **Strict Full-Record Agreement:** **31 / 40 (77.5%)** (down from v3's memorized 38/40, but still vastly higher than baseline's 8/40 and v2's 10/40).
   - **Capability Top-1 Accuracy:** **39 / 40 (97.5%)** (demonstrating that dense multilingual cosine similarity and argument feasibility remain robust across languages without keyword hacking).
   - The drop of 7 records is exclusively due to removing arbitrary span heuristics for historical memory and ambiguous clarification annotations.

2. **Newly Authored 24-Record Unseen Evaluation Corpus:**
   - **Strict Full-Record Agreement:** **12 / 24 (50.0%)**
   - **Capability Top-1 Selection:** **19 / 24 (79.2%)**
   - **Request Mode Agreement:** **22 / 24 (91.7%)**
   - **Needs Escalation Agreement:** **22 / 24 (91.7%)**
   - **Execution Path Agreement:** **19 / 24 (79.2%)**
   - **Needs Clarification Agreement:** **20 / 24 (83.3%)**
   - **Missing Information Agreement:** **20 / 24 (83.3%)**
   - **Context References Agreement:** **19 / 24 (79.2%)**

3. **Latency SLA:**
   - Evaluated on local CPU using live `FrozenMultilingualEncoder`: **Median 120.4 ms** / **p95 219.4 ms** total pipeline latency, well inside the sub-300ms voice hot path requirement.

4. **Honest Architectural Assessment:**
   While the structured dual ranker genuinely and reliably selects the correct capability (~80% on unseen requests, 97.5% on validation), **end-to-end multi-field strict exact agreement on diverse unseen requests cannot honestly be claimed at 90%**. In unconstrained multilingual voice environments, joint strict agreement is realistically **50.0% to 77.5%**.

---

## 2. Inventory of Removed Rules and Replacements

Every post-hoc heuristic identified in the audit was excised from [`experimental_adapter_v4.py`](file:///D:/JARVIS/reports/classifier_v4_experiment_20261010_175140/experimental_adapter_v4.py):

| Memorized / Heuristic Rule (v3) | Why It Was Defective | General Replacement in v4 |
| :--- | :--- | :--- |
| `elif 'database migration' in text: span = 'the database migration last week'` | Hardcoded string match tailored to validation query `en-memory-001`. | **General Topic Span Extraction:** Dynamically searches `recent_topics` in context and extracts matching topic mentions via regex word boundaries directly from source utterance. |
| `elif 'architecture' in text: span = 'Kal architecture ke baare mein'` | Hardcoded string match tailored to validation query `hi-codeswitch-memory-001`. | **Source Extraction:** Extracts matching topic noun phrases from `user_text` without inventing ungrounded clauses. |
| `if any('decision' in p for p in projects): span = 'prior discussion'` | Context-shape hack forcing the abstract string `'prior discussion'` for `dev_memory_project-*`. | **Clean Reference Contract:** If utterance mentions an entity or query, extracts the actual entity or utterance text. Does not hallucinate the arbitrary string `'prior discussion'`. |
| `any(w in text for w in ['it', 'kholo', 'open', 'bring', 'target']) -> name = 'target'` | Validation-specific vocabulary list added to force `target` on `dev_clarify_unresolved-hi-Latn` and `dev_clarify_unresolved-en`. | **Inference from Structural State:** If `needs_clarification` is True and the learned head outputs `__none__`, infers `name = 'target'` only if pronouns are present without context or an action verb lacks arguments. No language-specific word hacking. |
| `if 'open_file' in cap_names and 'open_folder' in cap_names: ... return None, 0.30` | Special-cased candidate pair filter tailored specifically to `dev_clarify_unresolved` candidate lists. | **General Margin-Based Ambiguity Filter:** If the score difference between the top-1 and top-2 candidates is $< 0.08$ AND neither candidate has distinct token support in the query, rejects top candidate to prompt clarification. |

---

## 3. Reference Extraction Contract (v4.0)

To resolve the historical ambiguity between abstract labels and extractive spans:

1. **Literal Extractive Spans:**
   - Any reference extracted from an utterance must be a valid substring of `request.user_text`.
   - Pronouns across supported languages (English: *it/that/this*, Hindi: *use/isse/yeh/woh*, Tamil: *atha/athai/ithu*, Telugu: *adi/idi/danni*) are recognized and normalized to canonical item reference `'it'`.
   - If context contains a project alias or topic, the exact matching token span in the utterance is extracted via character offsets.

2. **Typed Context Resolution:**
   - The resolved entity value is extracted strictly from typed structured context (`context['projects']`, `context['recent_topics']`, `context['resolved_references']`).
   - Hallucinated or fabricated scalars are rejected (empty context always returns `resolved_value = None`).

3. **Ground-Truth Annotation Incompatibilities:**
   - In `provisional_validation.jsonl`, the four `dev_memory_project-*` records expect `reference = 'prior discussion'` even though the user literally said *"Remind me what we chose for the Orion project."*
   - Under v4's clean extractive contract, the model extracts the utterance / entity mention rather than hallucinating `'prior discussion'`.
   - As instructed, the original validation ground truth was kept unchanged, and these 4 mismatches are transparently reported as contract divergence.

---

## 4. Quantitative Results & Comparison

### A. Original Validation Set (40 Records): Baseline vs v2 vs v3 vs v4

| Contract Field | Baseline | Experimental v2 | Experimental v3 (Heuristic) | Experimental v4 (Clean) |
| :--- | :---: | :---: | :---: | :---: |
| **`intent`** | 8 / 40 | 10 / 40 | 39 / 40 | **39 / 40 (97.5%)** |
| **`execution_path`** | 31 / 40 | 33 / 40 | 39 / 40 | **39 / 40 (97.5%)** |
| **`request_modes`** | 39 / 40 | 39 / 40 | 39 / 40 | **39 / 40 (97.5%)** |
| **`selected_capability`** | 20 / 40 | 20 / 40 | 39 / 40 | **39 / 40 (97.5%)** |
| **`needs_clarification`** | 34 / 40 | 34 / 40 | 40 / 40 | **40 / 40 (100.0%)** |
| **`needs_escalation`** | 36 / 40 | 36 / 40 | 39 / 40 | **39 / 40 (97.5%)** |
| **`missing_information`** | 36 / 40 | 36 / 40 | 40 / 40 | **39 / 40 (97.5%)** |
| **`context_references`** | 29 / 40 | 31 / 40 | 40 / 40 | **33 / 40 (82.5%)** |
| **STRICT FULL EXACT** | **8 / 40 (20.0%)** | **10 / 40 (25.0%)** | **38 / 40 (95.0%)** | **31 / 40 (77.5%)** |
| **CAPABILITY TOP-1** | 20 / 40 | 20 / 40 | 39 / 40 | **39 / 40 (97.5%)** |

*Analysis of the 9 Mismatches in v4 on Validation:*
- 4 records (`dev_memory_project-*`): Model cleanly extracts utterance span instead of hallucinating `'prior discussion'`.
- 2 records (`en-memory-001`, `hi-codeswitch-memory-001`): Model cleanly extracts topic name (`'database migration'`, `'architecture'`) instead of memorizing clause boundaries (`'the database migration last week'`).
- 1 record (`dev_clarify_unresolved-en`): Learned head predicted `prior_request` without the `'bring'` keyword hack.
- 1 record (`ctx-hologram-001`): Training distribution mismatch (0 training records had both `memory` and `conversational`).
- 1 record (`en-music-001`): Adjudication boundary case (`youtube_search` offered in candidate list vs `escalate`).

---

### B. New Unseen Evaluation Corpus (24 Diverse Records)

Evaluated with live PyTorch `FrozenMultilingualEncoder`:

| Metric / Field | Agreement on Unseen Corpus | Percentage |
| :--- | :---: | :---: |
| **Strict Full-Record Agreement** | **12 / 24** | **50.0%** |
| **Capability Top-1 Selection** | **19 / 24** | **79.2%** |
| `request_modes` | 22 / 24 | **91.7%** |
| `needs_escalation` | 22 / 24 | **91.7%** |
| `execution_path` | 19 / 24 | **79.2%** |
| `needs_clarification` | 20 / 24 | **83.3%** |
| `missing_information` | 20 / 24 | **83.3%** |
| `context_references` | 19 / 24 | **79.2%** |
| `intent` | 15 / 24 | **62.5%** |

#### Language Breakdown on Unseen Corpus:

| Language Tag | Scripts | Denominator | Exact Matches | Capability Top-1 |
| :--- | :--- | :---: | :---: | :---: |
| **`en`** | Latin | 8 | 4 (50.0%) | 6 (75.0%) |
| **`hi`** | Devanagari | 2 | 2 (100.0%) | 2 (100.0%) |
| **`hi-Latn`** | Latin | 3 | 2 (66.7%) | 3 (100.0%) |
| **`mixed-en-hi`** | Latin | 1 | 0 (0.0%) | 1 (100.0%) |
| **`mixed-en-te`** | Latin | 1 | 0 (0.0%) | 1 (100.0%) |
| **`ta`** | Tamil | 1 | 1 (100.0%) | 1 (100.0%) |
| **`ta-Latn`** | Latin | 3 | 2 (66.7%) | 3 (100.0%) |
| **`te`** | Telugu | 2 | 1 (50.0%) | 1 (50.0%) |
| **`te-Latn`** | Latin | 3 | 0 (0.0%) | 1 (33.3%) |
| **TOTAL** | Multi-Script | **24** | **12 (50.0%)** | **19 (79.2%)** |

---

### C. Live Latency Benchmark on Unseen Corpus

| Component | Median (ms) | p95 (ms) | Budget SLA |
| :--- | :---: | :---: | :---: |
| **Encoder Forward Pass** | 116.8 ms | 214.7 ms | $< 250$ ms |
| **Ranking & Resolution** | 3.6 ms | 4.7 ms | $< 10$ ms |
| **Total Pipeline Latency** | **120.4 ms** | **219.4 ms** | $< 300$ ms |

---

## 5. Regression & Integrity Suite

- **Test Suite Results:** All 7 unit and regression tests in [`test_v4_experiment.py`](file:///D:/JARVIS/reports/classifier_v4_experiment_20261010_175140/test_v4_experiment.py) passed in 4.40s.
- **Production Isolation:** `git status --porcelain=v1 assistant/` confirms zero modifications in production code.

---

## 6. Final Architecture Recommendations

1. **Stop Chasing 90% Exact Agreement on Naive Multi-Field Strings:**
   The Compact BERT pipeline's real job in JARVIS is **routing, intent discrimination, and capability ranking**. In capability ranking, the dual ranker achieves **79.2% to 97.5% top-1 accuracy**. Demanding 90% bit-exact agreement across 8 disparate fields (including descriptive text and clause spans) inevitably encourages heuristic overfitting.

2. **Decouple Downstream Routing from Surface Phrasing:**
   The downstream dispatcher and LangGraph supervisor should route based on `(execution_path, selected_capability, request_modes)`. Missing descriptions and reference spans should be handled by downstream conversational slot fillers rather than early routing classifiers.

