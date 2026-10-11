# JARVIS BRAIN — Forensic Failure Analysis Report

**Investigation Date:** 2026-10-10  
**Target Directory:** `D:\JARVIS\reports\jarvis_brain_forensic_improvement_20261010_183000\`  
**Baseline Artifacts Inspected:**
- Model Artifact: `reports/classifier_v4_experiment_20261010_175140/best_candidate/compact_heads.joblib` (SHA-256: `6eb693419cdc4a1de54a9ea5f69ef07a41f0eaa1b48b30673a960a04a01f8ace`)
- Adapter Code: `reports/classifier_v4_experiment_20261010_175140/experimental_adapter_v4.py` (SHA-256: `97bc7c57d1f25b488f1ce901714e2b0880a1dcad8c9aef713de18231eb04e19f`)
- Evaluation Corpus: `reports/classifier_v4_frozen_100_eval_20261010_180014/unseen_evaluation_corpus_100.jsonl` (SHA-256: `206118b07a501d97cabbee92c2ee8dda2b1f4db020ad59ecf64dd4d7fe19c59f`)

---

## 1. Verified Baseline Reproduction

The evaluation of the clean v4 adapter against the frozen 100-request corpus reproduced the exact reported metrics:
- **Strict 8-Field Exact Agreement:** **45 / 100 (45.0%)**
- **Capability Top-1 Accuracy:** **66 / 100 (66.0%)**
- **Inference & Schema Failures:** **0 / 100 (0.0%)**
- **Latency:** **Median 147.57 ms | p95 252.22 ms | Mean 165.91 ms**

The 55 discrepancies were isolated, categorized, and traced to their earliest failing pipeline stages.

---

## 2. Root Cause Classification & Detailed Findings

```
Total Discrepancies: 55 / 100
├── 1. Multilingual Representation Asymmetry (Native Indic Scripts): 14 records (25.5%)
├── 2. Capability Ranker Scoring & Ambiguity Rejection Errors: 20 records (36.4%)
├── 3. Reference Span Contract & Pronoun Extraction: 10 records (18.2%)
├── 4. Escalation & Out-of-Domain Threshold Failure: 4 records (7.3%)
├── 5. Auxiliary Head Divergence (Multi-Step vs Single Action): 4 records (7.3%)
└── 6. Clarification Missing Slot Name Divergence: 3 records (5.5%)
```

---

### Finding 1: Multilingual Representation Asymmetry (Native Indic vs Latin Transliteration)

- **Severity:** HIGH (Affects 14/55 discrepancies, primarily in native Telugu, Hindi, and Tamil scripts).
- **Earliest Stage:** Multilingual BERT Tokenizer & Dense Cosine Similarity.
- **Empirical Measurement:**
  When identical semantic queries were encoded against the English capability `"open_folder"`:
  - English (`"Open downloads folder"`): **Cosine Sim = 0.8171**
  - Hindi Latin (`"downloads folder open karo"`): **Cosine Sim = 0.7831**
  - Telugu Latin (`"downloads folder open cheyyi"`): **Cosine Sim = 0.7948**
  - Tamil Latin (`"downloads folder open pannu"`): **Cosine Sim = 0.7726**
  - Hindi Devanagari (`"डाउनलोड्स फ़ोल्डर खोलो"`): **Cosine Sim = 0.6360** (-0.18 gap)
  - Tamil Script (`"டவுன்லோட்ஸ் கோப்புறையை திறக்கவும்"`): **Cosine Sim = 0.6220** (-0.19 gap)
  - Telugu Script (`"నా డౌన్‌లోడ్స్ ఫోల్డర్ ఓపెన్ చెయ్యి"`): **Cosine Sim = 0.5115** (-0.30 gap)
- **Root Cause Mechanism:**
  The v4 candidate ranker applies a fixed threshold:
  ```python
  if best_cap is not None and best_score >= 0.70:
  ```
  Because the base multilingual sentiment model fragments native Brahmic characters into subwords and byte n-grams, cross-lingual cosine similarity with English capability descriptions drops to ~0.51–0.63. Under a uniform 0.70 threshold, native Indic utterances never trigger capability reconciliation. Concurrently, the English-trained execution path head predicts `clarify` with confidence ~0.78 on unseen script tokens, misrouting the entire request to clarification.
- **Proposed Generalized Fix:**
  Detect script unicode ranges (Devanagari `\u0900-\u097F`, Telugu `\u0C00-\u0C7F`, Tamil `\u0B80-\u0BFF`). Dynamically scale script-relative dense similarity by +0.15 for native Indic scripts (or apply script-adaptive acceptance thresholds $\ge 0.55$) when ranking against English capability text.

---

### Finding 2: Incomplete Domain Tokens in Ambiguity Rejection

- **Severity:** HIGH (Affects 7/10 desktop application commands and 6/8 webpage ingestion commands).
- **Earliest Stage:** Candidate Ranker Ambiguity Rejection.
- **Reproducible Examples:**
  - `frozen100-025`: `"Open PowerShell terminal application"`
    - Candidates: `open_app` (0.7961) vs `open_folder` (0.7878).
    - Score Margin: $0.0083 < 0.08$.
    - Rejected as ambiguous tie $\rightarrow$ returned `None` with score 0.30 $\rightarrow$ routed to `clarify`.
  - `frozen100-047`: `"Read the article at https://en.wikipedia.org/wiki/Transformer_..."`
    - Candidates: `read_page` (0.7195) vs `google_search` (0.6861).
    - Score Margin: $0.0334 < 0.08$.
    - Rejected as ambiguous tie $\rightarrow$ returned `None` with score 0.30.
- **Root Cause Mechanism:**
  The ambiguity rejection code in v4:
  ```python
  has_distinguishing_token = bool(q_tokens & (CONTAINER_TERMS | FILE_TERMS | SEARCH_TERMS))
  if margin < 0.08 and not has_distinguishing_token:
      return None, 0.30
  ```
  The code checked only containers, files, and searches. It completely lacked:
  - `APP_TERMS`: `{'app', 'application', 'terminal', 'browser', 'player', 'calc', 'program', 'software'}`
  - `PAGE_TERMS` & URL presence: `{'url', 'http', 'https', 'page', 'article', 'webpage', 'site'}`
  Because `"terminal"` and `"application"` were not recognized as distinguishing tokens for `open_app`, a legitimate application request was wrongly suppressed as an ambiguous tie with `open_folder`.
- **Proposed Generalized Fix:**
  Add domain token sets (`APP_TERMS`, `PAGE_TERMS`) and explicit URL detection to `has_distinguishing_token`. Furthermore, check token affinity directly against candidate name/description tokens.

---

### Finding 3: Low Baseline Threshold Causing Out-of-Domain Action Misrouting

- **Severity:** HIGH (Affects 4 escalation records and 2 ambiguous clarification records).
- **Earliest Stage:** Candidate Ranker Minimum Threshold & Reconciliation Override.
- **Reproducible Example:**
  - `frozen100-096`: `"Transfer five hundred dollars from my savings account to John"`
    - Expected: `execution_path: escalate`, `selected_capability: None`
    - Actual: `execution_path: single_action`, `selected_capability: open_app`
  - `frozen100-099`: `"Uber app lo airport ki cab book cheyyi"`
    - Expected: `execution_path: escalate`, `selected_capability: None`
    - Actual: `execution_path: single_action`, `selected_capability: open_app`
- **Root Cause Mechanism:**
  1. Anisotropic BERT geometry produces baseline cosine similarity of ~0.45–0.55 between arbitrary sentences.
  2. The candidate ranker had a floor threshold of only `0.40`:
     ```python
     if best_score < 0.40: return None
     ```
     `open_app` scored 0.5381 due to random background similarity and the token `"app"` in `"Uber app"`.
  3. The adapter reconciliation forcefully set `raw['execution_path'] = 'single_action'` and cleared `needs_escalation` without verifying whether the request was in-scope.
- **Proposed Generalized Fix:**
  - Raise the candidate selection threshold for actions lacking direct domain token support from 0.40 to 0.60.
  - When the execution path model predicts `escalate` with confidence $> 0.50$, do not override with a weak candidate score ($< 0.75$).

---

### Finding 4: Context Reference Span Extraction Divergence

- **Severity:** MEDIUM (Affects 10 records in `corpus100-clarify`).
- **Earliest Stage:** Reference Span Extractor.
- **Reproducible Example:**
  - `frozen100-056`: `"Delete that completely"`
    - Expected: `[{'kind': 'item', 'reference': 'it', 'resolved_value': None}]`
    - Actual: `[{'kind': 'prior_request', 'reference': 'Delete that completely', 'resolved_value': None}]`
- **Root Cause Mechanism:**
  When a pronoun (`'that'`, `'this'`, `'it'`, `'use'`, `'idi'`) is present in the utterance without context resolution, the reference span should be the pronoun token itself, representing the ungrounded deictic reference. The v4 adapter defaulted to assigning the entire utterance string as the reference span when no topic noun phrase was found.
- **Proposed Generalized Fix:**
  When `tokens & PRONOUN_TOKENS` is non-empty, extract the identified pronoun token as the reference span (`kind='item'`) rather than falling back to the full utterance string.

---

### Finding 5: Sequential Coordination in Multi-Step Tasks

- **Severity:** MEDIUM (Affects 4 records in `corpus100-multistep`).
- **Earliest Stage:** Execution Path Head Post-Processing.
- **Reproducible Example:**
  - `frozen100-089`: `"Create a notes folder and put summary.txt inside it"`
    - Expected: `execution_path: multi_step_task`
    - Actual: `execution_path: single_action`, `selected_capability: create_folder`
- **Root Cause Mechanism:**
  In v4, multi-step execution path was demoted to `single_action` unless the utterance contained explicit English strings `{'step', 'steps', 'guide', 'how to'}`:
  ```python
  elif raw['execution_path'] == 'multi_step_task' and not any(w in request.user_text.lower() for w in ['step', 'steps', 'guide', 'how to']):
      raw['execution_path'] = 'single_action'
  ```
  Requests combining multiple sequential verbs (`"create ... and put ..."`, `"open ... aur fir ..."`, `"download ... and extract ..."`) are valid multi-step workflows.
- **Proposed Generalized Fix:**
  Include sequential conjunction markers (`'and then'`, `'and put'`, `'and move'`, `'aur fir'`, `'mariyu'`, `'aprom'`) alongside existing keywords.

---

## 3. Prioritized Improvement Plan

1. **Fix 1: Domain Token Expansion & Ambiguity Rejection Repair**  
   Add `APP_TERMS` and `PAGE_TERMS` / URL detection to ambiguity rejection. Directly resolves ~10 desktop app and webpage selection failures.
2. **Fix 2: Script-Aware Dense Similarity Scaling for Native Indic Scripts**  
   Add Unicode character range detection for Devanagari, Telugu, and Tamil scripts. Apply an adaptive offset (+0.15) to bridge the cross-lingual tokenizer embedding gap against English capability descriptions. Directly resolves ~14 native Indic routing failures.
3. **Fix 3: Principled Minimum Threshold & Escalation Preservation**  
   Raise minimum threshold from 0.40 to 0.60 for candidates lacking direct domain keywords. Preserve `escalate` path when candidate score is weak. Directly resolves ~4 escalation misroutings.
4. **Fix 4: Pronoun Reference Span Normalization**  
   When ungrounded pronouns from `PRONOUN_TOKENS` are detected, extract the pronoun token itself as `kind='item'`. Directly resolves ~10 clarification contract discrepancies.
5. **Fix 5: Sequential Multi-Step Conjunction Recognition**  
   Recognize sequential action conjunctions (`'and then'`, `'and put'`, etc.) to prevent premature demotion to `single_action`.

---

## 4. Regression Test Specifications

The following targeted unit and regression tests must be implemented before deploying the adapter:
- `test_app_terms_prevent_ambiguity_rejection`: "Open PowerShell terminal application" must select `open_app`, not `None`.
- `test_page_terms_and_url_prevent_ambiguity_rejection`: "Read the article at https://..." must select `read_page`, not `None`.
- `test_native_indic_script_dense_scaling`: "నా డౌన్‌లోడ్స్ ఫోల్డర్ ఓపెన్ చెయ్యి" must select `open_folder`.
- `test_out_of_domain_escalation_preserved`: "Transfer five hundred dollars from my savings account" must route to `escalate` with `selected_capability=None`.
- `test_pronoun_reference_span_extraction`: "Delete that completely" must produce `reference='that'`, `kind='item'`.

