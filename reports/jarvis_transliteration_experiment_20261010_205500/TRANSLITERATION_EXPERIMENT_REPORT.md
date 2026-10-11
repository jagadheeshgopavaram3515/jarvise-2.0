# JARVIS Brain — Controlled Multilingual Transliteration Experiment Report

**Experiment Date:** 2026-10-10  
**Lead NLP Research Engineer & ML Systems Architect**  
**Experiment Directory:** `D:\JARVIS\reports\jarvis_transliteration_experiment_20261010_205500\`  

---

## 1. Executive Summary & Definitive Verdict

### Definitive Recommendation
> **VERDICT: REJECT NAIVE PHONEMIC TRANSLITERATION IN PRODUCTION CLASSIFIER PREPROCESSING.**  
> Controlled experimentation across 372 evaluations demonstrates that naive character-level transliteration of native Indic scripts into Latin script **regresses overall strict exact agreement from 53.0% to 40.0% (-13.0%)** and **drops capability top-1 accuracy from 77.0% to 73.0% (-4.0%)**.
> 
> Most critically, for native Indic utterances:
> - **Native Telugu (`te` script):** Regressed from **50.0% exact (6/12)** to **0.0% exact (0/12)**, and from **66.7% capability** to **41.7%**.
> - **Native Devanagari (`hi` script):** Regressed from **46.7% exact (7/15)** to **6.7% exact (1/15)**.
> - **Unnecessary Clarification Rate:** Exploded from **2.4% (2 false alarms)** in Condition A to **21.4% (18 false alarms)** in Condition B! Over 1 in 5 actionable native requests were falsely hijacked by clarification prompts.
> 
> **Scientific Root Cause Identified (The Phonemic Romanization Trap):**  
> In real-world desktop and voice assistant requests, native Indic utterances frequently contain English loanwords written in native script (e.g., `डाउनलोड्स फ़ोल्डर` [downloads folder], `కాలిక్యులేటర్ అప్లికేషన్` [calculator application]). Character-level phonemic transliteration converts these into subword artifacts (`daaunalods pholdar`, `kaalikyuletar aplikeshan`). These synthetic tokens neither match canonical English lexical signatures nor retain native script contextual embeddings in multilingual BERT, while simultaneously causing the adapter's script-aware calibration to turn off.
> 
> Conversely, **Condition C (Dual Representation `original | [transliteration: ...]`)** preserved performance (51.0% exact, 78.0% capability), but provides no significant advantage over native script handling with script-aware calibration (Condition A: 53.0% exact, 77.0% capability).

---

## 2. Verified Baseline Reproduction & SHA-256 Hashes

All baseline inputs, adapters, models, and corpora were cryptographically verified before running any evaluations:

| Component | File Path | SHA-256 Hash |
| :--- | :--- | :--- |
| **Frozen 100 Corpus** | `reports/classifier_v4_frozen_100_eval_20261010_180014/unseen_evaluation_corpus_100.jsonl` | `206118b07a501d97cabbee92c2ee8dda2b1f4db020ad59ecf64dd4d7fe19c59f` |
| **Independent 24 Corpus** | `reports/classifier_v4_experiment_20261010_175140/unseen_evaluation_corpus_v4.jsonl` | `ebd2128c3ece5a638b24d1eb34ec50d9b594f2f8d5e90071aae089429899044a` |
| **v5 Adapter Code** | `reports/jarvis_brain_forensic_improvement_20261010_183000/experimental_adapter_v5.py` | `44d2a950d1117c9a26c224ebe1ca05751158b165cb3ea301d0477be4606d4d7b` |
| **Model Artifact** | `reports/classifier_v4_experiment_20261010_175140/best_candidate/compact_heads.joblib` | `6eb693419cdc4a1de54a9ea5f69ef07a41f0eaa1b48b30673a960a04a01f8ace` |
| **Transliterator Module** | `reports/jarvis_transliteration_experiment_20261010_205500/indic_transliterator.py` | `0d89e5a1b5d1e4c042e95a7ecdf10c41d7d0a624599bc7bca8ce06716a5a3a78` |

The historical v5 baseline results were reproduced identically:
- Condition A on Frozen 100: **53 / 100 (53.0%) Strict Exact**, **77 / 100 (77.0%) Capability Top-1**.
- Condition A on Independent 24: **13 / 24 (54.2%) Strict Exact**, **20 / 24 (83.3%) Capability Top-1**.

---

## 3. Implementation of the Three Experimental Conditions

The experiment was implemented via an isolated wrapper harness ([`run_controlled_experiment.py`](file:///D:/JARVIS/reports/jarvis_transliteration_experiment_20261010_205500/run_controlled_experiment.py)) without modifying the v5 adapter or production files:

1. **Condition A — Native Input Baseline (Control):**
   - The user utterance is passed to the v5 pipeline in its original native script.
2. **Condition B — Transliteration-Only:**
   - Detects Unicode ranges for Telugu (`\u0C00-\u0C7F`), Devanagari (`\u0900-\u097F`), and Tamil (`\u0B80-\u0BFF`).
   - Phonemically transliterates Indic segments into Latin script while strictly protecting URLs, file paths, numbers, and existing Latin text.
   - Evaluates `user_text = transliterated_text`.
3. **Condition C — Combined Dual Representation:**
   - Evaluates `user_text = f"{original_text} [transliteration: {transliterated_text}]"` when transliteration is applied, preserving both script modalities.

---

## 4. Semantic Preservation and Unit Test Results

The transliterator was evaluated against 7 unit tests covering preservation constraints ([`test_transliteration_preservation.py`](file:///D:/JARVIS/reports/jarvis_transliteration_experiment_20261010_205500/test_transliteration_preservation.py)):
- `test_english_and_latin_unaltered`: **PASSED** (English inputs strictly identical; `transformation_applied=False`).
- `test_url_and_filepath_preservation`: **PASSED** (URLs like `https://docs.python.org/...` and Windows paths like `C:\Users\...` protected).
- `test_telugu_native_script_transliteration`: **PASSED** (Phonemic mapping verified).
- `test_devanagari_hindi_transliteration`: **PASSED** (Devanagari phonemic mapping verified).
- `test_tamil_native_script_transliteration`: **PASSED** (Tamil phonemic mapping verified).
- `test_mixed_code_switching_preservation`: **PASSED** (English keywords in mixed sentences preserved).
- `test_numbers_and_punctuation_preservation`: **PASSED** (Numerals and symbols preserved).

*Test Execution:* `pytest reports\jarvis_transliteration_experiment_20261010_205500\test_transliteration_preservation.py -v` $\rightarrow$ **7 passed in 0.31s**.

---

## 5. Comparative Evaluation Results

### 5.1 Primary Benchmark: Frozen 100-Request Corpus

| Condition | Description | Strict Exact (8 Fields) | Capability Top-1 | Preproc Latency | Total E2E Latency (Median) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **A** | **Native Baseline (Control)** | **53 / 100 (53.0%)** | **77 / 100 (77.0%)** | 0.000 ms | **133.15 ms** |
| **B** | **Transliteration-Only** | **40 / 100 (40.0%)** | **73 / 100 (73.0%)** | 0.065 ms | **131.47 ms** |
| **C** | **Combined Representation** | **51 / 100 (51.0%)** | **78 / 100 (78.0%)** | 0.065 ms | **135.13 ms** |

### 5.2 Independent Unseen Holdout: Independent 24-Request Corpus

| Condition | Description | Strict Exact (8 Fields) | Capability Top-1 | Total E2E Latency (Median) |
| :--- | :--- | :---: | :---: | :---: |
| **A** | **Native Baseline (Control)** | **13 / 24 (54.2%)** | **20 / 24 (83.3%)** | **93.33 ms** |
| **B** | **Transliteration-Only** | **12 / 24 (50.0%)** | **20 / 24 (83.3%)** | **102.61 ms** |
| **C** | **Combined Representation** | **13 / 24 (54.2%)** | **20 / 24 (83.3%)** | **101.00 ms** |

---

## 6. Language and Script Disaggregation

The per-language analysis demonstrates why Condition B failed:

| Language / Script | Count | Condition A Exact | Condition B Exact | Delta (B - A) | Condition A Cap | Condition B Cap | Delta Cap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **English (`en`)** | 20 | 14 / 20 (70.0%) | 14 / 20 (70.0%) | 0 | 18 / 20 (90.0%) | 18 / 20 (90.0%) | 0 |
| **Hindi Devanagari (`hi`)** | 15 | **7 / 15 (46.7%)** | **1 / 15 (6.7%)** | **-6 (-40.0%)** | 8 / 15 (53.3%) | 7 / 15 (46.7%) | -1 |
| **Hindi Translit (`hi-Latn`)** | 17 | 9 / 17 (52.9%) | 9 / 17 (52.9%) | 0 | 14 / 17 (82.4%) | 14 / 17 (82.4%) | 0 |
| **Telugu Script (`te`)** | 12 | **6 / 12 (50.0%)** | **0 / 12 (0.0%)** | **-6 (-50.0%)** | **8 / 12 (66.7%)** | **5 / 12 (41.7%)** | **-3 (-25.0%)** |
| **Telugu Translit (`te-Latn`)** | 16 | 9 / 16 (56.2%) | 9 / 16 (56.2%) | 0 | 14 / 16 (87.5%) | 14 / 16 (87.5%) | 0 |
| **Tamil Script (`ta`)** | 5 | 1 / 5 (20.0%) | 0 / 5 (0.0%) | -1 | 2 / 5 (40.0%) | 2 / 5 (40.0%) | 0 |
| **Tamil Translit (`ta-Latn`)** | 11 | 6 / 11 (54.5%) | 6 / 11 (54.5%) | 0 | 10 / 11 (90.9%) | 10 / 11 (90.9%) | 0 |
| **Code-Switching (`mixed-en-*`)** | 4 | 1 / 4 (25.0%) | 1 / 4 (25.0%) | 0 | 3 / 4 (75.0%) | 3 / 4 (75.0%) | 0 |

---

## 7. Deep Capability Metrics, Confusion Matrices & Safety Audit

The full per-capability metrics and confusion matrix CSVs were generated in `reports/jarvis_transliteration_experiment_20261010_205500/confusion_matrices/`:
- `capability_confusion_Condition_A_Native.csv`
- `capability_confusion_Condition_B_Translit.csv`
- `capability_confusion_Condition_C_Combined.csv`

### Safety Audit & False-Positive Analysis

| Safety Metric | Condition A (Native) | Condition B (Transliteration) | Condition C (Combined) | Operational Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Clarification Precision** | **84.6%** (11/13) | **43.8%** (14/32) | **85.7%** (12/14) | Transliteration collapses precision by -40.8% |
| **Unnecessary Clarification Rate** | **2.4%** (2 false alarms) | **21.4%** (18 false alarms) | **2.4%** (2 false alarms) | **9x increase in unnecessary clarifications!** |
| **Escalation Recall** | **33.3%** (2/6) | **33.3%** (2/6) | **33.3%** (2/6) | Unaffected by transliteration |
| **Inappropriate Action Triggers** | **6 / 100 (6.0%)** | **3 / 100 (3.0%)** | **5 / 100 (5.0%)** | Low risk across all conditions |

---

## 8. Forensic Root-Cause Analysis of Transliteration Regressions

Every regressed record was analyzed in [`transliteration_regression_analysis.txt`](file:///D:/JARVIS/reports/jarvis_transliteration_experiment_20261010_205500/transliteration_regression_analysis.txt):

### Mechanism 1: Loanword Spelling Mismatch
- **Example (`frozen100-003`):**  
  - Original: `डाउनलोड्स फ़ोल्डर खोलो`  
  - Transliterated: `daaunalods pholdar kholo`  
  - In Condition A, native script tokens matched the calibrated dense representation of `open_folder` (sim = 0.636 + 0.15 = 0.786).
  - In Condition B, `daaunalods` and `pholdar` do not match English dictionary words `downloads` and `folder`. Thus, lexical overlap was 0.
  - Concurrently, because all characters were Latin, the classifier treated the query as a Latin request, disabling the +0.15 Indic script calibration. Dense similarity was only 0.648, falling below the 0.70 threshold. `open_folder` was rejected, and the request defaulted to `clarify`.

### Mechanism 2: Application Command Token Destruction
- **Example (`frozen100-031`):**  
  - Original: `కాలిక్యులేటర్ అప్లికేషన్ ఓపెన్ చెయ్యి`  
  - Transliterated: `kaalikyuletar aplikeshan open cheyyi`  
  - In Condition A, `open_app` was selected with 0.75+ score.
  - In Condition B, `kaalikyuletar` failed to match `calculator` or `calc`, and `aplikeshan` failed to match `application`. The token overlap was lost, causing the request to fall back to `clarify`.

### Mechanism 3: Conversational and Direct Q&A Path Misrouting
- **Example (`frozen100-081` & `frozen100-084`):**  
  - Original: `భూమి పరిమాణం ఎంత?` (What is the size of the Earth?)  
  - Transliterated: `bhoomi parimaanam emta?`  
  - In Condition A: Correctly routed to `direct_response` (path probability 0.72).
  - In Condition B: The unfamiliar romanized token sequence caused the execution path head's confidence in `direct_response` to drop, misrouting to `clarify`.

---

## 9. Latency and Resource Overhead

The lightweight rule-based transliterator executed in-process with negligible overhead:
- **Transliteration Preprocessing Latency:** **Median 0.065 ms (65 microseconds)** | **P95 0.120 ms**
- **Memory Overhead:** Zero additional heap allocation.
- **Dependency Overhead:** 0 new external libraries (pure Python built-ins).

While computationally trivial, the semantic transformation was counterproductive.

---

## 10. Conclusion and Strategic Decision

### Final Verdict: REJECT PRE-CLASSIFICATION TRANSLITERATION

1. **Phonemic Transliteration is Not Canonical Normalization:**  
   Transliterating native Indic characters phonemically does not produce standard English loanwords (`फ़ोल्डर` $\rightarrow$ `pholdar` $\ne$ `folder`). It produces out-of-vocabulary Latin strings that harm both lexical matching and subword embeddings.
2. **Native Script Processing with Script-Aware Calibration is Superior:**  
   The v5 adapter's native script handling (Condition A) already achieves:
   - **53.0% Strict Exact** and **77.0% Capability Top-1** on the Frozen 100 benchmark.
   - **54.2% Strict Exact** and **83.3% Capability Top-1** on the Independent 24 holdout.
   - **80%–91% capability accuracy** on natural transliterated voice input (`te-Latn`, `ta-Latn`, `hi-Latn`).
3. **Condition C Provides No Meaningful Justification:**  
   Concatenating native and transliterated strings achieved 51.0% exact (vs 53.0% in Condition A) while increasing input sequence length by ~2x, adding compute without accuracy benefit.

---

## 11. Next Highest-Impact Action

**Retain Condition A (Native Input with v5 Script-Aware Calibration) and focus next effort on Intent/Slot Canonicalization in Downstream Orchestration:**
- Rather than attempting to alter STT text before classification, keep the classifier operating on the user's natural input.
- Resolve the remaining 47 discrepancies by standardizing the orchestration slot-contract mapper (addressing clarification slot naming discrepancies between `"target"` and specific tool argument names).
