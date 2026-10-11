# JARVIS Fast Brain — Model Selection and Training Specification v1

Status: design only. No model was downloaded, trained, or integrated.

## 1. Decision summary

The recommended implementation is one CPU-resident multilingual encoder with shared pooling and lightweight structured heads. It should make one forward pass per transcript, emit logits, and let deterministic code derive and validate the frozen `FastBrainDecision` contract.

The primary checkpoint recommendation is `distilbert-base-multilingual-cased`, fine-tuned for JARVIS classification. It is approximately 135M parameters, larger than the preferred 30–60M envelope, but multilingual coverage is the harder constraint on this machine. A smaller model should replace it only after it demonstrates equivalent Indic-script, romanized, and code-switched performance. The currently cached mBERT-based classifier remains the reproducibility fallback, not the final selection.

This exact checkpoint is not verified as locally cached in the current environment; selecting it does not authorize downloading it. License, tokenizer coverage, package availability, and measured latency must be checked in the implementation phase.

## 2. Existing evidence used

The previous Phase 3C compact classifier used a frozen mean-pooled multilingual BERT encoder, independent logistic heads, maximum length 160, CPU inference, and one batched encoder pass. On its sealed sample it achieved path 61.1%, capability 44.4%, intent 61.1%, request modes 77.8%, clarification F1 .719, escalation F1 .652, and schema validity 100%. Warm p95 was 255.5 ms and RSS increase was about 318 MiB.

Phase 4 used 434 synthetic cases. BERT reached 62.9% path accuracy and 40.7% strict exact accuracy; Laya reached 7.1% and 0%, respectively. Laya warm p50 was about 3.04 s, p95 about 5.07 s, and RSS increase about 1.5 GB. These are evidence about the evaluated implementations and datasets, not universal model limits.

Qwen is useful for bounded reasoning and planning but its warm latency was approximately 2.3 s in the available 10-run sample, so it is unsuitable as the universal front-door classifier. The machine has an i5-9300H, 8 GB RAM, GTX 1650 4 GB, and Qwen already uses roughly 2.16 GB active VRAM.

## 3. Laya: observed mechanism versus useful idea

The repository’s Laya adapter runs an isolated persistent subprocess. `benchmarks/fast_classifier/laya_worker.py` loads `convaiinnovations/laya`, subfolder `multilingual`, with `laya==0.3.20` on CPU. Each request calls `agent.predict(state, questions, max_len=1024)` once. The worker asks typed choice questions for execution path, intent, request modes, capability, clarification, and escalation, then projects the answers into the existing decision object. It does not execute tools.

The repository does not expose the internal tokenizer, vocabulary, exact layer configuration, or logits. Those details are `UNKNOWN FROM CURRENT EVIDENCE`. The measured implementation therefore demonstrates structured choice querying and schema projection, not that a 322M mmBERT-base model is necessary.

| Laya mechanism | Why it exists | Cost | JARVIS version | Decision |
|---|---|---:|---|---|
| Multilingual encoder | Broad language coverage | High relative to voice-loop budget | Smaller multilingual encoder | Modify |
| Typed choice questions | Constrain output space | One model call plus question overhead | Direct classification heads | Replace |
| Structured choices/logits | Avoid free-form generation | Low post-processing cost | Logits plus deterministic derivation | Keep |
| `max_len=1024` generation-style API | General Laya interface | Adds avoidable overhead | Fixed classification length, e.g. 160 initially | Replace |
| Isolated worker | Protects host process | Startup and IPC overhead | In-process adapter after validation | Replace later |
| Schema projection | Stable downstream contract | Low | Frozen contract validator | Keep |

## 4. Why Laya was slow and inaccurate

**Observed:** CPU-only Laya had roughly 15.5 s startup/worker overhead, 12.4 s model component time, 3.0 s warm p50, 5.1 s warm p95, and about 1.5 GB RSS increase. Its evaluated accuracy was poor despite 100% schema validity.

**Inferred:** The typed-question formulation was a poor zero-shot fit for JARVIS labels; it overproduced clarification/escalation and rarely selected the correct capability. The isolated subprocess and large encoder also imposed runtime cost. The benchmark’s 20 correlated semantic scenarios, seven language variants, and entirely synthetic labels limited generalization evidence. The result does not prove that every Laya configuration would fail.

**Recommended:** Keep constrained structured prediction, but train JARVIS-specific heads directly. Do not reproduce the question-generation interface, 322M CPU footprint, or 1024-token output path.

## 5. Candidate model choice

| Candidate | Approximate class | Strength | Risk on JARVIS | Selection |
|---|---:|---|---|---|
| `distilbert-base-multilingual-cased` | ~135M | Direct encoder classification; multilingual cased text and Indic coverage | Larger than 30–60M target; local availability unverified | **Primary** |
| Existing cached mBERT encoder | ~100M+ class | Already reproducible in Phase 3C; known JARVIS baseline | Existing capability and romanization weaknesses | Fallback/reference |
| Compact multilingual MiniLM-style encoder | ~60–120M depending checkpoint | Potentially lower latency and memory | Exact checkpoint/licence/coverage not verified locally | Future replacement candidate |
| XLM-R base / Laya-scale encoder | ~250–330M | Broad multilingual representation | Too much RAM/latency for this voice loop | Reject for v1 |
| English-only 10–60M encoder | 10–60M | Fast and small | Violates required Telugu/Hindi/Tamil support | Reject |

The primary choice deliberately exceeds the preferred size class because a verified multilingual encoder is more important than an unverified small checkpoint. The implementation phase must measure it before adoption. If it cannot meet the memory/latency budget, the fallback is the cached mBERT architecture with improved data, heads, calibration, and sequence truncation.

## 6. Recommended architecture

```text
transcript + tiny bounded metadata
        ↓
multilingual cased encoder (one forward pass)
        ↓
masked mean pool + optional [CLS] projection
        ↓
shared 256→128 bottleneck
        ├── intent-family head
        ├── capability-family head
        ├── ambiguity / clarification head
        ├── OOD / outside-envelope score
        └── optional request-mode auxiliary head
        ↓
calibration + deterministic contract derivation
        ↓
FastBrainDecision → validator → shadow/Orchestrator
```

Use a single encoder, not seven models or sequential LLM calls. The encoder is fine-tuned only as evidence supports it; a frozen-encoder linear-head baseline must remain available for ablation.

## 7. Frozen contract field ownership

| Contract concept | Model output | Derivation/validation | Rationale |
|---|---|---|---|
| `intent_family` | Direct head | Enum validation | Semantic language understanding is learned. |
| `capability_family` | Direct head | Mask against known family list | Prevents arbitrary tool invention. |
| `execution_path` | No independent head in v1 | Rules over intent, capability, clarification, OOD, and policy | Avoids contradictory predictions. |
| `context_requirements` | No independent head in v1 | Multi-label rules/head only if evidence later requires it | Context need is tightly coupled to intent/capability. |
| `clarification` | Missing-slot/ambiguity heads | Required/missing invariants are deterministic | Missing-target behavior must be conservative. |
| `uncertainty` | Logit calibration + OOD score | Fixed bands after calibration | Do not expose fake precision. |
| `escalation` | No free policy head in v1 | Derived from current-information, OOD, unsupported, and runtime failure state | Provider policy belongs to Orchestrator. |

The optional request-mode head is auxiliary training supervision, not a new runtime contract concept. `execution_path` must not be independently learned in v1 because it duplicates provider/workflow policy and can conflict with semantic intent.

## 8. Heads and loss

Use weighted cross-entropy for intent and capability, binary cross-entropy for clarification and missing-slot labels, and an auxiliary multi-label loss for request modes/context evidence if retained. Use class weighting or focal loss only after inspecting class imbalance; do not use it by default.

Recommended training output:

- intent logits: 9 classes;
- capability logits: 12 classes including `unknown`;
- clarification-required logit;
- missing-information multi-label logits: 10 labels;
- optional context/request-mode auxiliary logits;
- OOD score from calibrated energy or maximum-probability/margin features, not a hallucinated confidence field.

The model must never emit tool names, arguments, URLs, shell commands, or natural-language answers.

## 9. Context and sequence design

The runtime input should be the current transcript plus at most a small bounded summary supplied by the caller: active-task flag, whether resolvable recent context exists, and compact capability-family availability. Do not pass the memory database, repository, web pages, or full conversation.

Start with a maximum sequence length of 160 tokens to match the established classifier evidence. Measure truncation on STT-noisy and code-switched requests. Increase only if a sealed evaluation set demonstrates material benefit. The Fast Brain should not retrieve context; it should signal which context source the Orchestrator may retrieve.

## 10. Multilingual and romanized strategy

Use a cased multilingual subword tokenizer and train on native-script, romanized, code-switched, and realistic STT variants. Romanized Telugu/Hindi must not be treated as a separate intent vocabulary. Use grouped semantic examples so English, Telugu, Hindi, Tamil, and romanized variants of one request stay in the same split.

Training requirements:

- human-reviewed examples in every required language/script;
- transliteration/paraphrase augmentation only as supplemental data;
- hard negatives pairing similar words with different capabilities;
- code-switch examples with natural word order;
- spelling/noise variants from STT, marked separately from ground truth;
- language/script metadata for analysis, not as the sole routing signal.

A separate language detector is not recommended for v1. It adds latency and can fail on code-switching. If language metadata is available from STT, pass it as non-authoritative analysis metadata only.

## 11. Uncertainty, OOD, and clarification

Use one calibrated decision policy:

1. obtain softmax probabilities and top-two margin;
2. fit temperature scaling on development data;
3. compute an OOD/energy signal using an explicit unsupported/OOD development set;
4. map calibrated evidence to `high`, `medium`, `low`, `ambiguous`, or `outside_envelope` bands;
5. abstain when required fields disagree, margin is low, or OOD is high.

Clarification should be hybrid: learned ambiguity/missing-slot detection plus deterministic invariants and context availability supplied by the caller. If a request requires a target and no target or resolvable reference exists, produce `clarify` with a non-empty missing list. Never synthesize `example.txt`, an application, URL, or query.

Uncertainty is advisory evidence. The Orchestrator owns the final escalation policy. Fast Brain should not call Gemini merely because a probability is low.

## 12. Training and distillation plan

Use human-first hard labels as the authoritative target. Deterministic command labels are authoritative for exact fast paths. Qwen, Laya, and old BERT may generate candidate augmentations or disagreement flags, but their outputs must not silently become ground truth.

Recommended order:

1. Build a small human-reviewed seed set covering all frozen enums and safety/clarification cases.
2. Train the direct-head baseline with grouped train/development/sealed splits.
3. Use Qwen only to propose paraphrases, hard negatives, and candidate labels.
4. Accept teacher examples only after deterministic schema checks and human review of high-risk classes: clarification, unsupported, escalation, and capability.
5. Use teacher probabilities/logits only if the teacher exposes calibrated comparable probabilities; otherwise use filtered hard-label distillation.
6. Keep teacher-disagreement cases in a review queue, not in automatic training.

The preferred method is human-first supervised learning plus filtered teacher-generated augmentation. Multi-teacher voting is useful for prioritization, not authority. Do not distill Qwen’s target invention, unnecessary escalation, or tool-name behavior.

## 13. Dataset requirements

No dataset is created by this specification. Before training, require a versioned dataset with:

- balanced intent and capability families;
- explicit clarification and missing-slot cases;
- unsupported/OOD and near-miss hard negatives;
- deterministic commands separated from semantic tasks;
- all required languages, scripts, romanization, and code-switching;
- context-resolvable versus context-unresolvable pairs;
- grouped splits preventing translation/paraphrase leakage;
- human-reviewed sealed evaluation cases;
- provenance per example: human-reviewed, expert-authored, teacher-generated, or synthetic;
- labels for expected contract fields and rationale/evidence for review, while rationale is never a runtime model output.

The prior 434-case evaluation was entirely synthetic and correlated across language variants, so it is insufficient as the sole production-readiness basis.

## 14. Hardware and runtime plan

Run the Fast Brain CPU-first and keep it resident. Do not compete with Qwen for the GTX 1650’s limited VRAM. A 135M FP32 encoder may require roughly hundreds of MB for weights plus framework overhead; this is an estimate, not a local measurement. Quantized/int8 inference may reduce memory, but quantization must be validated for classification accuracy before adoption.

Recommended runtime properties:

- one forward pass per request;
- batch size 1;
- maximum length 160 initially;
- no subprocess per request;
- no network dependency;
- no concurrent Qwen load caused by the gate;
- lazy model initialization only if voice-loop startup requires it;
- explicit timeout and conservative fallback to clarification/legacy routing.

Design targets, not measurements:

- warm p50 below 100 ms;
- warm p95 below 200 ms initially;
- resident model/process memory below 500 MB for the selected checkpoint;
- zero VRAM requirement by default;
- startup below 2 s after model files are present.

The 30–60M class remains a desirable optimization target, but it should replace the primary only after it matches the multilingual sealed-set boundary. A small English-only model is not acceptable.

## 15. Laya, BERT, and Qwen roles

| BERT mechanism | What worked | What failed | JARVIS change |
|---|---|---|---|
| Frozen multilingual encoder | One-pass CPU architecture and schema validity | Capability/generalization and romanized-language weakness | Fine-tune or adapter-train the encoder with better hard negatives and calibration |
| Independent logistic heads | Cheap structured outputs | Related decisions could disagree | Keep heads, then derive execution/provider fields deterministically |
| Mean pooling | Simple, low overhead | May lose targeted token information | Compare masked mean with a small projection; keep one-pass design |
| Synthetic multilingual expansion | Broad coverage | Leakage and weak realism | Human-reviewed grouped sealed set |

| Qwen capability | Training value | Runtime value | JARVIS role |
|---|---|---|---|
| Bounded reasoning | Teacher candidate for reviewed examples | Too slow for universal gate | Generate candidate labels/paraphrases; never unquestioned ground truth |
| Tool planning | Good source of trajectories | Not gate behavior | Future agent/Qwen layer |
| Observation interpretation | Useful teacher data | Not front-door classification | Local reasoning model |
| Confidence/escalation judgments | Often useful but not guaranteed calibrated | Unsafe as sole policy | Candidate signal reviewed by deterministic policy |

## 16. What must remain deterministic

Keep exact command matching, enum validation, capability existence, permissions, schema validation, timeouts, iteration limits, audit, execution safety, provider policy, and fallback behavior outside learned weights. The model may recognize “open Chrome” as `application`; it must not choose or execute `open_app` directly.

## 17. Implementation/training sequence

1. Freeze the current contract and foundation tests.
2. Verify candidate checkpoint availability, license, tokenizer coverage, and package versions without changing production.
3. Implement an offline adapter around the selected encoder with one-pass logits and no routing integration.
4. Add deterministic derivation and calibration utilities against the existing contract.
5. Train the frozen-encoder linear-head baseline on a human-reviewed seed set.
6. Evaluate by language, family, clarification, OOD, and latency; compare against cached mBERT Phase 3C.
7. Fine-tune only if the frozen baseline shows a representation bottleneck.
8. Distill/augment only with filtered, provenance-tracked teacher examples.
9. Replay the sealed evaluation set and memory/latency checks before any shadow integration.

No step above authorizes production routing changes or model training in this phase.

## 18. Acceptance criteria for a future model

Acceptance must be reported per field and language, not as one arbitrary score. Require schema validity, no invalid enum output, no target invention on unresolved references, measured abstention/OOD behavior, capability-family accuracy, clarification precision/recall, grouped multilingual evaluation, warm latency distribution, startup/memory measurements, and regression comparison with the cached mBERT baseline.

The model is not ready for production control merely because it is faster. It must demonstrate that a conservative fallback is safer than the current route for uncertain cases and that deterministic commands remain unaffected.

## 19. Final architecture

```text
TRANSCRIPT
   ↓
exact deterministic fast-path check
   ↓ (if not handled)
CPU-resident multilingual cased encoder
   ↓ one forward pass
shared pooling/projection
   ↓
intent + capability + clarification/OOD heads
   ↓ calibrated logits
deterministic derivation of execution/context/escalation
   ↓
FastBrainDecision
   ↓ validator and conservative abstention
   ↓
shadow adapter first, then Orchestrator integration only after acceptance
   ├── fast_path / respond
   ├── Qwen agent + LangGraph
   ├── memory
   ├── research
   ├── clarify
   └── escalation policy → Gemini
```

## 20. Final recommendation

Build the first model adapter around `distilbert-base-multilingual-cased` as a CPU-only, one-forward-pass, multi-head classifier, while preserving the existing cached mBERT classifier as the immediate baseline. Train it only on a human-first, grouped, multilingual dataset; derive execution path, context requirements, and escalation deterministically; calibrate abstention; and keep it shadow-only until the sealed evaluation demonstrates a real improvement without exceeding the laptop’s memory and latency budget.

The next single implementation task is: create an offline, non-production encoder adapter that emits raw logits for the frozen Fast Brain contract and can be evaluated against the existing Phase 3C adapter—without downloading a checkpoint or changing routing.
