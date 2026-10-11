# Implementation and approval boundaries

Implemented only as new files: corrected candidate builder; full-context preservation; structured missing-name extraction; explicit Registry family/argument alignment; per-example capability-pair features; exact-schema two-reviewer gold gate; isolated trainer (not executed); tests; review queues and manifests.

Existing code, data, review ledgers, production routing and model artifacts are unchanged. Proposed source patches are review artifacts and were not applied. The isolated trainer rejects empty/unverified/insufficient-class data before encoder loading and refuses existing artifact destinations. It must not be run until a separate training experiment is approved.

The training proposal changes only the gold loader and independent descriptor feature branch. Integration should install the reviewed helper in the benchmark package before applying the proposal; its present report-directory import records the exact implementation under review. Existing train_candidate.py still lacks gold enforcement and must not be pointed at candidate files.

The reference patch is deliberately a proposal. Existing adapter first-scalar resolution remains unchanged. A legitimate runtime context producer would have to supply resolved_references independently of labels; this build does not inject gold resolved values into input context. Without a unique explicit reference, the proposal returns unresolved. This can change behavior and requires approved adapter/input-policy work. No partial resolver can recover arbitrary missing reference text; the exact gold reference spans and missing descriptions are preserved, and their incompatibility with existing deterministic output must not be disguised by relabeling.

No gold, native-language approval, adjudication or new final evaluation set can be manufactured by these code fixes. The blinded compact review packet needs exact-target approval; broader semantic grouping and fresh independent test collection remain human work.
