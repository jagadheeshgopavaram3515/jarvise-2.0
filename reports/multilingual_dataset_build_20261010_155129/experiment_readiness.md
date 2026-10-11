# Experiment readiness

**BLOCKED_PENDING_HUMAN_REVIEW**

290 canonical nonsealed candidate records in 34 conservatively grouped components; 250 provisional candidate train and 40 provisional candidate validation. All are pending_review/synthetic_unverified. Verified supervised TRAIN/VALIDATION/TEST files are intentionally empty; zero labels are eligible for fitting and zero new final-test records are locked. Empty-file schema handling does not claim successful fitting.

33 offline regression tests passed. Original sources/model/user changes are unchanged. Corrected builder, pair features and gold gate exist as new isolated implementation. Existing trainer/reference adapter patches are proposals only, not applied. No training or encoder inference ran.

Needed: exact ClassifierDecision approvals by two qualified independent humans; separate adjudication for disagreements and multilingual meanings; broader semantic/group verification; diverse coverage of absent/rare classes; approved runtime reference/span/description policy; a fresh independent reviewed test locked before tuning. No reviewer identities are confirmed and this build does not wait for them or invent approvals.

Do not pass candidate_train.jsonl to the legacy trainer. New train_verified_candidate.py refuses empty/non-gold/insufficient-class data before loading the encoder and refuses overwriting existing artifacts. Any next training experiment requires separate approval; no accuracy gain is claimed.
