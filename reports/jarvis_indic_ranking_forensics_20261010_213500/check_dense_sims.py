import json
from pathlib import Path
import numpy as np

EXP4 = Path("D:/JARVIS/reports/classifier_v4_experiment_20261010_175140")
PRIOR = Path("D:/JARVIS/reports/autonomous_bert_training_20261010_161410")
FROZEN_100_DIR = Path("D:/JARVIS/reports/classifier_v4_frozen_100_eval_20261010_180014")

import sys
sys.path.insert(0, "D:/JARVIS")

from assistant.orchestration.decision import Capability, DecisionRequest
from assistant.orchestration.compact_adapter import (
    FrozenMultilingualEncoder,
    serialize_request,
    serialize_capability,
)

cache_manifest = json.loads((PRIOR / "feature_cache_manifest.json").read_text(encoding="utf-8"))
encoder = FrozenMultilingualEncoder(cache_manifest["encoder_snapshot"], 160)

c100 = [json.loads(l) for l in (FROZEN_100_DIR / "unseen_evaluation_corpus_100.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

target_ids = ["frozen100-013", "frozen100-015", "frozen100-019", "frozen100-023", "frozen100-038", "frozen100-041"]

for r in c100:
    if r["id"] in target_ids:
        req = DecisionRequest(r["utterance"], r["context"], tuple(Capability(c["name"], c["description"], tuple(c["required_arguments"])) for c in r["capabilities"]))
        texts = [serialize_request(req)] + [serialize_capability(c.name, c.description, c.required_arguments) for c in req.capabilities]
        vecs = encoder.encode(texts)
        q_v = vecs[0:1]
        c_v = vecs[1:]
        sims = np.dot(c_v / np.linalg.norm(c_v, axis=1, keepdims=True), (q_v / np.linalg.norm(q_v)).T).flatten()
        print(f"{r['id']} ({r['language']}) '{r['utterance']}':")
        for i, c in enumerate(req.capabilities):
            print(f"   {c.name}: raw_dense_sim = {sims[i]:.4f}")
