import json
from pathlib import Path
import joblib

EXP_DIR = Path("D:/JARVIS/reports/jarvis_indic_ranking_forensics_20261010_213500")
EXP4 = Path("D:/JARVIS/reports/classifier_v4_experiment_20261010_175140")
PRIOR = Path("D:/JARVIS/reports/autonomous_bert_training_20261010_161410")

import sys
sys.path.insert(0, "D:/JARVIS")
sys.path.insert(0, str(EXP_DIR))

from assistant.orchestration.decision import Capability, DecisionRequest
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder
from experimental_adapter_v7 import ExperimentalV7Adapter
from slot_contract_resolver import SlotContractResolver

cache_manifest = json.loads((PRIOR / "feature_cache_manifest.json").read_text(encoding="utf-8"))
encoder = FrozenMultilingualEncoder(cache_manifest["encoder_snapshot"], 160)
artifact = joblib.load(EXP4 / "best_candidate/compact_heads.joblib")

ad = ExperimentalV7Adapter.__new__(ExperimentalV7Adapter)
ad.artifact = artifact
ad.encoder = encoder
ad._adapter_id = "test"
ad.inference_count = 0
ad.last_inference_metrics = {}
ad.contract_resolver = SlotContractResolver()

caps = (
    Capability("open_folder", "Open a specified folder or directory in file explorer.", ("path",)),
    Capability("find_file", "Find local files or documents matching a query.", ("query",)),
)
req = DecisionRequest("Open my documents directory please", {}, caps)
pred = ad.predict(req)
print("pred capability:", pred.get("selected_capability"))
print("pred execution_path:", pred.get("execution_path"))
print("diag:", ad.last_diagnostics)

