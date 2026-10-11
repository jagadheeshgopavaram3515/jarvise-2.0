import json
from pathlib import Path

EXP_DIR = Path("D:/JARVIS/reports/jarvis_indic_ranking_forensics_20261010_213500")
p_v5 = json.load((EXP_DIR / "v5_predictions_100.json").open("r", encoding="utf-8"))
p_v7 = json.load((EXP_DIR / "v7_predictions_100.json").open("r", encoding="utf-8"))

p_v5_dict = {p["id"]: p for p in p_v5}

FIELDS = ("intent", "execution_path", "request_modes", "selected_capability", "needs_clarification", "needs_escalation", "missing_information", "context_references")

lines = []
for p7 in p_v7:
    pid = p7["id"]
    p5 = p_v5_dict[pid]
    if p5["is_exact"] and not p7["is_exact_frozen"]:
        lines.append(f"Regressed ID: {pid} '{p7['utterance']}'")
        exp = p7["expected_frozen"]
        pred5 = p5["predicted"]
        pred7 = p7["predicted"]
        failed_fields_7 = [k for k in FIELDS if (sorted(pred7[k]) != sorted(exp[k]) if k == "request_modes" else pred7[k] != exp[k])]
        lines.append(f"   Failed in v7: {failed_fields_7}")
        lines.append(f"   Expected:     {{ {', '.join(f'{k}: {exp[k]}' for k in failed_fields_7)} }}")
        lines.append(f"   Predicted v7: {{ {', '.join(f'{k}: {pred7[k]}' for k in failed_fields_7)} }}")
        lines.append(f"   Predicted v5: {{ {', '.join(f'{k}: {pred5.get(k)}' for k in failed_fields_7)} }}")
        lines.append("-" * 50)

out = EXP_DIR / "exact_regressions_trace.txt"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote exact regressions trace to {out}")

