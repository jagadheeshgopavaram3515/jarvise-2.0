import json

diags = json.load(open(r"D:\JARVIS\reports\jarvis_indic_ranking_forensics_20261010_213500\capability_failure_diagnostics.json", encoding="utf-8"))

lines = []
lines.append("=== 12 RANKING COMPETITION CASES ===")
for d in diags:
    if d["failure_stage"] == "ranking_competition":
        lines.append(f"{d['id']} [{d['language']}/{d['script']}]")
        lines.append(f"  Utterance: {d['utterance']}")
        lines.append(f"  Exp: {d['expected_capability']} vs Pred: {d['predicted_capability']}")
        lines.append(f"  Scores: {d['candidate_scores']}")
        lines.append(f"  Mechanism: {d['mechanism']}")
        lines.append("-" * 50)

lines.append("\n=== 9 FALSE ABSTENTION CASES ===")
for d in diags:
    if d["failure_stage"] == "abstention_failure_false_abstention":
        lines.append(f"{d['id']} [{d['language']}/{d['script']}]")
        lines.append(f"  Utterance: {d['utterance']}")
        lines.append(f"  Exp: {d['expected_capability']} vs Pred: {d['predicted_capability']}")
        lines.append(f"  Scores: {d['candidate_scores']}")
        lines.append(f"  Mechanism: {d['mechanism']}")
        lines.append("-" * 50)

lines.append("\n=== 6 FALSE ACTION CASES ===")
for d in diags:
    if d["failure_stage"] == "abstention_failure_false_action":
        lines.append(f"{d['id']} [{d['language']}/{d['script']}]")
        lines.append(f"  Utterance: {d['utterance']}")
        lines.append(f"  Exp: {d['expected_capability']} vs Pred: {d['predicted_capability']}")
        lines.append(f"  Scores: {d['candidate_scores']}")
        lines.append(f"  Mechanism: {d['mechanism']}")
        lines.append("-" * 50)

with open(r"D:\JARVIS\reports\jarvis_indic_ranking_forensics_20261010_213500\detailed_failure_breakdown.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Saved detailed_failure_breakdown.txt")

