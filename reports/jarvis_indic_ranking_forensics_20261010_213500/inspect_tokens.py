import json
import re
from pathlib import Path

FROZEN_100_DIR = Path("D:/JARVIS/reports/classifier_v4_frozen_100_eval_20261010_180014")
c100 = [json.loads(l) for l in (FROZEN_100_DIR / "unseen_evaluation_corpus_100.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

target_ids = ["frozen100-015", "frozen100-019", "frozen100-023", "frozen100-028", "frozen100-038", "frozen100-041", "frozen100-042", "frozen100-045"]

lines = []
for cid in target_ids:
    r = [x for x in c100 if x["id"] == cid][0]
    tokens = set(re.findall(r"\w+", r["utterance"].lower()))
    lines.append(f"{cid} ({r['language']}) '{r['utterance']}':\n  tokens: {tokens}\n  expected: {r['expected']['selected_capability']}")

out_file = Path("D:/JARVIS/reports/jarvis_indic_ranking_forensics_20261010_213500/tokens_inspection.txt")
out_file.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved token inspection to {out_file}")

