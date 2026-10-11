import re
from pathlib import Path

text = "टैक्स रसीद की पीडीएफ फाइल ढूंढो"
te_text = "కాలేజ్ సర్టిఫికెట్ పిడిఎఫ్ ఫైల్ వెతుకు"

h_re = re.findall(r"\w+", text)
h_sp = [w.strip(".,?!:;\"'") for w in text.split()]
t_re = re.findall(r"\w+", te_text)
t_sp = [w.strip(".,?!:;\"'") for w in te_text.split()]

lines = [
    f"Hindi re.findall: {h_re}",
    f"Hindi split(): {h_sp}",
    f"Telugu re.findall: {t_re}",
    f"Telugu split(): {t_sp}",
]

Path("D:/JARVIS/reports/jarvis_indic_ranking_forensics_20261010_213500/split_results.txt").write_text("\n".join(lines), encoding="utf-8")
print("Done")
