import re
from pathlib import Path

text = "टैक्स रसीद की पीडीएफ फाइल ढूंढो"
te_text = "కాలేజ్ సర్టిఫికెట్ పిడిఎఫ్ ఫైల్ వెతుకు"
dinni = "దీన్ని వెంటనే డిలీట్ చెయ్యి"

pattern = re.compile(r'[^\s.,?!:;\'"()\[\]{}]+')

lines = [
    f"Hindi: {pattern.findall(text)}",
    f"Telugu: {pattern.findall(te_text)}",
    f"Dinni: {pattern.findall(dinni)}",
]

Path("D:/JARVIS/reports/jarvis_indic_ranking_forensics_20261010_213500/unicode_word_test.txt").write_text("\n".join(lines), encoding="utf-8")
print("Done")

