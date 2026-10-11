import json
from pathlib import Path
from transformers import AutoTokenizer

EXP_DIR = Path("D:/JARVIS/reports/jarvis_indic_ranking_forensics_20261010_213500")
PRIOR = Path("D:/JARVIS/reports/autonomous_bert_training_20261010_161410")
cache_manifest = json.loads((PRIOR / "feature_cache_manifest.json").read_text(encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(cache_manifest["encoder_snapshot"])

words = [
    # Hindi
    ("ढूंढो", "Hindi find"),
    ("तलाशो", "Hindi search"),
    ("खोलो", "Hindi open"),
    ("हटाओ", "Hindi delete"),
    ("बंद करो", "Hindi close"),
    ("फाइल", "Hindi file"),
    ("ऐप", "Hindi app"),
    # Telugu
    ("వెతుకు", "Telugu find/search"),
    ("తెరవు", "Telugu open"),
    ("డిలీట్", "Telugu delete"),
    ("ఓపెన్", "Telugu open loanword"),
    ("ఫైల్", "Telugu file"),
    ("యాప్", "Telugu app"),
    # Tamil
    ("தேடு", "Tamil search"),
    ("திற", "Tamil open"),
    ("அழி", "Tamil delete"),
    ("கோப்பு", "Tamil file"),
    ("செயலி", "Tamil app")
]

lines = []
for w, desc in words:
    subwords = tok.tokenize(w)
    lines.append(f"{w} ({desc}): {len(subwords)} subwords -> {subwords}")

out_path = EXP_DIR / "subword_fragmentation.txt"
out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved subword fragmentation to {out_path}")

