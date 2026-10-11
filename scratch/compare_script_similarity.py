"""Analyze dense cosine similarity between native and transliterated queries against capabilities."""
import json
import joblib
import numpy as np
from pathlib import Path

ROOT = Path('D:/JARVIS')
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'

import sys
sys.path.insert(0, str(ROOT))
from assistant.orchestration.compact_adapter import FrozenMultilingualEncoder, serialize_capability

cache_manifest = json.loads((PRIOR / 'feature_cache_manifest.json').read_text(encoding='utf-8'))
encoder = FrozenMultilingualEncoder(cache_manifest['encoder_snapshot'], 160)

cap_text = serialize_capability("open_folder", "Open a directory or folder in file explorer", ("path",))
cap_vec = encoder.encode([cap_text])[0]
cap_norm = cap_vec / np.linalg.norm(cap_vec)

queries = [
    ("en", "Open downloads folder"),
    ("hi-Latn", "downloads folder open karo"),
    ("hi", "डाउनलोड्स फ़ोल्डर खोलो"),
    ("te-Latn", "downloads folder open cheyyi"),
    ("te", "నా డౌన్‌లోడ్స్ ఫోల్డర్ ఓపెన్ చెయ్యి"),
    ("ta-Latn", "downloads folder open pannu"),
    ("ta", "டவுன்லோட்ஸ் கோப்புறையை திறக்கவும்"),
]

q_vecs = encoder.encode([q[1] for q in queries])
for (lang, text), vec in zip(queries, q_vecs):
    norm = vec / np.linalg.norm(vec)
    sim = float(np.dot(norm, cap_norm))
    print(f"{lang:<10} | Sim: {sim:.4f} | Utterance: {text}")

