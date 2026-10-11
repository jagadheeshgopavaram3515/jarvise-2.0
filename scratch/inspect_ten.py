import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path('D:/JARVIS')
PRIOR = ROOT / 'reports/autonomous_bert_training_20261010_161410'
OUT = ROOT / 'reports/classifier_completion_20261010_163425'

val = [json.loads(l) for l in (PRIOR / 'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
bm = json.loads((OUT / 'best_metrics.json').read_text(encoding='utf-8'))

ten_ids = [
    'dev_memory_project-te-Latn',
    'dev_memory_project-hi-Latn',
    'dev_memory_project-ta-Latn',
    'dev_memory_project-en',
    'en-memory-001',
    'hi-codeswitch-memory-001',
    'dev_clarify_unresolved-hi-Latn',
    'dev_clarify_unresolved-en',
    'ctx-hologram-001',
    'en-music-001'
]

for p in bm['predictions']:
    if p['original_id'] in ten_ids:
        print("="*70)
        print(f"ID: {p['original_id']}")
        rec = next(r for r in val if r['original_id'] == p['original_id'])
        print(f"Utterance: {rec['utterance']}")
        print(f"Context: {rec['context']}")
        print("Expected:")
        for k, v in p['expected'].items():
            print(f"  {k}: {v}")
        print("Prediction (experimental v2):")
        for k, v in p['prediction'].items():
            if p['expected'].get(k) != v:
                print(f"  * DIFF * {k}: pred={v} != exp={p['expected'].get(k)}")
            else:
                print(f"           {k}: {v}")

