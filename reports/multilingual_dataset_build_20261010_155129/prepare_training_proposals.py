"""Generate new isolated trainer and unapplied source patches. Never train."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import difflib,json
from dataset_tools import ROOT
OUT=Path(__file__).resolve().parent
original=(ROOT/'benchmarks/fast_classifier/train_candidate.py').read_text(encoding='utf-8')
oldblock='''        capability_texts = {}
        for example in examples:
            for cap in example.capabilities:
                capability_texts[cap.name] = serialize_capability(
                    cap.name, cap.description, cap.required_arguments
                )
        capability_names = sorted(capability_texts)
        capability_vectors = encoder.encode(
            [capability_texts[name] for name in capability_names], batch_size=16
        )
        vector_by_capability = dict(zip(capability_names, capability_vectors))
        pair_features = []
        for example, request_vector in zip(examples, vectors):
            for cap in example.capabilities:
                pair_features.append(_pair_features(request_vector, vector_by_capability[cap.name]))
                pair_labels.append(int(cap.name == example.expected.selected_capability))
        pair_features = np.stack(pair_features)
'''
newblock='''        pair_features, pair_labels, _descriptor_texts = exact_pair_features(
            examples, vectors, encoder
        )
'''
assert original.count(oldblock)==1
safe=original.replace('dataset = load_dataset(dataset_path, split="train")','dataset = load_verified_training(dataset_path)',1).replace(oldblock,newblock)
safe=safe.replace('    process = psutil.Process()','    if artifact_path.exists() or artifact_path.with_suffix(".json").exists():\n        raise ValueError("Refusing to overwrite existing model artifacts")\n\n    process = psutil.Process()',1)
safe=safe.replace('capability_encoding: str = "joint_pair"','capability_encoding: str = "independent"',1).replace('default="joint_pair",','default="independent",',1)
safe=safe.replace('from benchmarks.fast_classifier.evaluate import BenchmarkExample, load_dataset','from benchmarks.fast_classifier.evaluate import BenchmarkExample, load_dataset\nfrom dataset_tools import load_verified_training, exact_pair_features',1)
safe=safe.replace('from assistant.orchestration.compact_adapter import (','import sys\nsys.path.insert(0, str(Path(__file__).resolve().parents[2]))\n\nfrom assistant.orchestration.compact_adapter import (',1)
safe=safe.replace('"""Train lightweight decision heads over the locally cached frozen encoder."""','"""Approval-gated isolated trainer. NOT RUN during dataset construction.\n\nRejects non-gold data before encoder loading; preserves per-example descriptors.\nThis is a new file; the original trainer and frozen artifact are unchanged.\n"""',1)
safe=safe.replace('DEFAULT_DATASET = ROOT / "benchmarks" / "fast_classifier" / "v2" / "dataset.jsonl"','DEFAULT_DATASET = Path(__file__).resolve().parent / "train.jsonl"')
safe=safe.replace('DEFAULT_ARTIFACT = ROOT / "benchmarks" / "fast_classifier" / "models" / "v2" / "compact_heads.joblib"','DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "future_approved_model" / "compact_heads.joblib"')
compile(safe,'train_verified_candidate.py','exec')
(OUT/'train_verified_candidate.py').write_text(safe,encoding='utf-8')
# Source proposal is not applied. It points at the reviewed helper in this immutable build.
proposed=original.replace('dataset = load_dataset(dataset_path, split="train")','dataset = load_verified_training(dataset_path)',1).replace(oldblock,newblock)
proposed=proposed.replace('from benchmarks.fast_classifier.evaluate import BenchmarkExample, load_dataset','from benchmarks.fast_classifier.evaluate import BenchmarkExample, load_dataset\nfrom reports.'+OUT.name+'.dataset_tools import load_verified_training, exact_pair_features',1)
patch=''.join(difflib.unified_diff(original.splitlines(keepends=True),proposed.splitlines(keepends=True),fromfile='a/benchmarks/fast_classifier/train_candidate.py',tofile='b/benchmarks/fast_classifier/train_candidate.py'))
(OUT/'proposed_training_patch.diff').write_text(patch,encoding='utf-8')
adapter=(ROOT/'assistant/orchestration/compact_adapter.py').read_text(encoding='utf-8')
# Avoid importing dataset_tools from adapter (circular). Inline the narrow pure resolver proposal.
resolver='''def _resolve_explicit_reference(context: Mapping[str, Any], kind: str) -> str | None:
    entries = context.get("resolved_references", [])
    if not isinstance(entries, list):
        return None
    values = {
        item.get("resolved_value") for item in entries
        if isinstance(item, Mapping) and item.get("kind") == kind
        and isinstance(item.get("resolved_value"), str)
        and item["resolved_value"].strip()
    }
    return next(iter(values)) if len(values) == 1 else None


'''
adapter_proposal=adapter.replace('resolved = _first_context_value(request.context) if resolved_probability >= 0.5 else None','resolved = _resolve_explicit_reference(request.context, kind) if resolved_probability >= 0.5 else None',1)
adapter_proposal=adapter_proposal.replace('def _first_context_value(value: Any) -> str | None:',resolver+'def _first_context_value(value: Any) -> str | None:',1)
compile(adapter_proposal,'compact_adapter_proposal','exec')
(OUT/'proposed_reference_patch.diff').write_text(''.join(difflib.unified_diff(adapter.splitlines(keepends=True),adapter_proposal.splitlines(keepends=True),fromfile='a/assistant/orchestration/compact_adapter.py',tofile='b/assistant/orchestration/compact_adapter.py')),encoding='utf-8')
(OUT/'approval_boundaries.md').write_text('''# Implementation and approval boundaries

Implemented only as new files: corrected candidate builder; full-context preservation; structured missing-name extraction; explicit Registry family/argument alignment; per-example capability-pair features; exact-schema two-reviewer gold gate; isolated trainer (not executed); tests; review queues and manifests.

Existing code, data, review ledgers, production routing and model artifacts are unchanged. Proposed source patches are review artifacts and were not applied. The isolated trainer rejects empty/unverified/insufficient-class data before encoder loading and refuses existing artifact destinations. It must not be run until a separate training experiment is approved.

The training proposal changes only the gold loader and independent descriptor feature branch. Integration should install the reviewed helper in the benchmark package before applying the proposal; its present report-directory import records the exact implementation under review. Existing train_candidate.py still lacks gold enforcement and must not be pointed at candidate files.

The reference patch is deliberately a proposal. Existing adapter first-scalar resolution remains unchanged. A legitimate runtime context producer would have to supply resolved_references independently of labels; this build does not inject gold resolved values into input context. Without a unique explicit reference, the proposal returns unresolved. This can change behavior and requires approved adapter/input-policy work. No partial resolver can recover arbitrary missing reference text; the exact gold reference spans and missing descriptions are preserved, and their incompatibility with existing deterministic output must not be disguised by relabeling.

No gold, native-language approval, adjudication or new final evaluation set can be manufactured by these code fixes. The blinded compact review packet needs exact-target approval; broader semantic grouping and fresh independent test collection remain human work.
''',encoding='utf-8')
print('Created isolated gated trainer and two unapplied source patch proposals; no training performed.')
