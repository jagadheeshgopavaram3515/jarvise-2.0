"""Complete reports from audit artifacts and read-only source metadata."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,csv,collections,ast,hashlib,datetime
ROOT=Path('D:/JARVIS'); OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
def load(n): return json.loads((OUT/n).read_text(encoding='utf-8'))
def write(n,v): (OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def readcsv(n): return list(csv.DictReader((OUT/n).open(encoding='utf-8-sig')))
def csvwrite(n,rs):
 with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(rs[0])); w.writeheader(); w.writerows(rs)
s=load('analysis_summary.json'); integrity=load('integrity_checks.json'); prov=load('source_provenance.json'); manifest=load('audit_manifest.json')
def source(p):
 b=(ROOT/p).read_bytes(); manifest['inputs'][p]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}; return b.decode('utf-8-sig')
def records(p): return [json.loads(l) for l in source(p).splitlines() if l.strip()]
v3='benchmarks/fast_classifier/v3/dataset.jsonl'; v4='benchmarks/fast_classifier/v4/dataset.jsonl'; hist='benchmarks/fast_classifier/fast_brain_v1/dataset.jsonl'
rs=[]
for r in records(v3):
 if r['split']!='test': rs.append(r)
histmeta=[{k:r.get(k) for k in ('id','original_id','split','group_id','review_status','language_group','script_group','provenance')} for r in records(hist)]
reviewmeta=[{k:r.get(k) for k in ('record_id','reviewer_id','reviewer_decision','adjudication_status','reviewed_at')} for r in records('benchmarks/fast_classifier/fast_brain_v1/review_decisions.jsonl')]
tids=set(prov['model_metadata']['training_example_ids']); suspect=readcsv('suspicious_labels.csv'); contexts=readcsv('context_quality.csv'); distributions=readcsv('label_distribution.csv')
def add(r,f,label,typ,evidence,severity='HIGH'):
 suspect.append(dict(record_id=r['id'],source_file=v3,split=r['split'],training_overlap=r['id'] in tids,language=r['language'],script=next((x['script'] for x in suspect if x['record_id']==r['id'] and x['source_file']==v3),''),original_utterance=r['utterance'],target_field=f,current_label=json.dumps(label,ensure_ascii=False),issue_type=typ,severity=severity,evidence=evidence,suggested_review_action='Review against the original contract/Registry; retain the source annotation.',review_status=r.get('review_status','')))
reg={}
tree=ast.parse(source('assistant/tools/registry.py'))
for node in ast.walk(tree):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='ToolSpec':
  kw={x.arg:x.value for x in node.keywords}
  if 'name' in kw and 'parameters' in kw:
   try: reg[ast.literal_eval(kw['name'])]=ast.literal_eval(kw['parameters']).get('required',[])
   except (ValueError,TypeError): pass
registry_mismatches=[]
builder_registry_mismatches=[]
builder_tree=ast.parse(source('benchmarks/fast_classifier/prepare_v4.py'))
for n in builder_tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CAPS' for t in n.targets):
  builder_caps=ast.literal_eval(n.value)
  builder_registry_mismatches=[{'capability':k,'builder_required':v[1],'registry_required':reg[k]} for k,v in builder_caps.items() if k in reg and sorted(v[1])!=sorted(reg[k])]
for r in rs:
 for cap in r['capabilities']:
  if cap['name'] in reg and sorted(cap['required_arguments'])!=sorted(reg[cap['name']]):
   ev=f"Dataset required_arguments={cap['required_arguments']}; current Registry required={reg[cap['name']]}."
   add(r,'capabilities.required_arguments',cap,'registry_argument_mismatch',ev)
   registry_mismatches.append({'id':r['id'],'split':r['split'],'capability':cap['name'],'dataset_required':cap['required_arguments'],'registry_required':reg[cap['name']]})
 for f,items,key in [('missing_information',r['expected']['missing_information'],'name'),('reference_kind',r['expected']['context_references'],'kind')]:
  if items and items[0][key] not in integrity['heads'][f]['classes']:
   add(r,f,items[0][key],'unseen_head_class',f"Label is absent from fitted head classes {integrity['heads'][f]['classes']}.")
# Show absent classes explicitly for train/dev; observed nonsealed labels and fitted classes only.
for f in ('missing_information','reference_kind','selected_capability_contract'):
 universe={x['class_label'] for x in distributions if x['source_file']==v3 and x['target_field']==f}
 if f=='selected_capability_contract': universe.update(reg)
 for split in ('train','dev'):
  present={x['class_label'] for x in distributions if x['source_file']==v3 and x['split']==split and x['target_field']==f}
  for cl in sorted(universe-present): distributions.append(dict(source_file=v3,split=split,target_field=f,class_label=cl,count=0,labelled_total=216 if split=='train' else 78,proportion=0,rare_lt_5=True,status='ABSENT_CLASS_FROM_NONSEALED_OR_REGISTRY'))
# Actual training cache discrepancy, rather than only potential variants.
from assistant.orchestration.compact_adapter import serialize_capability
last={}
for r in rs:
 if r['split']=='train':
  for c in r['capabilities']: last[c['name']]=serialize_capability(c['name'],c['description'],c['required_arguments'])
cache_mismatch=collections.Counter(); mismatch_ids=set()
for r in rs:
 if r['split']=='train':
  for c in r['capabilities']:
   if serialize_capability(c['name'],c['description'],c['required_arguments'])!=last[c['name']]: cache_mismatch[c['name']]+=1; mismatch_ids.add(r['id'])
integrity['descriptor_training_inference_actual_mismatch']={'capability_pairs':dict(cache_mismatch),'unique_training_records':len(mismatch_ids),'total_pairs':sum(cache_mismatch.values())}
integrity['registry_argument_mismatches_nonsealed']=registry_mismatches
integrity['v4_builder_registry_descriptor_mismatches_static']=builder_registry_mismatches
integrity['sealed_policy']='Automated label analysis excluded original test and sealed_candidate labels. No sealed outcomes were analyzed or used for tuning. An initial source preview displayed one v3 sealed record; it was not used in analysis. Historical evaluation prose was read incidentally; its test outcomes were not used.'
integrity['input_hashes_unchanged_after_analysis']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==info['sha256'] for p,info in manifest['inputs'].items()}
integrity['split_manifest_hash_checks']={f'v{v}':json.loads(source(f'benchmarks/fast_classifier/v{v}/split_manifest.json'))['dataset_sha256']==manifest['inputs'][f'benchmarks/fast_classifier/v{v}/dataset.jsonl']['sha256'] for v in (2,3,4)}
integrity['historical_source_hash_matches']=prov['historical_analysis']['records_imported']==359
integrity['limitations']=['Independent annotation correctness not established by schema validity or reviewer identity alone.','No local encoder revision/weight fingerprint or saved source-code-at-training snapshot.','No semantic encoder was run; character-similarity triage can miss translations, template variants and paraphrases.','Sealed labels/distributions excluded, including historical copies of original sealed records; class tables are partial for the historical queue.','Early source preview included one sealed record; zero incidental sealed visibility cannot be claimed.']
suspect=list({(r['source_file'],r['record_id'],r['target_field'],r['issue_type'],r['current_label']):r for r in suspect}.values())
csvwrite('suspicious_labels.csv',suspect); csvwrite('label_distribution.csv',distributions); write('integrity_checks.json',integrity)
overlap_by_split=collections.Counter(r['split'] for r in histmeta if r['original_id'] in tids)
reviewers=collections.Counter(r['reviewer_id'] for r in reviewmeta)
review_summary={'recorded_decisions':s['review_decisions'],'unique_record_ids':len({r['record_id'] for r in reviewmeta}),'reviewer_ids':dict(reviewers),'never_reviewed':359-len({r['record_id'] for r in reviewmeta}),'skipped':13,'accepted_or_corrected':18,'ambiguous_decisions':0,'rejected_decisions':0,'eligible_nonsealed_labels':s['eligible_nonsealed_labels'],'eligible_later_reviews_of_training_records':s['training_records_with_eligible_later_review'],'verified_original_full_contract_training_labels':0,'note':'These counts measure recorded human decisions, not independently adjudicated semantic correctness or the labels used during original training.'}
prov['review_reconciliation']=review_summary; prov['historical_training_overlap_by_current_split']=dict(overlap_by_split); write('source_provenance.json',prov)
types=collections.Counter(r['issue_type'] for r in suspect)
lang=readcsv('language_coverage.csv'); trainlang={}
for r in lang:
 if r['source_file']==v3 and r['split']=='train' and r['target_field']=='execution_path': trainlang[(r['language'],r['observed_script'])]=trainlang.get((r['language'],r['observed_script']),0)+int(r['count'])
pathrows=[x for x in distributions if x['source_file']==v3 and x['target_field']=='execution_path']
paths=sorted({r['class_label'] for r in pathrows}); class_table='| Execution path | TRAIN | DEV (validation) |\n|---|---:|---:|\n'
for c in paths:
 nums=[next(int(r['count']) for r in pathrows if r['split']==sp and r['class_label']==c) for sp in ('train','dev')]
 class_table+=f'| {c} | {nums[0]} ({nums[0]/216:.1%}) | {nums[1]} ({nums[1]/78:.1%}) |\n'
lang_table='| Recorded language | Observed script | Training rows |\n|---|---|---:|\n'+''.join(f'| {l} | {sc} | {n} |\n' for (l,sc),n in sorted(trainlang.items()))
head_table='| Head | Target source | Feature shape |\n|---|---|---|\n| execution_path | expected.execution_path | 768 request features |\n| 4 request_modes | membership in expected.request_modes | 768 request features each |\n| clarification / escalation | respective expected boolean | 768 request features |\n| missing_information | first expected item name, else __none__ | 768 request features |\n| reference_kind | first expected reference kind, else __none__ | 768 request features |\n| reference_resolved | first reference resolved_value is not null | 768 request features |\n| capability_matcher | descriptor name == expected.selected_capability | 3072 pair features |\n'
report=f'''# JARVIS frozen BERT dataset forensic audit

Final readiness: **BLOCKED_DATA_QUALITY**. The current heads have verifiable dataset provenance, but the original labels are synthetic and lack independent full-contract verification. The dataset does not substantiate a meaningful >=90% generalization claim. No model was trained, no inference/network/provider call was made, and no existing project file was intentionally written. Only this report directory was created.

| Category | Status | Basis |
|---|---|---|
| Exact dataset/head provenance | PASS | v3 SHA-256 and all 216 training IDs match loaded joblib and JSON metadata |
| Encoder revision/historical source snapshot | NOT_VERIFIED | Model ID recorded; resolved encoder revision, weight hash and source-code-at-training hash absent |
| Label/schema quality | FAIL | No independently verified original training labels; later taxonomy differs; missing trained classes |
| Duplicate and split integrity | FAIL | Confirmed v4 train/test text duplication; historical re-split preserves that duplicate across train/development |
| Class balance and independent coverage | FAIL | Only 22 train scenario groups; sparse capabilities; missing-information and reference classes absent |
| Multilingual quality | WARNING | Native and Latin examples exist; no adequate independent native-speaker verification |
| Context/input fidelity | FAIL | Consolidation discards actual input context/capability descriptors; reference resolver chooses unrelated first scalar |
| Features/preprocessing | FAIL | Name-keyed descriptor overwrite creates training/inference differences |
| Human review traceability | WARNING | 31 ledger entries conflict with stale zero-review summary; 18 accept/correct entries use newer contract |
| Sealed-test label distribution | NOT_VERIFIED | Labels and outcomes excluded from analysis; only authorized IDs/text/group/split/hash metadata used |
| Read-only source integrity | PASS | All hashed inputs unchanged after analysis; no existing validation runners with write side effects executed |

## Verified source-to-head flow

v1 seed has 28 rows. `prepare_v2.py` produces 228 rows (160 train, 50 dev, 18 test) through scenario synthesis and prior-label adjudication. `prepare_v3.py` derives those rows, changes metadata/limited historical annotations, and adds 84 manually authored synthetic examples: 56 train and 28 dev. V3 has **312 rows: 216 TRAIN, 78 DEV, 18 sealed TEST**. DEV is the validation/model-selection split; no distinct validation split exists.

`train_candidate.py:train()` invokes `load_dataset(dataset_path, split="train")`. The loader parses every line before filtering, including test records, but fitting only consumes the selected train rows. Thus the training loader has unnecessarily broad sealed-label visibility; ID/hash evidence does not show test targets were fitted. No status gate excludes pending/synthetic examples. All 216 training rows are explicitly synthetic candidates pending human review.

Actual artifact: `benchmarks/fast_classifier/models/v3/compact_heads.joblib`; adapter `compact-mbert-frozen-v3`; source `benchmarks/fast_classifier/v3/dataset.jsonl`; SHA-256 `{prov['model_metadata']['dataset_sha256']}`. The dataset hash, IDs, split membership, JSON metadata and loaded artifact agree. All 216 IDs are listed in `source_provenance.json`. Runtime fitting history and encoder revision cannot be independently reconstructed from IDs alone.

{head_table}
Features use `nlptown/bert-base-multilingual-uncased-sentiment` via AutoModel, ignoring the sentiment classification head. Tokenizer is local-files-only; max_length=160, padding/truncation enabled; attention-mask mean pooling followed by L2 normalization. Forward calls use eval/inference_mode. Only LogisticRegression `.fit()` optimizes parameters: balanced class weights, C=4, seed=42, lbfgs, max_iter=2000. No encoder fine-tuning, transliteration or dataset text normalization appears in this training path.

Actual encoding is **independent**, despite the current trainer's default of joint_pair. Request input is `User request: <utterance>\\nContext: <sorted JSON truncated to 2000 characters>`. Capabilities are encoded separately. Pair vector concatenates request, capability, absolute difference and elementwise product. No gold fields or IDs are serialized. No saved embedding cache exists in this trainer, so stale persistent-cache checking is not applicable; training arrays are ephemeral. Capability descriptors form a name-keyed in-memory cache, creating the discrepancy below. Encoder token truncation effects were not measured because no encoder/tokenizer was instantiated.

Intent, reference span/value, missing-information description, evidence and confidence are partly deterministic outputs, not independently learned heads. The older ClassifierDecision only enums request_modes; intent/path/reference kind/missing name are nonempty strings, not closed enums. The newer FastBrainDecision family/path enums must not be substituted into this contract. In particular `qwen_agent`, `respond` and `fast_path` in later reviews are not the compact model's seven semantic path classes. No exact intent-composition mismatch was found in nonsealed v3 records; the initial concern was resolved by checking actual targets.

Production source has no import/call of CompactMultilingualAdapter outside its own module; existing instantiations are offline evaluation runners. Its actual input contract is DecisionRequest(user_text, context mapping, Capability tuples). This is source wiring evidence, not proof about a running/custom-injected service. The IDE's Qwen/LoRA evaluation plan describes a separate end-to-end task; no Qwen/LoRA evaluation was performed here.

## The 359-record benchmark and human reviews

`prepare_v4.py` retains 216 train and 78 dev rows and supplies 140 test rows, for 434 total. `build_fast_brain_dataset.py` imports v4 into **359 historical_synthetic candidates**, quarantining 75 records. It then re-splits by declared group into **239 train, 57 development, 63 sealed_candidate**. This queue is a mixture of original train/dev/test lineage, not the exact data fitted by compact v3. The older v3 test IDs are not the newly authored v4 test set. Review queue and adjudication artifacts are workflow artifacts, not additional independent examples.

Review ledger: **15 accept, 3 correct, 13 skip**, 31 distinct records, one recorded reviewer identity. **328 records have no review decision**; 13 skips remain unresolved. All 359 source records still say pending_review/historical_synthetic, while the ledger separately holds 18 accept/correct decisions. That is preserved history, not grounds for overwriting either file. The saved review_analysis.json reports zero reviewed records and is stale. There are no recorded reject/ambiguous decisions in this ledger; zero is not evidence that ambiguity is absent. The 75 builder rejections are import/schema quarantine, not human rejections.

Of the recorded later accept/correct decisions, 12 lie in the nonsealed analysis scope; 11 refer to original training IDs. The other six accept/correct decisions are protected by candidate/original-test lineage and their labels were not inspected. **Zero original full ClassifierDecision training labels are established as independently verified.** Eighteen later recorded human decisions are not eighteen independently adjudicated original labels. Two-reviewer/native-speaker correctness, naturalness and translation preservation remain NOT_VERIFIED.

## Concrete high-severity findings

1. **Clarification records vanish during consolidation.** `normalize()` casts each structured missing-information object with `str(x)` instead of extracting its name; `FastBrainRecord.validate()` rejects these strings against the actual MissingInformation enum. All 75 quarantine entries report unknown missing-information labels; accepted queue metadata shows no clarification-positive examples. Example nonsealed source IDs: `synth_ambiguous_open-en-01`, `train_clarify_missing_target-en`, `train_clarify_search-en`. The 359 queue is therefore biased by import failure, not evidence that clarification is unnecessary. No sealed label was used to establish this mechanism.
2. **Capability-family builder omissions.** Real Registry actions such as open_folder, recent_files, grep_code and run_project_tests are not handled explicitly by `capability()`. Categories can incidentally rescue some examples; {types.get('builder_capability_omission',0)} nonsealed records with these actions actually map to none. `train_open_downloads-en` is an inspectable example. These are bugs in the later queue builder, not proof of bad original compact labels.
3. **Training/inference descriptor discrepancy.** Five capability names have multiple descriptor strings in training. The independent trainer stores the last descriptor for each name, whereas inference embeds each request's actual descriptor. **{len(mismatch_ids)} of 216 training requests** contain a differing descriptor; **{sum(cache_mismatch.values())} training capability pairs** differ. Counts by name: `{dict(cache_mismatch)}`. The CSV flags all occurrences of variant names as potential overwrites; this measured subset establishes the actual mismatch.
4. **Unlearnable label classes.** Training missing_information classes are only __none__/target. DEV includes prior_request (20) and query (1), absent from the fitted head. Training reference_kind is only __none__/item/prior_conversation; DEV also includes prior_request (20), folder, project, search_result and topic (one each). Those predictions cannot be emitted by the fitted heads. The older contract accepts these strings; this is coverage failure, not invalid enum usage.
5. **Deterministic reference mismatch.** {types.get('resolver_first_scalar_mismatch',0)} nonsealed v3 records have annotated resolved_value differing from `_first_context_value(context)`. Training `synth_recall_trip-en-01` annotates the decision but traversal chooses the topic name. There are {types.get('unreachable_exact_reference_span',0)} fixed reference-span mismatches and {types.get('unreachable_exact_missing_description',0)} fixed description mismatches. Correct gold labels can be incompatible with implementation; these are not model-disagreement-based relabel claims.
6. **Registry descriptor mismatches in the later builder.** {len(registry_mismatches)} nonsealed v3 descriptor occurrences disagree with current registered required arguments. Static inspection of v4 CAPS finds {len(builder_registry_mismatches)} builder mismatches: `{builder_registry_mismatches}`. Registry `search_files` requires keyword, not query; list_directory/run_project_tests have no required parameters, unlike the v4 builder. This conclusion uses source configuration, not sealed labels. Registry-at-training provenance is not recorded. See integrity_checks.json.
7. **Context lost and metadata oversimplified.** All consolidated records lack the original context and capability arrays; context_resolvable is merely bool(original context), which does not prove resolution. Original IDs allow reconstruction from v4, but the standalone queue does not reproduce inference inputs. {types.get('context_requirement_unsubstantiated',0)} nonsealed later reviews specify active_task while original context is empty; this is human-review triage, not a rewritten decision. {types.get('code_switch_not_mixed_script',0)} nonsealed queue examples say mixed script although their text is Latin-only. clean/noise metadata is defaulted by the builder, not measured STT robustness.

## Duplicates, related examples and split contamination

No duplicate IDs were found in any source dataset. V3 has one exact/normalized duplicate pair, entirely in test; label comparison is NOT_VERIFIED. No exact/normalized TRAIN–DEV duplicate or declared-group crossing was found in v3. Training has no exact utterance duplicates. This does not prove semantic independence: source scenario templates, translations and small topic families are correlated.

V4 has one **confirmed exact TRAIN–TEST pair**: `train_open_downloads-en` (train) and `v4_open_downloads-en` (test), both "Open my Downloads folder." One contaminated test text is **1/140 = 0.71%** of test; 2/434 = 0.46% of dataset rows belong to that pair. Test target agreement was not checked. The later queue carries this pair into `hist-2ab4a4ab05600583` (train) and `hist-35138bdc0c6c8e35` (development): 2/359 = 0.56% of queue rows and 1/57 = 1.75% of development. Different declared group IDs conceal the duplication from group-only validation.

Same-script character triage finds **7 uncertain cross-split pairs in v4** (5 TRAIN–TEST, 2 DEV–TEST), **5 in the historical queue**, and zero in v3 at similarity >=0.85. Examples include Python-docs URL variants, close-Notepad transliterations, Downloads translations and repeated unresolved-open forms. They are candidates for semantic review, not confirmed label errors. Cross-language translation leakage and minor-substitution template families are NOT_VERIFIED by this character method. Declared groups are reported separately; different same-class actions are not automatically duplicates.

The current 359-record benchmark overlaps **188/359 = 52.37%** original training IDs, versus 171 nonmembers. Its overlap by current queue split is `{dict(overlap_by_split)}`. V3/V4 DEV and original sealed TEST IDs have zero training-ID membership; that did not prevent the exact v4 text contamination. Historical ID hashing changes record IDs, so matching uses original_id. No absence-of-ID claim is treated as leakage-free.

## Balance, coverage and language evidence

{class_table}
TRAIN has only **22 declared scenario groups**: eight original 20-row synthetic scenario families plus fourteen four-language targeted scenarios. DEV has 17 declared groups. The 216 rows therefore substantially overstate independent semantic breadth; even independence between differently named groups is not guaranteed. Action classes close_app/open_folder/open_url have only four positive TRAIN examples each, derived from one scenario; no training positive for many real Registry actions (including weather/location, project inspection/testing, read_file, recent_files and search_files). Capability matcher has binary matched/unmatched classes rather than a capability enum, but absence of positive actions still constrains its learned compatibility.

All request mode/binary/missing/reference/capability distributions and proportions are in label_distribution.csv. Zero-count classes are explicitly added from nonsealed labels and the actual Registry. __none__ is an explicit head sentinel, distinct from source selected_capability null and empty lists. No fake labels were added to sources. Balanced class weighting is reasonable for skewed binary heads but cannot create missing classes or independent examples; path imbalance is moderate, while DEV is dominated by clarify and single_action. Sampling would need group/language-aware human-reviewed data, not duplication.

{lang_table}
English, Telugu, Hindi, Tamil, Latin transliteration and language mixing are present. Their training rows are authored scenario variants, not independent proof of language support. Native-script coverage is especially concentrated in a few paths; no language/script has adequate independently verified original training-label coverage. DEV native Telugu/Hindi/Tamil counts are small; language-specific accuracy cannot support a robust 90% claim. Spelling, grammar, meaning-preserving translations and transliteration variants require native-speaker adjudication. No source was translated or normalized in place; original utterances remain in the review CSV, with detected script stored separately.

## Integrity, evidence limits and 90% readiness

Every loaded head is LogisticRegression with finite coefficients; actual class arrays and feature dimensions are recorded in integrity_checks.json. Probability lookups use model.classes_ and correctly handle binary positive class indices. No direct target fields are included in encoder input. Constructed context/capability sets can nevertheless act as synthetic scenario cues, and real inference may supply different sets. No live adapter integration or production-context fidelity was inferred from offline examples.

Historical DEV confirmation reports 61/78 execution paths (78.2%), 12/36 applicable capabilities (33.3%), 49/78 intents (62.8%) and 65/78 exact mode sets (83.3%). These are pre-existing diagnostics against unverified labels, not new accuracy results. Sealed outcomes were not used. Accuracy must specify the field or full-decision exactness and be measured on independent reviewed groups; a class-imbalanced aggregate alone is not the requested generalization evidence.

Schema parsing passed for all analyzed nonsealed examples and nonsealed reviewed labels. A schema pass is not semantic correctness: the older parser accepts any nonempty path/intent string and does not fully enforce path/uncertainty/capability invariants. The original trainer lacks a human-review gate and parses sealed rows before filtering. Encoder/tokenizer revision, actual token-loss distribution and source-code version at fitting remain NOT_VERIFIED. Early source preview incidentally displayed one sealed record and historical prose containing test outcomes; neither was used for label analysis or tuning. Automated reports suppress sealed labels, including historical copies; this audit cannot claim zero incidental sealed visibility.

Input hashes were rechecked without changes. Existing working-tree changes were preserved. No pytest, dataset builders, training/evaluation main functions, model downloads or reference-project access were performed. Automated parsing and head inspection are analysis only. Semantic-review CSV rows are flags with original evidence, not corrected labels; multiple rows per record and multiple dataset versions must not be added as independent bad-record counts.

## Explicit answers and priorities

1. Exact fitted data: 216 train rows in hash-matched v3 dataset, listed verbatim by ID in source_provenance.json; 160 synthesized v2 train rows plus 56 targeted v3 rows.
2. Independently verified original training labels: **0 established**. Later ledger records 18 accept/correct decisions under another contract, 12 nonsealed eligible and 11 later training-ID reviews; independent full-contract/native correctness remains NOT_VERIFIED.
3. Synthetic/pending/unreviewed: **216/216 original training rows** synthetic pending review; **359/359 consolidated rows** historical_synthetic/pending metadata. Of queue review decisions, 18 accept/correct, 13 skip, 328 absent, zero recorded ambiguous/rejected; 75 import rejections. Semantic ambiguity and actual contradictory-gold counts remain NOT_VERIFIED. The visible differing "Continue that." labels have different context and are not established contradictions.
4. Training overlap: **216** source training IDs and **188/359** later queue original IDs; overlap by split above. Every available source record receives an overlap CSV row.
5. Leakage: one confirmed v4 TRAIN–TEST duplicate pair, propagated into queue TRAIN–development; seven/five uncertain near-text cross-split pairs respectively. No v3 exact TRAIN–DEV overlap. Translation/template leakage beyond these checks is NOT_VERIFIED.
6. Unreliable/sparse labels: missing prior_request/query and five reference kinds absent from fitted heads; four-example capabilities; consolidation missing clarification entirely; none capability-family mappings; context/reference-derived labels and unreviewed synthetic gold.
7. Adequate verified language/script coverage: **none established**. Presence and counts are documented; naturalness and independent intent/action preservation are unverified.
8. Bugs: structured missing-information conversion, omitted capability mappings, name-keyed descriptor overwrite, fixed reference/description reconstruction and missing class coverage; current Registry argument discrepancies and weak schema/review gating. No evidence of direct gold-field feature leakage or encoder fine-tuning.
9. Meaningful 90% target: **not yet supported** by trusted independent coverage/evaluation. BLOCKED_DATA_QUALITY, not BLOCKED_PROVENANCE, because exact source hash/IDs are established. This does not prove 90% is impossible after approved data/contract work.
10. Five highest-priority fixes, requiring separate approval:
   1. Commission blind full-contract human/native-speaker review of original train/dev labels; reconcile newer review decisions explicitly without rewriting history.
   2. Correct and audit the consolidation missing-information and capability-family mappings; recover quarantined candidates through review.
   3. Align training/inference capability descriptors, Registry argument schemas and reference/missing-field output semantics.
   4. Rebuild splits from audited paraphrase/translation/template families; remove confirmed cross-split contamination and establish a fresh locked reviewed evaluation set.
   5. Add independent reviewed scenarios for missing/rare classes and native/Latin/code-switched coverage; freeze encoder/source fingerprints and report per-head, per-class and language metrics with uncertainty.

No fixes were applied. Await explicit approval before changing datasets, labels, code, models or reviewer history.
'''
(OUT/'dataset_quality_audit.md').write_text(report,encoding='utf-8')
# Record nonsealed DEV diagnostic provenance without opening sealed prediction reports.
source('benchmarks/fast_classifier/PHASE_3C_EVALUATION.md'); source('benchmarks/fast_classifier/v4/DATASET_CARD.md'); source('benchmarks/fast_classifier/v4/ANNOTATION_GUIDE.md'); source('benchmarks/jarvis_fast_brain/EVALUATION_PLAN.md')
manifest['timestamp_completed']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat()
manifest['script_versions']={'audit.py':'1.0','finalize.py':'1.0'}
manifest['outputs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='audit_manifest.json'}
manifest['input_hashes_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==x['sha256'] for p,x in manifest['inputs'].items())
write('audit_manifest.json',manifest)
print(json.dumps({'descriptor_mismatch_records':len(mismatch_ids),'descriptor_mismatch_pairs':dict(cache_mismatch),'registry_mismatches':len(registry_mismatches),'human_reviews':review_summary,'queue_overlap_by_split':dict(overlap_by_split),'types':dict(types),'inputs_unchanged':manifest['input_hashes_unchanged'],'metadata_match':all(integrity['metadata_equals_joblib'].values())},indent=2))
