import sys,json,copy
from pathlib import Path
import numpy as np,pytest
ROOT=Path('D:/JARVIS');OUT=Path(__file__).resolve().parent;PRIOR=ROOT/'reports/autonomous_bert_training_20261010_161410';BUILD=ROOT/'reports/multilingual_dataset_build_20261010_155129'
sys.path.insert(0,str(OUT));sys.path.insert(0,str(BUILD))
from experimental_adapter import typed_resolve,project_reference,pair_features,safe_fallback,ExperimentalCompactAdapter
from assistant.orchestration.decision import ClassifierDecision
from dataset_tools import extract_missing,capability_family,registry_specs
VAL=[json.loads(l) for l in (PRIOR/'provisional_validation.jsonl').read_text(encoding='utf-8').splitlines()]
LIMIT=json.loads((PRIOR/'deterministic_output_limits.json').read_text(encoding='utf-8'))
CASES=[r for r in VAL if r['id'] in {x['record_id'] for x in LIMIT['records']}]

@pytest.mark.parametrize('record',CASES,ids=[r['original_id'] for r in CASES])
def test_each_eleven_typed_context_resolution_without_targets(record):
 for ref in record['expected']['context_references']:
  actual=typed_resolve(record['context'],ref['kind'],record['utterance'])
  assert actual==ref['resolved_value']

def test_resolver_never_first_arbitrary_scalar():
 assert typed_resolve({'unrelated':'wrong','recent_topics':[{'topic':'something','decision':'correct'}]},'prior_conversation','remember something')=='correct'
def test_multiple_unmatched_decisions_unresolved():
 assert typed_resolve({'recent_topics':[{'topic':'a','decision':'x'},{'topic':'b','decision':'y'}]},'prior_conversation','remember previous choice') is None
def test_matching_named_decision_selects_correct_entity():
 assert typed_resolve({'projects':[{'name':'Alpha','decision':'one'},{'name':'Beta','decision':'two'}]},'prior_conversation','remind me Beta')=='two'
def test_unseen_project_alias_and_span():
 context={'projects':[{'name':'New initiative','aliases':['space elevator']}]}
 assert project_reference(context,'Remember our space elevator?')=='our space elevator'
 assert typed_resolve(context,'project','Remember our space elevator?')=='New initiative'
def test_conflicting_project_aliases_not_resolved():
 context={'projects':[{'name':'A','aliases':['shared']},{'name':'B','aliases':['shared']}]}
 assert typed_resolve(context,'project','shared') is None
def test_empty_context_never_fabricated():assert typed_resolve({},'item','open that') is None
def test_nonstring_context_decision_not_coerced():assert typed_resolve({'recent_topics':[{'decision':False}]},'prior_conversation','remember') is None
def test_same_capability_descriptor_feature_structure():
 a=np.array([1.,2.]);b=np.array([3.,4.]);assert pair_features(a,b,'full').shape==(8,);assert pair_features(a,b,'interaction').shape==(4,)
def test_unknown_feature_style_explicit_error():
 with pytest.raises(ValueError):pair_features(np.ones(2),np.ones(2),'unknown')
def test_missing_objects_preserve_names_descriptions():assert extract_missing([{'name':'prior_request','description':'original'}])==[{'name':'prior_request','description':'original'}]
@pytest.mark.parametrize('name,family',[('open_folder','file'),('recent_files','file'),('grep_code','project'),('run_project_tests','project')])
def test_all_omissions_stay_fixed(name,family):assert capability_family(name)==family
def test_safe_fallback_contract():
 d=safe_fallback('low confidence');ClassifierDecision.from_mapping(d);assert d['abstained'];assert d['selected_capability'] is None;assert not d['missing_information'];assert not d['context_references']
def test_model_has_no_unseen_validation_span_vocabulary():
 import joblib
 model=joblib.load(OUT/'best_candidate/compact_heads.joblib');train=[json.loads(l) for l in (PRIOR/'provisional_train.jsonl').read_text(encoding='utf-8').splitlines()]
 allowed={'__none__','__utterance__'}|{x['reference'] for r in train for x in r['expected']['context_references']}
 assert set(model['reference_span'].classes_)<=allowed
def test_original_artifacts_hashes():
 import hashlib
 m=json.loads((OUT/'run_manifest.json').read_text());
 for key in ('baseline_model_sha256','previous_candidate_sha256'):
  path=ROOT/'benchmarks/fast_classifier/models/v3/compact_heads.joblib' if key.startswith('baseline') else PRIOR/'best_candidate/compact_heads.joblib'
  assert hashlib.sha256(path.read_bytes()).hexdigest()==m[key]
def test_protected_records_excluded():
 ids=set(json.loads((BUILD/'protected_test_manifest.json').read_text())['original_protected_record_ids'])
 assert not ({r['original_id'] for r in VAL}&ids)
def test_baselines_reproduced():
 for r in json.loads((OUT/'baseline_reproduction_checks.json').read_text()).values():assert r['exact_agreement']==8 and r['exact_prediction_reproduction']
def test_actual_encoder_fixed_denominator():
 r=json.loads((OUT/'actual_encoder_evaluation.json').read_text(encoding='utf-8'));assert r['denominator']==40;assert r['encoder_frozen']
def test_every_best_output_valid_and_multilingual_preserved():
 r=json.loads((OUT/'best_metrics.json').read_text(encoding='utf-8'))
 assert len(r['languages'])==10
 for p in r['predictions']:ClassifierDecision.from_mapping(p['prediction'])
def test_abstention_does_not_redefine_main_denominator():
 rows=json.loads((OUT/'abstention_metrics.json').read_text())
 assert all(r['full_set_denominator']==40 and r['denominator']==40 for r in rows)
 assert all(a['accepted']>=b['accepted'] for a,b in zip(rows,rows[1:]))
def test_missing_positive_capability_class_explicit():
 class Stub:
  classes_=np.array([0])
 a=ExperimentalCompactAdapter.__new__(ExperimentalCompactAdapter);a.artifact={'capability_matcher':Stub()};a.last_diagnostics={}
 from assistant.orchestration.decision import DecisionRequest,Capability
 with pytest.raises(ValueError,match='positive class'):a._select_capability(DecisionRequest('x',{},(Capability('test','descriptor'),)),np.ones(2),np.ones((1,2)))
