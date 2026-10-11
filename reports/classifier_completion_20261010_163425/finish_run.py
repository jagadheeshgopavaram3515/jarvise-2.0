"""Resume reporting/real encoder timing from saved fits; never refit/overwrite models."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
path=Path(__file__).resolve().parent/'run_completion.py'
source=path.read_text(encoding='utf-8')
scope={'__file__':str(path),'__name__':'completion_resume'}
exec(compile(source[:source.index('history=[];best=None;best_art=None;best_name=None')],str(path),'exec'),scope)
meta=scope['js'](scope['OUT']/'best_candidate_metadata.json')
scope.update(best=scope['js'](scope['OUT']/'best_metrics.json'),best_art=scope['joblib'].load(scope['OUT']/'best_candidate/compact_heads.joblib'),best_name=meta['selected_experiment'],history=list(scope['csv'].DictReader((scope['OUT']/'experiment_history.csv').open(encoding='utf-8-sig'))))
final_history=[];best=None;best_art=None;best_name=None
for row in scope['history']:
 artifact=scope['joblib'].load(scope['OUT']/row['experiment']/'compact_heads.joblib')
 metrics=scope['evaluate'](artifact,True);scope['dump'](row['experiment']+'/final_policy_metrics.json',metrics)
 final_history.append({**row,'strict_exact':metrics['exact_agreement'],'path':metrics['fields']['execution_path'],'capability':metrics['fields']['selected_capability'],'reference':metrics['fields']['context_references'],'missing':metrics['fields']['missing_information'],'abstentions':metrics['abstentions']})
 if best is None or (metrics['exact_agreement'],metrics['fields']['execution_path'],metrics['fields']['selected_capability'])>(best['exact_agreement'],best['fields']['execution_path'],best['fields']['selected_capability']):best=metrics;best_art=artifact;best_name=row['experiment']
scope.update(best=best,best_art=best_art,best_name=best_name,history=final_history)
scope['tab']('final_experiment_history.csv',final_history);scope['dump']('best_metrics.json',best)
meta.update(selected_experiment=best_name,strict_exact=best['exact_agreement'],final_policy='explicit unresolved-span rejection and capability confidence support',configuration=best_art['experiment_configuration'])
# Existing best copy is preserved; choose another immutable model path if tie-break changes.
meta['model_path']=str(scope['OUT']/best_name/'compact_heads.joblib');meta['model_sha256']=scope['sha'](meta['model_path']);scope['dump']('best_candidate_metadata.json',meta)
abstention=[]
for threshold in (0,.25,.5,.65,.8,.9):
 adapter=scope['make_adapter'](best_art,True);accepted=correct=full=0
 for record in scope['val']:
  prediction=adapter.predict_with_abstention(scope['request'](record),threshold);accept=not prediction.get('abstained',False);accepted+=accept;correct+=accept and scope['exact'](record,prediction);full+=scope['exact'](record,prediction)
 abstention.append({'threshold':threshold,'accepted':accepted,'denominator':40,'coverage':accepted/40,'accepted_exact':correct,'accepted_correctness':correct/accepted if accepted else None,'full_set_exact':full,'full_set_denominator':40})
scope['dump']('abstention_metrics.json',abstention)
exec(compile(source[source.index("cases=[];ids={r['id']:r for r in val}"):],str(path),'exec'),scope)
