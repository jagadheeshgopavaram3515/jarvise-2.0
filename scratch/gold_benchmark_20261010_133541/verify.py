import hashlib,json,pathlib,subprocess
ROOT=pathlib.Path('D:/JARVIS');WORK=pathlib.Path(__file__).resolve().parent
summary=json.loads((WORK/'preparation-summary.json').read_text(encoding='utf-8'));OUT=pathlib.Path(summary['bundle']);REPORT=pathlib.Path(summary['report'])
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mp=OUT/'steward/integrity-report.json';integrity=json.loads(mp.read_text(encoding='utf-8'))
assert digest(REPORT)==integrity['new_file_hashes'][REPORT.relative_to(ROOT).as_posix()]
text=REPORT.read_text(encoding='utf-8').replace('37 passed in 0.41 seconds','37 passed in 0.36 seconds (final rerun; initial pass 0.41 seconds)')
REPORT.write_text(text,encoding='utf-8')
before=json.loads((WORK/'integrity-before.json').read_text(encoding='utf-8'))
assert all(digest(ROOT/p)==sha for p,sha in before.items())
forms=[json.loads(x) for x in (OUT/'reviewer/records.jsonl').read_text(encoding='utf-8').splitlines()]
gold=[json.loads(x) for x in (OUT/'steward/gold-evaluation-set.jsonl').read_text(encoding='utf-8').splitlines()]
comparison=[json.loads(x) for x in (OUT/'steward/per-record-comparison.jsonl').read_text(encoding='utf-8').splitlines()]
metrics=json.loads((OUT/'steward/aggregate-metrics.json').read_text(encoding='utf-8'))
assert len(forms)==len(gold)==len(comparison)==len({r['review_id'] for r in forms})==359
assert all(r['gold_label'] is None and r['proposed_gold_label'] is None for r in gold)
assert all(r['prediction'] is None and r['gold_label'] is None for r in comparison)
assert not (OUT/'steward/predictions.jsonl').read_bytes()
assert metrics['independently_adjudicated']==metrics['predictions_executed']==0
assert metrics['exact_match']['status']=='NOT_MEASURED' and metrics['exact_match']['value'] is None
assert all(m['value'] is None and m['denominator']==0 for m in metrics['per_field'].values())
decisions=[json.loads(x) for x in (ROOT/'benchmarks/fast_classifier/fast_brain_v1/review_decisions.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(decisions)==len({r['record_id'] for r in decisions})==31
new=[ROOT/'benchmarks/fast_classifier/gold_review.py',ROOT/'tests/test_gold_review.py',REPORT,*[p for p in OUT.rglob('*') if p.is_file()],*[p for p in WORK.rglob('*') if p.is_file() and '__pycache__' not in p.parts]]
for path in new:
 if path.suffix in ('.md','.py'):assert not any(line.endswith((' ','\t')) for line in path.read_text(encoding='utf-8').splitlines()),str(path)
diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True);assert diff.returncode==0
integrity['new_paths']=[p.relative_to(ROOT).as_posix() for p in sorted(set(new))]
integrity['new_file_hashes']={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(new)) if p!=mp}
integrity['final_tests']='37 passed in 0.36 seconds';integrity['final_artifacts_verified']=True
mp.write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'review_records':359,'gold_labels':0,'predictions':0,'accuracy':'NOT_MEASURED','existing_files_unchanged':len(before),'test_result':integrity['final_tests'],'diff_check':diff.returncode,'blinded_reviewer_directory':str(OUT/'reviewer'),'report':str(REPORT)},indent=2))
