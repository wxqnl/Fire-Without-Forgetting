import datetime,json,math,os,subprocess,time
from pathlib import Path
r=Path('/data/Minko/fire-without-forgetting');q=json.loads((r/'runs/queue_v2_status.json').read_text());runs=[]
for name in ['queue_r2i_v4_status.json','queue_dreamerv3_v1_status.json']:
 extra=r/'runs'/name
 if extra.exists():
  more=json.loads(extra.read_text())
  for k in ['active','completed','failed','pending']:q[k]+=more[k]
for w in q['active']:
 d=r/'runs/v2'/w['job'];p=d/'metrics.jsonl';records=[]
 if p.exists():
  for line in p.read_text().splitlines():
   try:records.append(json.loads(line))
   except json.JSONDecodeError:pass
 evals=[{k:x[k] for k in ['step','train_action_steps','eval_return','eval_return_std','eval_episodes'] if k in x} for x in records if 'eval_return' in x]
 bad=[];auxiliary=[]
 for x in records:
  for k,v in x.items():
   if isinstance(v,(int,float)) and not math.isfinite(v):
    # R2I balance_stats explicitly emits NaN for empty positive/negative subsets.
    if any(s in k for s in ['pos_loss','neg_loss','pos_acc','neg_acc']) or k.startswith(('timer/','replay/','eval_replay/')):
     auxiliary.append(k);continue
    bad.append({'step':x.get('step'),'metric':k})
 runs.append(dict(**w,process_alive=Path('/proc/'+str(w['pid'])).exists(),latest_step=max([x.get('step',0) for x in records],default=0),latest_action_steps=max([x.get('train_action_steps',0) for x in records],default=0) if any('train_action_steps' in x for x in records) else (max([x.get('step',0) for x in records],default=0) if w['job'].startswith(('persistent_ssm','mcn_')) else None),metrics_age_seconds=round(time.time()-p.stat().st_mtime,1) if p.exists() else None,latest_evaluations=evals[-3:],nonfinite_core_metrics=bad[-10:],nonfinite_auxiliary_metrics=sorted(set(auxiliary)),checkpoint_metadata=json.loads((d/'final_checkpoint.json').read_text()) if (d/'final_checkpoint.json').exists() else None,checkpoints=[dict(name=f.name,bytes=f.stat().st_size) for f in d.iterdir() if f.is_file() and f.suffix in ['.pt','.ckpt']]))
print(json.dumps(dict(checked_at=datetime.datetime.now().astimezone().isoformat(),queue_state=q['state'],runs=runs,completed=q['completed'],failed=q['failed'],pending=q['pending'],gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)),indent=2))
