"""Preserve stable checkpoints for reproducible probes and numerical audits."""
import datetime,fcntl,json,shutil,time
from pathlib import Path
r=Path('/data/Minko/fire-without-forgetting');lock=(r/'runs/checkpoint_archive.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
 q=json.loads((r/'runs/queue_v2_status.json').read_text())
 for name in ['queue_r2i_v4_status.json','queue_dreamerv3_v1_status.json']:
  extra=r/'runs'/name
  if extra.exists():
   more=json.loads(extra.read_text());q['active']+=more['active'];q['pending']+=more['pending']
 for w in q['active']:
  run=r/'runs/v2'/w['job']
  sources=[run/'training_state.pt'] if (run/'training_state.pt').exists() else [run/'latest_model.pt',run/'checkpoint.ckpt']
  for src in sources:
   if not src.exists():continue
   before=src.stat()
   if time.time()-before.st_mtime<20:continue
   folder=run/'checkpoints';folder.mkdir(exist_ok=True)
   stamp=datetime.datetime.fromtimestamp(before.st_mtime).strftime('%Y%m%dT%H%M%S')
   target=folder/(stamp+'_'+src.name)
   if target.exists():continue
   tmp=target.with_suffix(target.suffix+'.copying');shutil.copy2(src,tmp);after=src.stat()
   if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:
    tmp.unlink();continue
   tmp.replace(target)
   target.with_suffix(target.suffix+'.json').write_text(json.dumps(dict(run=w['job'],source=src.name,source_mtime=datetime.datetime.fromtimestamp(before.st_mtime).astimezone().isoformat(),bytes=before.st_size,archived_at=datetime.datetime.now().astimezone().isoformat()),indent=2))
   print('Archived',str(target),flush=True)
 if not q['active'] and not q['pending']:break
 time.sleep(30)
