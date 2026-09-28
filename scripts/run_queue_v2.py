"""Run the approved single-seed experiment manifest on idle node-42 GPUs."""
import datetime, fcntl, json, os, signal, subprocess, time
from pathlib import Path
ROOT=Path('/data/Minko/fire-without-forgetting')
os.chdir('/data/Minko')
lock=(ROOT/'runs/queue_v2.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
active={}
def atomic(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def result(job):
 p=ROOT/'runs/v2'/job['name']/'result.json'
 return json.loads(p.read_text()) if p.exists() else None
while True:
 cfg=json.loads((ROOT/'configs/experiments_v2.json').read_text())
 for gpu,w in list(active.items()):
  code=w['process'].poll()
  if code is None:
   if time.time()-w['start']>w['timeout']*3600:
    if not w.get('terminated'):
     w['process'].terminate();w['terminated']=time.time()
    elif time.time()-w['terminated']>30:w['process'].kill()
   continue
  atomic(w['run']/'result.json',dict(returncode=124 if w.get('terminated') else code,gpu=gpu,elapsed_seconds=time.time()-w['start'],command=w['command'],ended_at=datetime.datetime.now().astimezone().isoformat()))
  w['log'].close();del active[gpu]
 outcomes={j['name']:result(j) for j in cfg['jobs']}
 completed=[n for n,r in outcomes.items() if r and r['returncode']==0]
 failed=[n for n,r in outcomes.items() if r and r['returncode']!=0]
 running={w['name'] for w in active.values()}
 pending=[j for j in cfg['jobs'] if outcomes[j['name']] is None and j['name'] not in running]
 ready=[j for j in pending if j.get('enabled',True) and all(n in completed for n in j.get('requires',[]))]
 if ready:
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
  usage={int(v[0]):(int(v[1]),int(v[2])) for l in rows if (v:=l.split(','))}
  for gpu in cfg['gpus']:
   if gpu in active or usage[gpu][0]>=128 or usage[gpu][1]!=0:continue
   eligible=[j for j in ready if gpu in j.get('allowed_gpus',cfg['gpus'])]
   if not eligible:continue
   job=eligible[0];ready.remove(job);name=job['name'];run=ROOT/'runs/v2'/name
   run.mkdir(parents=True,exist_ok=True)
   if (run/'command.json').exists():raise RuntimeError(f'Interrupted run {name}; inspect and use a new run name or explicit full-state resume.')
   command=job['command'];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONWARNINGS='ignore',WANDB_MODE='disabled',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',XLA_PYTHON_CLIENT_PREALLOCATE='false')
   env.update(job.get('env',{}))
   atomic(run/'command.json',command);atomic(run/'job.json',job)
   log=(run/'train.log').open('w');start=time.time()
   process=subprocess.Popen(command,cwd='/data/Minko',env=env,stdout=log,stderr=subprocess.STDOUT)
   active[gpu]=dict(name=name,pid=process.pid,process=process,run=run,start=start,log=log,command=command,timeout=job.get('timeout_hours',48))
 report=dict(state='running' if active else ('waiting' if pending else 'completed'),queue_pid=os.getpid(),active=[dict(job=w['name'],gpu=g,pid=w['pid'],started_at=datetime.datetime.fromtimestamp(w['start']).astimezone().isoformat(),log=str(w['run']/'train.log')) for g,w in sorted(active.items())],completed=completed,failed=failed,pending=[j['name'] for j in pending if j['name'] not in {w['name'] for w in active.values()}],seed_policy='seed 0 only',updated_at=datetime.datetime.now().astimezone().isoformat())
 atomic(ROOT/'runs/queue_v2_status.json',report)
 if not active and not pending:break
 time.sleep(5)
