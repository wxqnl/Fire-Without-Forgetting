"""Run protocol baselines alongside known project jobs, within GPU memory limits."""
import datetime,fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/data/Minko/fire-without-forgetting')
os.chdir('/data/Minko')
lock=(ROOT/'runs/queue_dreamerv3_v1.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
active={}
def atomic(path,obj):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(path)
def read_result(j):
 p=ROOT/'runs/v2'/j['name']/'result.json'
 return json.loads(p.read_text()) if p.exists() else None
while True:
 cfg=json.loads((ROOT/'configs/experiments_dreamerv3_v1.json').read_text())
 for name,w in list(active.items()):
  code=w['process'].poll()
  if code is None:
   if time.time()-w['start']>w['timeout']*3600:
    if not w.get('terminated'):w['process'].terminate();w['terminated']=time.time()
    elif time.time()-w['terminated']>30:w['process'].kill()
   continue
  atomic(w['run']/'result.json',dict(returncode=124 if w.get('terminated') else code,gpu=w['gpu'],elapsed_seconds=time.time()-w['start'],command=w['command'],ended_at=datetime.datetime.now().astimezone().isoformat()))
  w['log'].close();del active[name]
 outcomes={j['name']:read_result(j) for j in cfg['jobs']}
 completed=[n for n,r in outcomes.items() if r and r['returncode']==0]
 failed=[n for n,r in outcomes.items() if r and r['returncode']!=0]
 pending=[j for j in cfg['jobs'] if outcomes[j['name']] is None and j['name'] not in active]
 blocked={}
 for job in pending:
  if not job.get('enabled',False):continue
  gpu=job['gpu'];name=job['name']
  assert gpu in [0,1,2,3] and job['seed']==0
  if any(w['gpu']==gpu for w in active.values()):continue
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
  usage={int((v:=l.split(','))[0]):dict(uuid=v[1].strip(),used=int(v[2]),total=int(v[3])) for l in rows}
  row=usage[gpu]
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
  pids={int(v[1]) for l in apps if (v:=l.split(','))[0].strip()==row['uuid']}
  known=[]
  for status in ['queue_v2_status.json','queue_r2i_v4_status.json']:
   p=ROOT/'runs'/status
   if p.exists():known+=json.loads(p.read_text())['active']
  allowed={w['pid'] for w in known if w['gpu']==gpu}
  if pids-allowed:
   blocked[name]='Unknown GPU processes; no colocation permitted';continue
  if row['total']-row['used']<job['device_memory_budget_mib']+job['device_headroom_mib']:
   blocked[name]='Insufficient measured free VRAM plus headroom';continue
  run=ROOT/'runs/v2'/name
  run.mkdir(parents=True,exist_ok=True)
  if (run/'command.json').exists():
   blocked[name]='Prior attempt exists; inspect and assign a new run name after a concrete fix';continue
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONWARNINGS='ignore',WANDB_MODE='disabled',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0')
  env.update(job['env'])
  atomic(run/'command.json',job['command']);atomic(run/'job.json',job)
  log=(run/'train.log').open('w');start=time.time()
  p=subprocess.Popen(job['command'],cwd='/data/Minko',env=env,stdout=log,stderr=subprocess.STDOUT)
  active[name]=dict(name=name,gpu=gpu,process=p,pid=p.pid,run=run,command=job['command'],log=log,start=start,timeout=job['timeout_hours'])
  print('Started',name,'GPU',gpu,'PID',p.pid,flush=True)
 report=dict(state='running' if active else ('waiting' if pending else 'completed'),queue_pid=os.getpid(),
  active=[dict(job=w['name'],gpu=w['gpu'],pid=w['pid'],started_at=datetime.datetime.fromtimestamp(w['start']).astimezone().isoformat(),log=str(w['run']/'train.log')) for w in active.values()],
  completed=completed,failed=failed,pending=[j['name'] for j in pending if j['name'] not in active],
  blocked=blocked,seed_policy='seed 0 only',updated_at=datetime.datetime.now().astimezone().isoformat())
 atomic(ROOT/'runs/queue_dreamerv3_v1_status.json',report)
 if not active and not pending:break
 time.sleep(10)
