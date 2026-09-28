"""Measure shared GPU resources; never attribute co-located power to one model."""
import datetime,fcntl,json,subprocess,time
from pathlib import Path
r=Path('/data/Minko/fire-without-forgetting')
lock=(r/'runs/resources.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
 active=[];pending=[]
 for name in ['queue_v2_status.json','queue_r2i_v4_status.json','queue_dreamerv3_v1_status.json']:
  p=r/'runs'/name
  if p.exists():
   q=json.loads(p.read_text());active+=q['active'];pending+=q['pending']
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True).splitlines()
 memrows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines()
 process_mem={int(v[0]):int(v[1]) for l in memrows if (v:=l.split(','))[1].strip().isdigit()}
 samples=[]
 for line in rows:
  v=[x.strip() for x in line.split(',')];gpu=int(v[0])
  jobs=[dict(job=w['job'],pid=w['pid'],process_memory_mib=process_mem.get(w['pid'])) for w in active if w['gpu']==gpu]
  if jobs:samples.append(dict(gpu=gpu,jobs=jobs,shared_device_measurement=len(jobs)>1,memory_mib=int(v[1]),utilization_percent=int(v[2]),power_watts=float(v[3])))
 if samples:
  with (r/'runs/resource_samples.jsonl').open('a') as f:f.write(json.dumps(dict(timestamp=datetime.datetime.now().astimezone().isoformat(),schema=2,samples=samples))+'\n')
 if not active and not pending:break
 time.sleep(30)
