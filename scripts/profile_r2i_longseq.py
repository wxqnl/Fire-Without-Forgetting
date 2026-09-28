import os,sys,json,time,pickle,subprocess
from pathlib import Path
root=Path('/data/Minko/fire-without-forgetting')
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[64:80])
sys.path[:0]=[str(root/'code/r2i-resetfix-v4'),str(root/'code/r2i-resetfix-v4/recall2imagine')]
import numpy as np
import embodied,jax
from recall2imagine import agent,train
cfg=embodied.Config(agent.Agent.configs['defaults']).update(agent.Agent.configs['popgym'])
cfg=cfg.update({'jax.prealloc':False,'jax.metrics_every':1,'jax.precision':'float16','envs.parallel':'none','envs.restart':False})
print('PROFILE CONFIG',cfg.batch_size,cfg.batch_length,cfg.jax.precision,flush=True)
env=train.make_env(cfg,seed=1000000)
model=agent.Agent(env.obs_space,env.act_space,embodied.Counter(),cfg)
run=root/'runs/v2/r2i_repeat_easy_s0_cuda'
# Fresh FP16 optimizer state includes the required dynamic loss scaler.
files=sorted((run/'replay').glob('*.npz'))[-4:]
episodes=[]
for f in files:
 with np.load(f) as z:episodes.append({k:z[k].copy() for k in z.files if k!='id'})
data={k:np.stack([e[k] for e in episodes]) for k in episodes[0]}
data['is_first'][:,0]=True
assert data['reward'].shape==(4,1024)
batch=model._convert_inps(data,model.train_devices)
times=[];metrics_all=[];state=None
for i in range(12):
 t=time.monotonic()
 _,state,mets=model.train(batch,state)
 jax.block_until_ready(model.varibs)
 dt=time.monotonic()-t
 core={k:float(v) for k,v in mets.items() if any(x in k for x in ['opt_loss','grad_norm','loss_mean'])}
 if not all(np.isfinite(v) for v in core.values()):raise RuntimeError(core)
 metrics_all.append(core)
 if i>=2:times.append(dt)
 print(json.dumps({'update':i,'seconds':dt,'core':core}),flush=True)
mem=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
own=[line for line in mem.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
report=dict(status='passed',code='r2i-resetfix-v4',batch=[4,1024],precision='float16',seed=0,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],source='real saved replay and fresh seed-0 official FP16 initialization; 12 complete world-model/actor/critic updates; first 2 excluded from timing',warm_update_seconds=times,median_update_seconds=float(np.median(times)),replay_positions_per_second=4096/float(np.median(times)),process_gpu_memory=own,core_metrics=metrics_all[-1],co_located_with='existing project SSM Hard run',scope='hardware/integration measurement; not a learning run or isolated efficiency result')
(root/'docs/r2i_longseq_gpu_profile_fresh.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
env.close()
