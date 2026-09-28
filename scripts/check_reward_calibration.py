"""One full real-replay update for the bounded reward-scale calibration."""
import os
os.environ["JAX_PLATFORMS"]="cpu";os.environ["CUDA_VISIBLE_DEVICES"]=""
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-8:])
import sys,json,pickle
from pathlib import Path
import numpy as np
from ruamel.yaml import YAML
root=Path("/data/Minko/fire-without-forgetting")
sys.path[:0]=[str(root/"code/r2i-v3"),str(root/"code/r2i-v3/recall2imagine")]
import embodied,jax,jax.numpy as jnp
from recall2imagine import agent,train
run=root/"runs/v2/r2i_repeat_easy_s0_cuda"
raw=YAML(typ="safe").load((run/"config.yaml").read_text())
raw["training_reward_scale"]=48.0
raw["jax"]["platform"]="cpu"
raw["jax"]["metrics_every"]=1
cfg=embodied.Config(raw)
env=train.make_env(cfg,seed=1000000)
step=embodied.Counter()
model=agent.Agent(env.obs_space,env.act_space,step,cfg)
saved=pickle.loads((run/"checkpoint.ckpt").read_bytes())
model.load(saved["agent"])
files=sorted((run/"replay").glob("*.npz"))[-16:]
episodes=[]
for f in files:
    with np.load(f) as ep:episodes.append({k:ep[k][:128].copy() for k in ep.files if k!="id"})
data={k:np.stack([ep[k] for ep in episodes]) for k in episodes[0]}
assert data["reward"].shape==(16,128)
scaled_rewards=data["reward"]*cfg.training_reward_scale
batch=model._convert_inps(data,model.train_devices)
outs,state,metrics=model.train(batch)
core={k:float(v) for k,v in metrics.items() if any(t in k for t in ["opt_loss","loss_mean","grad_norm","reward_max_data"])}
if not core or not all(np.isfinite(x) for x in core.values()):raise RuntimeError(core)
folder=root/"runs/integration_r2i_reward_scale";folder.mkdir(exist_ok=True)
cp=embodied.Checkpoint(str(folder/"checkpoint.ckpt"),parallel=False)
cp.step=step;cp.agent=model
step.increment()
cp.save()
written=pickle.loads((folder/"checkpoint.ckpt").read_bytes())
if int(written["step"])!=1:raise RuntimeError("Synchronous checkpoint did not persist.")
obs=env.step({"action":np.zeros(4,np.float32),"reset":True})
max_raw_reward=0.
for _ in range(8):
    obs=env.step({"action":np.eye(4,dtype=np.float32)[0],"reset":False})
    max_raw_reward=max(max_raw_reward,abs(float(obs["reward"])))
report=dict(status="passed",source="actual saved official POPGym Easy replay; one full world-model/actor/critic update on CPU",
    source_checkpoint_step=int(saved["step"]),batch_size=16,batch_length=128,training_reward_scale=48.0,
    raw_reward_range=[float(data["reward"].min()),float(data["reward"].max())],
    scaled_target_range=[float(scaled_rewards.min()),float(scaled_rewards.max())],
    environment_reward_max_abs=max_raw_reward,core_metrics=core,synchronous_checkpoint_saved_step=int(written["step"]),
    interpretation="integration only; not an additional seed, learning result, or calibration outcome")
(root/"docs/reward_calibration_integration.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
env.close()

