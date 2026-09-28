"""Compare the trained official R2I parallel core with its recurrent core on real replay."""
import os
os.environ["JAX_PLATFORMS"]="cpu";os.environ["CUDA_VISIBLE_DEVICES"]=""
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-8:])
import sys,json,pickle,argparse
from pathlib import Path
import numpy as np
from ruamel.yaml import YAML
p=argparse.ArgumentParser();p.add_argument("--code",required=True);p.add_argument("--output",required=True)
a=p.parse_args();root=Path("/data/Minko/fire-without-forgetting")
sys.path[:0]=[a.code,str(Path(a.code)/"recall2imagine")]
import embodied,jax,jax.numpy as jnp
from recall2imagine import agent,ninjax as nj,jaxutils,nets
run=root/"runs/v2/r2i_repeat_easy_s0_cuda"
cfg=embodied.Config(YAML(typ="safe").load((run/"config.yaml").read_text()))
obs_space={"observation":embodied.Space(np.float32,(4,),0,1),"reward":embodied.Space(np.float32),
"is_first":embodied.Space(bool),"is_last":embodied.Space(bool),"is_terminal":embodied.Space(bool),"step_no":embodied.Space(np.int32)}
act_space={"action":embodied.Space(np.float32,(4,),0,1),"reset":embodied.Space(bool)}
inner=agent.Agent.inner(obs_space,act_space,embodied.Counter(),cfg,name="agent")
jaxutils.COMPUTE_DTYPE=jnp.float32
saved=pickle.loads((run/"checkpoint.ckpt").read_bytes());vs=jax.device_put(saved["agent"])
files=sorted((run/"replay").glob("*.npz"))[-4:]
episodes=[]
for f in files:
 with np.load(f) as ep:episodes.append({k:ep[k][:128].copy() for k in ["observation","action","reward","is_first","is_terminal"]})
data={k:np.stack([ep[k] for ep in episodes]) for k in episodes[0]}
data["is_first"][:,0]=True
def prepare(d):
 d={k:v.astype(jnp.float32) for k,v in d.items()}
 rssm=inner.wm.rssm
 initial=rssm.initial(len(d["reward"]))
 embed=inner.wm.encoder(d)
 actions=jnp.concatenate([jnp.zeros_like(d["action"][:,:1]),d["action"][:,:-1]],1)
 post,_=rssm.observe(embed,actions,d["is_first"])
 prev=jnp.concatenate([initial["stoch"][:,None],post["stoch"][:,:-1]],1).reshape(*actions.shape[:2],-1)
 actions=rssm._mask_steps(actions,1-d["is_first"])
 x=rssm.get("img_in",nets.Linear,**rssm._kw)(jnp.concatenate([prev,actions],-1))
 return x,initial["hidden"]
prep=nj.pure(prepare)
x,initial=jax.jit(lambda v,d:prep(v,jax.random.PRNGKey(0),d,create=False,modify=False)[0])(vs,jax.device_put(data))
par=nj.pure(lambda x,init,first:inner.wm.rssm.core((x,first),init,init))
step=nj.pure(lambda x,state:inner.wm.rssm.core(x,state))
par=jax.jit(lambda v,x,init,first,fn=par:fn(v,jax.random.PRNGKey(0),x,init,first,create=False,modify=False)[0])
step=jax.jit(lambda v,x,state,fn=step:fn(v,jax.random.PRNGKey(0),x,state,create=False,modify=False)[0])
results=[]
for mode in ["no_interior_reset","actual_episode_resets"]:
 first=np.zeros_like(data["is_first"],dtype=np.float32)
 if mode=="actual_episode_resets":first=data["is_first"].astype(np.float32)
 yp,xp=par(vs,x,initial,jax.device_put(first));yp=yp[0]
 state=initial;ys=[];xs=[]
 for t in range(x.shape[1]):
  mask=jax.device_put(first[:,t,None,None])
  state=state*(1-mask)+initial*mask
  y,state=step(vs,x[:,t:t+1],state)
  ys.append(y[0][:,0]);xs.append(state)
 yr=jnp.stack(ys,1);xr=jnp.stack(xs,1)
 state_error=np.asarray(jax.device_get(jnp.abs(xp-xr)))
 output_error=np.asarray(jax.device_get(jnp.abs(yp-yr)))
 row=dict(mode=mode,hidden_max_error=float(state_error.max()),hidden_mean_error=float(state_error.mean()),
 output_max_error=float(output_error.max()),output_mean_error=float(output_error.mean()),
 hidden_rms=float(jnp.mean(jnp.abs(xr)**2)**0.5),
 reset_positions=[np.where(z)[0].tolist() for z in first])
 results.append(row);print(json.dumps(row),flush=True)
report=dict(code=a.code,source="actual saved R2I Easy replay, fixed trained parameters and identical projected inputs",
 checkpoint_step=int(saved["step"]),replay_files=[f.name for f in files],batch_shape=list(x.shape),
 results=results,scope="isolates SSM core numerical consistency and episode resets; does not establish the sole cause of a learning failure")
Path(a.output).write_text(json.dumps(report,indent=2))

