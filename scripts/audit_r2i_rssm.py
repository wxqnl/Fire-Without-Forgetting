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

def parallel(d):
 d={k:v.astype(jnp.float32) for k,v in d.items()}
 initial=inner.wm.rssm.initial(len(d['reward']))
 actions=jnp.concatenate([jnp.zeros_like(d['action'][:,:1]),d['action'][:,:-1]],1)
 post,prior=inner.wm.rssm.observe(inner.wm.encoder(d),actions,d['is_first'])
 return post,prior,initial,actions
fn=nj.pure(parallel)
post,prior,initial,actions=jax.jit(lambda v,d:fn(v,jax.random.PRNGKey(0),d,create=False,modify=False)[0])(vs,jax.device_put(data))
def step_fn(state,action,first):
 rssm=inner.wm.rssm
 state=jax.tree_util.tree_map(lambda x,y:rssm._mask(x,1-first)+rssm._mask(y,first),state,initial)
 action=rssm._mask(action,1-first)
 return rssm.img_step(state,action)
fn_step=nj.pure(step_fn)
fn_step_pure=fn_step
step=jax.jit(lambda v,state,action,first:fn_step_pure(v,jax.random.PRNGKey(0),state,action,first,create=False,modify=False)[0])
state=initial;seq=[]
for t in range(actions.shape[1]):
 pr=step(vs,state,actions[:,t],jax.device_put(data['is_first'][:,t].astype(np.float32)))
 seq.append(pr)
 state={k:post[k][:,t] if k in ['stoch','logit'] else pr[k] for k in initial}
recurrent=jax.tree_util.tree_map(lambda *x:jnp.stack(x,1),*seq)
errors={}
for k in ['hidden','deter','logit']:
 delta=np.asarray(jax.device_get(jnp.abs(prior[k]-recurrent[k])))
 errors[k]={'max':float(delta.max()),'mean':float(delta.mean())}
report=dict(code=a.code,checkpoint_step=int(saved['step']),batch=[4,128],
 source='saved real Easy replay; coupled posterior samples; complete RSSM parallel path vs recurrent img_step with obs_step reset semantics',
 reset_positions=[np.where(x)[0].tolist() for x in data['is_first']],
 errors=errors,scope='numerical consistency, not a learning result')
Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
