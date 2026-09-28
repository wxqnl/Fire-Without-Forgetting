"""Checkpoint diagnosis on full held-out episodes of the official POPGym task."""
import os
os.environ["JAX_PLATFORMS"]="cpu"
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-8:])
import argparse, sys, json, pickle
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument("--run",required=True);p.add_argument("--output",required=True)
p.add_argument("--episodes",type=int,default=16)
p.add_argument("--reference",default=None)
p.add_argument("--code",default=None)
p.add_argument("--checkpoint",default=None)
p.add_argument("--precision",choices=["float16","float32"],default=None)
a=p.parse_args()
root=Path("/data/Minko/fire-without-forgetting")
code=Path(a.code) if a.code else root/"code/r2i-v2"
sys.path[:0]=[str(code),str(code/"recall2imagine")]
import embodied, jax, jax.numpy as jnp
from recall2imagine import agent, ninjax as nj, jaxutils
import gymnasium as gym
import popgym
from ruamel.yaml import YAML
cfg=embodied.Config(YAML(typ="safe").load((Path(a.run)/"config.yaml").read_text())).update({"jax.platform":"cpu"})
env=gym.make(cfg.task.split("_",1)[1])
k=env.unwrapped.k
scale=env.unwrapped.deck.num_cards-k
obs_space={"observation":embodied.Space(np.float32,(4,),0,1),
           "reward":embodied.Space(np.float32),"is_first":embodied.Space(bool),
           "is_last":embodied.Space(bool),"is_terminal":embodied.Space(bool),
           "step_no":embodied.Space(np.int32)}
act_space={"action":embodied.Space(np.float32,(4,),0,1),"reset":embodied.Space(bool)}
inner=agent.Agent.inner(obs_space,act_space,embodied.Counter(),cfg,name="agent")
jaxutils.COMPUTE_DTYPE=getattr(jnp,a.precision or cfg.jax.precision)
checkpoint=Path(a.checkpoint) if a.checkpoint else Path(a.run)/"checkpoint.ckpt"
saved=pickle.loads(checkpoint.read_bytes())
varibs=jax.device_put(saved["agent"])
def forward(data):
    data=inner.preprocess(data)
    embed=inner.wm.encoder(data)
    post,prior=inner.wm.rssm.observe(embed,data["action"],data["is_first"])
    recon=inner.wm.heads["decoder"](post)["observation"].mode()
    reward=inner.wm.heads["reward"](post).mean()
    return dict(reconstruction=recon, reward=reward, stoch=post["stoch"],
                hidden_real=post["hidden"].real,hidden_imag=post["hidden"].imag, deter=post["deter"],
                posterior_entropy=inner.wm.rssm.get_dist(post).entropy())
pure=nj.pure(forward)
infer=jax.jit(lambda vs,rng,data: pure(vs,rng,data,create=False,modify=False)[0])
rng=np.random.default_rng(0)
episodes=[]
for i in range(a.episodes):
    ob,_=env.reset(seed=1000000 if i==0 else None)
    ep={"observation":[np.eye(4,dtype=np.float32)[ob]],
        "action":[np.zeros(4,np.float32)],"reward":[0.],
        "is_first":[True],"is_terminal":[False]}
    done=False
    while not done:
        act=int(rng.integers(4))
        ob,rew,term,trunc,_=env.step(act)
        done=term or trunc
        for key,value in dict(observation=np.eye(4,dtype=np.float32)[ob],
            action=np.eye(4,dtype=np.float32)[act],reward=rew,
            is_first=False,is_terminal=term).items():
            ep[key].append(value)
    episodes.append({key:np.asarray(value) for key,value in ep.items()})
if a.reference:
    files=sorted(Path(a.reference).glob("*.npz"))[:120]
    if len(files)!=120:raise ValueError("Need 120 shared reference episodes.")
    episodes=[]
    for f in files:
        with np.load(f) as ep:
            episodes.append({key:ep[key].copy() for key in ["observation","action","reward","is_first","is_terminal"]})
    a.episodes=len(episodes)
data={key:np.stack([ep[key] for ep in episodes]) for key in episodes[0]}
# Incoming action a[t] produced obs[t], reward[t], matching rssm.observe input.
labels=data["observation"].argmax(-1)
actions=data["action"].argmax(-1)
expected=np.zeros_like(data["reward"])
expected[:,k:]=np.where(actions[:,k:]==labels[:,:-k],1.,-1.)/scale
alignment_error=float(np.max(np.abs(expected-data["reward"])))
print("Official reward alignment max error",alignment_error,flush=True)
if alignment_error>1e-6:raise ValueError("Reward/action/observation alignment failed.")
outputs=[]
for offset in range(0,a.episodes,4):
    batch=jax.device_put({key:value[offset:offset+4] for key,value in data.items()})
    out=jax.device_get(infer(varibs,jax.random.PRNGKey(offset),batch))
    outputs.append(out)
    print("Evaluated real episodes",offset+4,flush=True)
out={key:np.concatenate([x[key] for x in outputs]) for key in outputs[0]}
truth=data["reward"][:,k:]
pred=out["reward"][:,k:]
balanced=np.mean([np.mean((pred>0)[truth>0]),np.mean((pred<0)[truth<0])])
report=dict(run=a.run,code=str(code),checkpoint=str(checkpoint),compute_precision=a.precision or cfg.jax.precision,training_compute_precision=cfg.jax.precision,
    training_step=int(saved["step"]),source=(("shared held-out trajectories: "+a.reference) if a.reference else "official POPGym full episodes; evaluation seed stream 1000000; fixed uniform action RNG 0"),
    episodes=a.episodes,k=k,scored_trials=scale,reward_alignment_max_error=alignment_error,
    reconstruction_accuracy=float(np.mean(out["reconstruction"].argmax(-1)==labels)),
    posterior_entropy_mean=float(out["posterior_entropy"].mean()),
    posterior_entropy_std=float(out["posterior_entropy"].std()),
    reward_sign_accuracy=float(np.mean((pred>0)==(truth>0))),
    reward_balanced_sign_accuracy=float(balanced),
    reward_mse=float(np.mean((pred-truth)**2)),
    random_policy_constant_reward_mse=float(np.mean((-0.5/scale-truth)**2)),
    reward_prediction_mean=float(pred.mean()),reward_prediction_std=float(pred.std()),
    predicted_positive_fraction=float(np.mean(pred>0)),actual_positive_fraction=float(np.mean(truth>0)),
    predicted_reward_on_positive=float(pred[truth>0].mean()),
    predicted_reward_on_negative=float(pred[truth<0].mean()),
    interpretation="held-out reconstruction and reward diagnosis of a fixed checkpoint; no training or architecture change")
if a.reference:
    features={}
    for view,keys in {"policy_readout":["stoch","hidden_real","hidden_imag"],"world_model_readout":["stoch","deter"],"current_observation":[]}.items():
        features[view]=np.concatenate([out[key].reshape(*labels.shape,-1) for key in keys],-1) if keys else data["observation"]
    results=[]
    for view,x in features.items():
        for delay in [1,4,8,16,32,64]:
            if delay>=x.shape[1]:continue
            train=x[:80,delay:].reshape(-1,x.shape[-1]).astype(np.float32)
            test=x[80:,delay:].reshape(-1,x.shape[-1]).astype(np.float32)
            y=labels[:80,:-delay].reshape(-1)
            truth=labels[80:,:-delay]
            mean=train.mean(0);std=np.maximum(train.std(0,ddof=1),1e-3)
            train=(train-mean)/std;test=(test-mean)/std
            target=np.eye(4,dtype=np.float32)[y];bias=target.mean(0);target-=bias
            gram=train.T@train/len(train)
            gram.flat[::len(gram)+1]+=0.1
            weights=np.linalg.solve(gram,train.T@target/len(train))
            pred=(test@weights+bias).argmax(-1).reshape(truth.shape)
            acc=(pred==truth).mean(-1)
            boot=np.random.default_rng(0).choice(acc,size=(1000,len(acc)),replace=True).mean(-1)
            row=dict(view=view,delay=delay,accuracy=float(acc.mean()),episode_bootstrap_interval=np.quantile(boot,[0.025,0.975]).tolist(),features=x.shape[-1])
            results.append(row);print(json.dumps(row),flush=True)
    report["memory_probe"]=dict(reference=a.reference,fit_episodes=80,test_episodes=40,ridge=0.1,
        split="whole episodes, disjoint; training-only standardization; sampled posterior; conditional episode intervals",
        results=results)
dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
dest.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
env.close()

