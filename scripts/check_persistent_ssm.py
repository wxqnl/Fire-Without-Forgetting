"""Integration and E4 numerical audit on actual POPGym replay, using the full model."""
import argparse,sys,json,time
from pathlib import Path
import numpy as np
import torch
import ruamel.yaml as yaml
from types import SimpleNamespace
ROOT=Path('/data/Minko/fire-without-forgetting')
parser=argparse.ArgumentParser();parser.add_argument('--code',default=str(ROOT/'code/persistent-ssm-v2'));parser.add_argument('--output',default=str(ROOT/'docs/persistent_ssm_v2_integration.json'));args=parser.parse_args()
CODE=Path(args.code)
sys.path.insert(0,str(CODE))
import dreamer,tools

def config():
 allcfg=yaml.safe_load((CODE/'configs.yaml').read_text())
 cfg={}
 def merge(x,y):
  for k,v in y.items():
   if isinstance(v,dict) and k in x: merge(x[k],v)
   else:x[k]=v
 for key in ['defaults','memory','persistent']:merge(cfg,allcfg[key])
 cfg.update(num_actions=4,device='cuda:0',logdir=str(ROOT/'runs/ssm_integration'),task='popgym_RepeatPreviousMedium')
 return SimpleNamespace(**cfg)

torch.set_num_threads(4);tools.set_seed_everywhere(0)
cfg=config();env=dreamer.make_env(cfg,'train',0)
run=Path(cfg.logdir);run.mkdir(exist_ok=True)
eps=tools.load_episodes(ROOT/'runs/phase0_initial/original_tau2_seed0/train_eps',limit=8000)
dataset=dreamer.make_dataset(eps,cfg);data=next(dataset)
agent=dreamer.Dreamer(env.observation_space,env.action_space,cfg,tools.Logger(run,0),dataset).to(cfg.device)
agent.requires_grad_(requires_grad=False)
dyn=agent._wm.dynamics
obs=agent._wm.preprocess(data);embed=agent._wm.encoder(obs)
encoder_spike_rate=float(embed.detach().mean())
encoder_unique_vectors=len(torch.unique(embed.detach().flatten(0,-2),dim=0))
assert encoder_spike_rate>0 and encoder_unique_vectors>=4,(encoder_spike_rate,encoder_unique_vectors)
with torch.no_grad():
 post,prior=dyn.observe(embed,obs['action'],obs['is_first'],sample=False)
 state=dyn.initial(cfg.batch_size);seq=[]
 for t in range(cfg.batch_length):
  state,_=dyn.obs_step(state,obs['action'][:,t],embed[:,:,t],obs['is_first'][:,t],sample=False)
  seq.append(state)
 serial={k:torch.stack([x[k] for x in seq],dim=2 if k=='deter' else 1) for k in state}
 errors={k:float((post[k]-serial[k]).abs().max()) for k in post}
 assert errors['ssm_real']<2e-5 and errors['ssm_imag']<2e-5,errors
 assert errors['deter']==0.0 and errors['stoch']==0.0,errors
 # Compare affine scan and step gradients on the same real latent/action inputs.
 st=dyn.initial(cfg.batch_size)
 prev=torch.cat([st['stoch'][:,None],post['stoch'][:,:-1]],dim=1)
 x=torch.cat([prev.flatten(-2),obs['action']],dim=-1)*(1-obs['is_first'][...,None])
layer=dyn.layers[0]
with tools.RequiresGrad(layer):
 pars=list(layer.parameters())[:6]
 z=torch.zeros(cfg.batch_size,cfg.ssm_states,device=cfg.device,dtype=torch.complex64)
 _,ds=layer(x.detach(),z,obs['is_first'])
 loss=(ds.real.square()+ds.imag.square()).mean()
 gs=torch.autograd.grad(loss,pars,allow_unused=True)
 values=[]
 for t in range(cfg.batch_length):
  _,z=layer.step(x[:,t].detach(),z,obs['is_first'][:,t]);values.append(z)
 ds2=torch.stack(values,1)
 loss2=(ds2.real.square()+ds2.imag.square()).mean()
 gt=torch.autograd.grad(loss2,pars,allow_unused=True)
 grad_error=max(float((a-b).abs().max()) for a,b in zip(gs,gt) if a is not None)
 assert grad_error<1e-4,grad_error
encoder_gradients={}
with tools.RequiresGrad(agent._wm.encoder):
 handles=[p.register_hook(lambda grad,name=name: None if grad is None else encoder_gradients.__setitem__(name,grad.detach().square().sum())) for name,p in agent._wm.encoder.named_parameters()]
updates=[]
for i in range(3):
 encoder_gradients.clear()
 agent._train(next(dataset))
 metrics={k:float(v[-1]) for k,v in agent._metrics.items() if len(v)}
 assert all(np.isfinite(v) for v in metrics.values()),metrics
 assert metrics['model_grad_norm']>0 and metrics['actor_grad_norm']>0
 encoder_grad_norm=float(torch.sqrt(sum(encoder_gradients.values())))
 assert encoder_grad_norm>0,encoder_grad_norm
 metrics['encoder_grad_norm']=encoder_grad_norm
 updates.append({k:metrics[k] for k in ['model_loss','model_grad_norm','actor_grad_norm','value_grad_norm','ssm_spike_rate','ssm_state_rms','encoder_grad_norm']})
result={'status':'passed','code':str(CODE),'encoder_spike_rate':encoder_spike_rate,'encoder_unique_vectors':encoder_unique_vectors,'source':'real POPGym replay; full 512-wide 3-layer complex SSM, 32x32 WTA latent, all decoder/reward/continuation/actor/critic updates','scan_step_max_errors':errors,'scan_step_gradient_max_error':grad_error,'updates':updates,'world_model_parameters':sum(p.numel() for p in agent._wm.parameters())}
Path(args.output).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
env.close()
