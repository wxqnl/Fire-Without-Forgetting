"""Episode-held-out linear memory probes on real, shared POPGym trajectories."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
p=argparse.ArgumentParser()
p.add_argument('--code',required=True);p.add_argument('--run',required=True)
p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
p.add_argument('--checkpoint',default=None)
p.add_argument('--posterior_sampling',choices=['sample','mode'],default='sample')
p.add_argument('--device',default='cpu');p.add_argument('--train_episodes',type=int,default=80)
p.add_argument('--test_episodes',type=int,default=40);p.add_argument('--ridge',type=float,default=0.1)
a=p.parse_args();sys.path.insert(0,a.code)
import dreamer,models,tools
torch.set_num_threads(4);tools.set_seed_everywhere(0)
run=Path(a.run);cfg=json.loads((run/'resolved_config.json').read_text());cfg.update(device=a.device,logdir=str(run))
cfg=SimpleNamespace(**cfg);env=dreamer.make_env(cfg,'eval',0)
cfg.num_actions=env.action_space.shape[0]
model=models.WorldModel(env.observation_space,env.action_space,0,cfg).to(a.device)
checkpoint=Path(a.checkpoint) if a.checkpoint else run/'latest_model.pt'
weights=torch.load(checkpoint,map_location=a.device,weights_only=False)
training_step=weights.get('environment_steps',None)
if 'agent' in weights:weights=weights['agent']
model.load_state_dict({k[len('_wm.'):]:v for k,v in weights.items() if k.startswith('_wm.')})
model.eval().requires_grad_(False)
torch.manual_seed(0)
files=sorted(Path(a.reference).glob('*.npz'))
count=a.train_episodes+a.test_episodes
if len(files)<count:raise ValueError(f'Require {count} distinct reference episodes, found {len(files)}')
files=files[:count];features={'readout':[],'full_state':[],'current_observation':[]};labels=[]
for offset in range(0,count,8):
 episodes=[]
 for f in files[offset:offset+8]:
  with np.load(f) as ep:episodes.append({k:ep[k].copy() for k in ep.files})
 lengths={len(e['observation']) for e in episodes}
 if len(lengths)!=1:raise ValueError('Probe batches require equal-length episodes; group the reference task first.')
 data={k:np.stack([ep[k] for ep in episodes]) for k in ['observation','action','reward','is_first','is_terminal']}
 with torch.no_grad():
  data=model.preprocess(data);embed=model.encoder(data);state=model.dynamics.initial(len(episodes))
  readout=[];full=[]
  for t in range(data['action'].shape[1]):
   state,_=model.dynamics.obs_step(state,data['action'][:,t],embed[:,:,t],data['is_first'][:,t],sample=(a.posterior_sampling=='sample'))
   feat=torch.cat([state['deter'].permute(1,0,2).flatten(1),state['stoch'].flatten(1)],dim=-1)
   extra=[v.flatten(1) for k,v in state.items() if k.startswith('mem_') or k.startswith('ssm_')]
   readout.append(feat.cpu());full.append(torch.cat([feat]+extra,dim=-1).cpu())
  features['readout'].append(torch.stack(readout,1))
  features['full_state'].append(torch.stack(full,1))
  features['current_observation'].append(data['observation'].cpu())
  labels.append(data['observation'].argmax(-1).cpu())
 print(f'Encoded {offset+len(episodes)}/{count} real episodes',flush=True)
features={k:torch.cat(v) for k,v in features.items()};labels=torch.cat(labels)
nt=a.train_episodes;results=[]
for view,x in features.items():
 for delay in [1,4,8,16,32,64]:
  if delay>=x.shape[1]:continue
  train=x[:nt,delay:].reshape(-1,x.shape[-1]).to(a.device)
  test=x[nt:,delay:].reshape(-1,x.shape[-1]).to(a.device)
  y=labels[:nt,:-delay].reshape(-1).to(a.device)
  truth=labels[nt:,:-delay]
  mean=train.mean(0);std=train.std(0).clamp_min(1e-3)
  train=(train-mean)/std;test=(test-mean)/std
  target=torch.nn.functional.one_hot(y,4).float();bias=target.mean(0);target=target-bias
  gram=train.T@train/len(train)
  gram.diagonal().add_(a.ridge)
  w=torch.linalg.solve(gram,train.T@target/len(train))
  pred=(test@w+bias).argmax(-1).cpu().reshape(truth.shape)
  episode_accuracy=(pred==truth).float().mean(-1).numpy()
  rng=np.random.default_rng(0)
  boot=episode_accuracy[rng.integers(0,len(episode_accuracy),size=(1000,len(episode_accuracy)))].mean(1)
  row=dict(view=view,delay=delay,accuracy=float(episode_accuracy.mean()),episode_bootstrap_interval=np.quantile(boot,[0.025,0.975]).tolist(),features=int(x.shape[-1]),fit_examples=len(train),test_examples=len(test))
  results.append(row);print(json.dumps(row),flush=True)
report=dict(run=str(run),code=a.code,checkpoint=str(checkpoint),training_step=training_step,reference=a.reference,inference=a.posterior_sampling+' posterior; same held-out trajectories and inference RNG seed 0 for all models',training_seed=0,train_probe_episodes=nt,test_probe_episodes=a.test_episodes,ridge=a.ridge,split='whole episodes, disjoint fit and test; standardization fit on train only',interval_scope='conditional episode variability; not variability across model training seeds',results=results)
p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2))
env.close()
