"""Held-out world-model representation and reward diagnosis, without training."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
p=argparse.ArgumentParser()
for key in ["code","run","checkpoint","reference","output"]:p.add_argument("--"+key,required=True)
a=p.parse_args();sys.path.insert(0,a.code)
import dreamer,models,tools
torch.set_num_threads(4);tools.set_seed_everywhere(0)
run=Path(a.run);cfg=json.loads((run/"resolved_config.json").read_text());cfg.update(device="cpu")
cfg=SimpleNamespace(**cfg);env=dreamer.make_env(cfg,"eval",0);cfg.num_actions=env.action_space.shape[0]
model=models.WorldModel(env.observation_space,env.action_space,0,cfg)
saved=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
model.load_state_dict({k[4:]:v for k,v in saved["agent"].items() if k.startswith("_wm.")})
model.eval().requires_grad_(False)
reward_layers={}
for name,layer in model.heads["reward"].layers.named_children():
    if "LIF" in layer.__class__.__name__:
        def record(module,inputs,output,name=name):
            flat=output.flatten(0,-2)
            reward_layers[name]=dict(spike_rate=float(output.mean()),unique_vectors=int(torch.unique(flat,dim=0).shape[0]))
        layer.register_forward_hook(record)
files=sorted(Path(a.reference).glob("*.npz"))[80:96]
episodes=[]
for f in files:
    with np.load(f) as ep:episodes.append({k:ep[k].copy() for k in ep.files})
data={k:np.stack([ep[k] for ep in episodes]) for k in ["observation","action","reward","is_first","is_terminal"]}
raw=data["reward"]
with torch.no_grad():
    data=model.preprocess(data);embed=model.encoder(data)
    post,prior=model.dynamics.observe(embed,data["action"],data["is_first"])
    feat=model.dynamics.get_feat(post)
    recon=model.heads["decoder"](feat)["observation"].mode()
    pred=model.heads["reward"](feat).mode().cpu().numpy().reshape(raw.shape)
    valid=raw!=0;truth=raw[valid];prediction=pred[valid]
    report=dict(run=a.run,code=a.code,checkpoint=a.checkpoint,training_step=saved["environment_steps"],episodes=len(files),
        reference_files=[f.name for f in files],inference="sampled posterior; actual held-out full episodes",
        encoder_spike_rate=float(embed.mean()),encoder_unique_vectors=int(torch.unique(embed.flatten(0,-2),dim=0).shape[0]),
        readout_spike_rate=float(post["deter"].mean()),readout_unique_vectors=int(torch.unique(post["deter"].flatten(0,-2),dim=0).shape[0]),
        reconstruction_accuracy=float((recon.argmax(-1)==data["observation"].argmax(-1)).float().mean()),
        posterior_entropy_mean=float(model.dynamics.get_dist(post).entropy().mean()),
        reward_sign_accuracy=float(np.mean((prediction>0)==(truth>0))),
        reward_balanced_sign_accuracy=float(np.mean([np.mean(prediction[truth>0]>0),np.mean(prediction[truth<0]<0)])),
        reward_mse=float(np.mean((prediction-truth)**2)),
        random_policy_constant_reward_mse=float(np.mean((-0.5*np.abs(truth)-truth)**2)),
        reward_prediction_mean=float(prediction.mean()),reward_prediction_std=float(prediction.std()),
        predicted_positive_fraction=float(np.mean(prediction>0)),actual_positive_fraction=float(np.mean(truth>0)),
        predicted_reward_on_positive=float(prediction[truth>0].mean()),predicted_reward_on_negative=float(prediction[truth<0].mean()))
report["reward_hidden_layers"]=reward_layers
Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));env.close()

