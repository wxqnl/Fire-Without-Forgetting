"""Numerical scan/step agreement on an archived trained world model."""
import argparse, sys, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
p = argparse.ArgumentParser()
for key in ["code", "run", "reference", "output", "checkpoint"]:
    p.add_argument("--" + key, required=True)
a = p.parse_args()
sys.path.insert(0, a.code)
import dreamer, models, tools
torch.set_num_threads(4)
tools.set_seed_everywhere(0)
run = Path(a.run)
cfg = json.loads((run / "resolved_config.json").read_text())
cfg.update(device="cpu")
cfg = SimpleNamespace(**cfg)
env = dreamer.make_env(cfg, "eval", 0)
cfg.num_actions = env.action_space.shape[0]
model = models.WorldModel(env.observation_space, env.action_space, 0, cfg)
checkpoint = Path(a.checkpoint)
saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
model.load_state_dict({k[4:]: v for k, v in saved["agent"].items() if k.startswith("_wm.")})
model.eval().requires_grad_(False)
files = sorted(Path(a.reference).glob("*.npz"))[80:84]
if len(files) != 4:
    raise ValueError("Need the shared held-out reference episodes 80:84.")
eps = []
for f in files:
    with np.load(f) as ep:
        eps.append({k: ep[k].copy() for k in ep.files})
data = {k: np.stack([ep[k] for ep in eps]) for k in
        ["observation", "action", "reward", "is_first", "is_terminal"]}
results = []
with torch.no_grad():
    data = model.preprocess(data)
    embed = model.encoder(data)
    for sampling in ["mode", "sample"]:
        torch.manual_seed(0)
        post, _ = model.dynamics.observe(
            embed, data["action"], data["is_first"], sample=(sampling == "sample"))
        initial = {k: v[:, :, 31] if k == "deter" else v[:, 31] for k, v in post.items()}
        for horizon in [15, 50, 100]:
            prev = initial
            inputs, states, rngs = [], [], []
            for t in range(horizon):
                action = data["action"][:, t + 1]
                inputs.append(torch.cat([prev["stoch"].flatten(-2), action], -1))
                rngs.append(torch.get_rng_state())
                prev = model.dynamics.img_step(prev, action, sample=(sampling == "sample"))
                states.append(prev)
            seq = {k: torch.stack([s[k] for s in states], dim=2 if k == "deter" else 1)
                   for k in prev}
            x = torch.stack(inputs, 1)
            spikes, re, im = model.dynamics.filter(x, initial, x.new_zeros(x.shape[:2]), parallel=True)
            logits = model.dynamics.prior_stats(spikes)
            if sampling == "sample":
                # Reuse exactly each sequential draw, not newly sampled independent latents.
                saved_rng = torch.get_rng_state()
                zs = []
                for t in range(horizon):
                    torch.set_rng_state(rngs[t])
                    zs.append(model.dynamics.sample(logits[:, t], True))
                torch.set_rng_state(saved_rng)
                z = torch.stack(zs, 1)
            else:
                z = model.dynamics.sample(logits, False)
            par = dict(deter=spikes.unsqueeze(0), ssm_real=re, ssm_imag=im, logit=logits, stoch=z)
            rseq = model.heads["reward"](model.dynamics.get_feat(seq)).mode()
            rpar = model.heads["reward"](model.dynamics.get_feat(par)).mode()
            discounts = cfg.discount ** torch.arange(horizon)[None, :, None]
            row = dict(
                sampling=sampling, horizon=horizon,
                state_max_error=max(float((seq[k]-par[k]).abs().max()) for k in ["ssm_real","ssm_imag"]),
                prior_logits_max_error=float((seq["logit"]-logits).abs().max()),
                spike_mismatch_fraction=float((seq["deter"] != par["deter"]).float().mean()),
                latent_mismatch_fraction=float((seq["stoch"] != par["stoch"]).float().mean()),
                reward_max_error=float((rseq-rpar).abs().max()),
                discounted_return_max_error=float(((rseq-rpar)*discounts).sum(1).abs().max()))
            results.append(row)
            print(json.dumps(row), flush=True)
report = dict(
    run=str(run), code=a.code, checkpoint=str(checkpoint),
    training_step=saved["environment_steps"], reference_files=[f.name for f in files],
    mode="conditional scan on the same sequentially generated latent/action inputs; mode and coupled sampled latent paths",
    results=results,
    interpretation="finite-precision consistency of the exact recurrence on these held-out inputs; not reward-prediction accuracy or an independently parallelized autoregressive imagination")
out=Path(a.output)
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2))
env.close()

