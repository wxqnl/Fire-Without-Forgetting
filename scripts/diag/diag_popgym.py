"""What the world model of a POPGym RepeatPrevious run holds about past cards.

Loads the world model of an RL run (latest_model.pt) or of a probe_wm.py run (wm.pt) and
filters held-out random-action episodes through it (posterior). Row t holds the observation o_t
and the action a_t that led to it; its reward is +-1/(N-k) depending on whether a_t equals the suit of o_{t-k}. Reports

  reward_acc   argmax over actions of the per-action reward prediction at row t equals the suit
               of o_{t-k}: what the actor can learn from in imagination (0.25 = chance)
  lag_<set>    held-out accuracy of a 4-class logistic regression that reads the suit of
               o_{t-j}, j = 0..8, from the features at row t: z (stochastic state), h (deter,
               the spikes the heads read), mem (PDRNN dendrites), z+h (the heads' input)
  tau          quantiles of the learned dendritic time constants (PDRNN)

With --init the same model is read at its initialisation (no checkpoint): what the architecture
holds before any training.

usage: python scripts/diag/diag_popgym.py [--init] runs/r2i/popgym_RepeatPreviousEasy/<cell>/s0 ...
"""
import argparse
import json
import pathlib
import shlex
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[0] = str(ROOT)  # the modules live in the repository root

import numpy as np
import torch

import models
import probe_wm

LAGS = range(9)


def job_argv(logdir):
    for line in (ROOT / "runs" / "jobs.jsonl").open():
        job = json.loads(line)
        if pathlib.Path(job["logdir"]).resolve() == logdir.resolve():
            argv = shlex.split(job["cmd"])
            script = next(a for a in argv if a in ("dreamer.py", "probe_wm.py"))
            return argv[argv.index(script) + 1:]
    raise KeyError(logdir)


def softmax_probe(x, y, n_train, classes=4):
    """Held-out accuracy of an L2-regularised softmax regression (rows < n_train train)."""
    mu, sd = x[:n_train].mean(0), x[:n_train].std(0) + 1e-6
    x = (x - mu) / sd
    w = torch.zeros(x.shape[1] + 1, classes, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = x[:n_train] @ w[:-1] + w[-1]
        loss = torch.nn.functional.cross_entropy(logits, y[:n_train]) + 1e-3 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    pred = (x[n_train:] @ w[:-1] + w[-1]).argmax(-1)
    return float((pred == y[n_train:]).float().mean())


@torch.no_grad()
def diagnose(logdir, init, episodes, seed):
    config, _ = probe_wm.parse_config(job_argv(logdir))
    config.device = "cuda"
    eps, env = probe_wm.collect(config, episodes, seed, "eval")
    config.num_actions = env.action_space.shape[0]
    wm = models.WorldModel(env.observation_space, env.action_space, 0, config).to(config.device)
    if not init and (logdir / "wm.pt").exists():  # probe_wm.py: world model only
        wm.load_state_dict(torch.load(logdir / "wm.pt", map_location=config.device))
    elif not init:  # dreamer.py: the whole agent
        state = torch.load(logdir / "latest_model.pt", map_location=config.device)
        state = {k[len("_wm."):].replace("_orig_mod.", ""): v for k, v in state.items() if k.startswith("_wm.")}
        wm.load_state_dict(state)  # _orig_mod.: torch.compile
    wm.eval()
    k = {"Easy": 4, "Medium": 32, "Hard": 64}[next(d for d in ("Easy", "Medium", "Hard") if config.task.endswith(d))]

    batch = {key: np.stack([ep[key] for ep in eps.values()]) for key in next(iter(eps.values()))}
    suit = batch["vector"].argmax(-1)  # [B, L]
    act = batch["action"].argmax(-1)
    rew = batch["reward"]
    L = suit.shape[1]
    # the reward convention this diagnosis relies on: r_t = + iff a_t == suit(o_{t-k})
    assert np.all((rew[:, k:] > 0) == (act[:, k:] == suit[:, :-k])), "reward convention"

    feats = {"z": [], "h": [], "mem": []}
    reward_pred = []
    for s in range(0, len(suit), 128):
        data = wm.preprocess({key: v[s: s + 128] for key, v in batch.items()})
        post, _ = wm.dynamics.observe(wm.encoder(data), data["action"], data["is_first"])
        feat = wm.dynamics.get_feat(post)
        pred = wm.heads["reward"](feat)._mode
        reward_pred.append((pred.mean(0) if pred.dim() == 4 else pred).float())
        stoch = post["stoch"]
        feats["z"].append(stoch.reshape(*stoch.shape[:2], -1).float())
        feats["h"].append(post["deter"].float().mean(0))  # mean over the T spike steps
        mem = sorted(key for key in post if key.startswith("mem_"))
        if mem:
            feats["mem"].append(torch.cat([post[key] for key in mem], -1).float())
    feats = {name: torch.cat(v) for name, v in feats.items() if v}
    feats["z+h"] = torch.cat([feats["z"], feats["h"]], -1)
    reward_pred = torch.cat(reward_pred).cpu().numpy()  # [B, L, actions]

    out = {"run": str(logdir.relative_to(ROOT)), "init": init, "k": k,
           "reward_acc": float((reward_pred[:, k:].argmax(-1) == suit[:, :-k]).mean()),
           "h_rate": float(feats["h"].mean())}
    rows = slice(max(LAGS), L)  # the same rows for every lag
    n_train = (len(suit) // 2) * (L - max(LAGS))
    for name, x in feats.items():
        x = x[:, rows].reshape(-1, x.shape[-1])
        accs = []
        for j in LAGS:
            y = torch.as_tensor(suit[:, max(LAGS) - j: L - j], device=x.device).reshape(-1)
            accs.append(round(softmax_probe(x, y, n_train), 3))
        out[f"lag_{name}"] = accs
    cell = wm.dynamics._cell
    if hasattr(cell, "log_tau"):
        out["tau"] = {f"layer{i}": [round(float(q), 1) for q in torch.quantile(
            lt.detach().exp().float(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=lt.device))]
            for i, lt in enumerate(cell.log_tau)}
        out["tau_frac_below_8"] = [round(float((lt.detach().exp() < 8).float().mean()), 3) for lt in cell.log_tau]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    for run in args.runs:
        print(json.dumps(diagnose(pathlib.Path(run).resolve(), args.init, args.episodes, args.seed)), flush=True)
