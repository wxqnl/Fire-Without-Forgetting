"""Diagnostic (EXPERIMENTS.md §19.22): train only the world model on a fixed set of recorded episodes, so that
world models can be compared on the same data (in RL each one trains on the data of its own policy).

data     the episodes of a run's replay buffer (e.g. DreamerV3's pixel walker train_eps); the last 64 files are
         held out for evaluation
train    the world model's usual loss, batches of batch_size windows of batch_length steps, each window
         starting with is_first (as the replay sampler of dreamer.py)
eval     every eval_every updates, on windows of 32 steps from the held-out episodes:
           recon   image loss of the posterior (as image_loss in training)
           kl      KL(posterior || prior)
           ol_k    image loss of the open-loop prior k steps after the last observed step (observe 16 steps,
                   then imagine with the recorded actions)
           r2_*    R^2 of a ridge read-out of the true walker velocity (9) and orientations (14) from the
                   spikes h_t (mean over the T simulation steps) and, for the PDN, from the last layer's
                   dendritic state; fit on half of the held-out episodes, tested on the other half

usage: python scripts/diag/diag_wm_offline.py --configs dmc_vision ours resonanthalf --data <episode dir> --updates 20000
           --eval_every 2000 --logdir <out dir>
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # the repository root

import numpy as np
import torch

import dreamer
import models
import probe_wm


def load(files):
    eps = []
    for f in files:
        ep = dict(np.load(f))
        eps.append({k: ep[k] for k in ("image", "action", "reward", "discount", "is_terminal", "velocity", "orientations")})
    return eps


def window(ep, start, length):
    w = {k: ep[k][start:start + length] for k in ("image", "action", "reward", "discount", "is_terminal")}
    w["is_first"] = np.zeros(length, bool)
    w["is_first"][0] = True
    return w


def stack(ws):
    return {k: np.stack([w[k] for w in ws]) for k in ws[0]}


def ridge_r2(X, Y, train):
    best = -np.inf
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    Xs = (X - mu) / sd
    ym = Y[train].mean(0)
    A = Xs[train]
    G = A.T @ A
    for lam in (0.1, 1.0, 10.0, 100.0, 1000.0):
        w = np.linalg.solve(G + lam * np.eye(A.shape[1]), A.T @ (Y[train] - ym))
        P = Xs[~train] @ w + ym
        r2 = 1 - ((Y[~train] - P) ** 2).sum(0) / ((Y[~train] - Y[~train].mean(0)) ** 2).sum(0)
        best = max(best, r2.mean())
    return best


@torch.no_grad()
def evaluate(wm, eps, device, obs=16, length=32):
    starts = (0, 100, 200, 300, 400)
    ws = [(e, window(ep, s, length)) for e, ep in enumerate(eps) for s in starts if s + length <= len(ep["image"])]
    out = {"recon": [], "kl": [], "ol1": [], "ol4": [], "ol16": []}
    feats = {"h": [], "d": []}
    targ = {"vel": [], "ori": []}
    groups = []
    index = getattr(wm.dynamics._cell, "state_index", None)
    last_mem = "mem_" + wm.dynamics._mem_keys[-1] if wm.dynamics._mem_keys else None
    for i in range(0, len(ws), 16):
        chunk = ws[i:i + 16]
        data = wm.preprocess(stack([w for _, w in chunk]))
        post, prior = wm.dynamics.observe(wm.encoder(data), data["action"], data["is_first"])
        dec = wm.heads["decoder"](wm.dynamics.get_feat(post))["image"]
        out["recon"].append(-dec.log_prob(data["image"]).float().mean().item())
        _, kl, _, _ = wm.dynamics.kl_loss(post, prior, 0.0, 0.5, 0.1)
        out["kl"].append(kl[:, 1:].float().mean().item())
        st = {k: (v[:, :, obs - 1] if k == "deter" else v[:, obs - 1]) for k, v in post.items()}
        fs = []
        for t in range(obs, length):
            st = wm.dynamics.img_step(st, data["action"][:, t], None, sample=False)
            fs.append(wm.dynamics.get_feat(st))
        ol = -wm.heads["decoder"](torch.stack(fs, 2))["image"].log_prob(data["image"][:, obs:]).float()
        for k in (1, 4, 16):
            out[f"ol{k}"].append(ol[:, k - 1].mean().item())
        h = post["deter"].float().mean(0).cpu().numpy()
        d = None
        if last_mem:
            d = post[last_mem]
            d = (d[..., index] if index is not None else d).float().cpu().numpy()
        for b, (e, w) in enumerate(chunk):
            sl = slice(8, length)
            feats["h"].append(h[b, sl])
            if d is not None:
                feats["d"].append(d[b, sl])
            groups.append(np.full(length - 8, e))
    s_of = [s for e, ep in enumerate(eps) for s in starts if s + length <= len(ep["image"])]
    for (e, _), s in zip(ws, s_of):
        targ["vel"].append(eps[e]["velocity"][s + 8:s + length])
        targ["ori"].append(eps[e]["orientations"][s + 8:s + length])
    groups = np.concatenate(groups)
    train = groups % 2 == 0
    res = {k: float(np.mean(v)) for k, v in out.items()}
    for fk, fv in feats.items():
        if fv:
            X = np.concatenate(fv)
            for tk, tv in targ.items():
                res[f"r2_{tk}_{fk}"] = float(ridge_r2(X, np.concatenate(tv), train))
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--logdir", required=True)
    args, rest = parser.parse_known_args()
    config, pargs = probe_wm.parse_config(rest + ["--task", "dmc_walker_walk", "--envs", "1", "--train_episodes", "1"])
    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    device = config.device
    env = dreamer.make_env(config, "eval")
    config.num_actions = env.action_space.shape[0]
    wm = models.WorldModel(env.observation_space, env.action_space, 0, config).to(device)
    files = sorted(pathlib.Path(args.data).glob("*.npz"))
    train_eps, eval_eps = load(files[:-64]), load(files[-64:])
    rng = np.random.default_rng(pargs.data_seed)
    metrics_file = open(logdir / "metrics.jsonl", "a")
    log, t0 = {}, time.time()
    for step in range(1, pargs.updates + 1):
        ws = []
        for _ in range(config.batch_size):
            ep = train_eps[rng.integers(len(train_eps))]
            ws.append(window(ep, rng.integers(len(ep["image"]) - config.batch_length + 1), config.batch_length))
        wm.train()
        _, _, mets = wm._train(stack(ws))
        for k in ("image_loss", "kl", "reward_loss", "deter_rate"):
            log.setdefault(k, []).append(float(mets[k]))
        if step % pargs.eval_every == 0:
            wm.eval()
            row = {"step": step, "sec_per_update": (time.time() - t0) / pargs.eval_every,
                   **{f"train_{k}": float(np.mean(v)) for k, v in log.items()}, **evaluate(wm, eval_eps, device)}
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
            log, t0 = {}, time.time()
    torch.save({"_wm." + k: v for k, v in wm.state_dict().items()}, logdir / "latest_model.pt")


if __name__ == "__main__":
    main()
