"""Memory Maze offline probing on a frozen world model, with the probe heads of the benchmark's code.

The paper (Pasukonis et al., ICLR 2023, §3.3) trains the representation model first and then a
separate probe on its internal state. The benchmark code (pydreamer, models/probes.py, MapGoalsProbe)
uses a separate MLP per target:
  walls    MapProbeHead: model state + agent position and orientation -> 9x9 wall map
  objects  GoalsProbe:   model state only (targets_vec is agent-centric) -> 3 object vectors
probe_mm.py instead trains one shared MLP for both targets alongside the world model, with the pose
also given to the objects probe. This script loads a probe_mm.py checkpoint, freezes the world
model and trains the benchmark's probes from scratch on its state (and, as the no-memory baseline,
on the current image embedding), with the data, TBTT streams (32 x 48), MLP (4 x 1024, LayerNorm,
ELU), optimiser and scoring (second half of the evaluation trajectories) of probe_mm.py.

For models that carry a state besides their spikes (the dendrites of PDRNN, mem_* in the posterior),
a third probe reads the full internal state [stoch, deter, mem_*] (diagnostic: is the information
stored but not passed on by the spikes?).

usage: python probe_mm_frozen.py --configs mmprobe grucell annwm --ckpt runs/.../model.pt --logdir DIR
"""
import argparse
import concurrent.futures
import json
import pathlib
import time

import gym
import numpy as np
import torch
from torch import nn

import models
import probe_wm
import tools
from probe_mm import Episodes, Streams, embed_feat, last_state, pose, split_sources, state_feat, wm_batch

PREFIX = {"state": "", "no_memory": "nomem_", "full": "full_"}


class MLP(nn.Module):
    """Table E.3: 4 hidden layers of 1024 units, LayerNorm (eps 1e-3), ELU."""

    def __init__(self, dim, out, units=1024, layers=4):
        super().__init__()
        body = []
        for _ in range(layers):
            body += [nn.Linear(dim, units), nn.LayerNorm(units, eps=1e-3), nn.ELU()]
            dim = units
        self.net = nn.Sequential(*body, nn.Linear(units, out))

    def forward(self, x):
        return self.net(x)


class Probes(nn.Module):
    """walls: feat + pose -> 81 logits (BCE per tile); objects: feat -> 3 x 2 (squared error)."""

    def __init__(self, dim):
        super().__init__()
        self.walls, self.objects = MLP(dim + 4, 81), MLP(dim, 6)

    def losses(self, feat, batch, device):
        B, L = feat.shape[:2]
        walls_logit = self.walls(torch.cat([feat, pose(batch, device)], -1))
        walls = torch.tensor(batch["maze_layout"].reshape(B, 1, 81), dtype=torch.float32, device=device).expand(B, L, 81)
        target = torch.tensor(batch["targets_vec"].reshape(B, L, 3, 2), dtype=torch.float32, device=device)
        walls_loss = nn.functional.binary_cross_entropy_with_logits(walls_logit, walls)
        objects_se = ((self.objects(feat).view(B, L, 3, 2) - target) ** 2).sum(-1).mean(-1)
        walls_acc = ((walls_logit > 0).float() == walls).float().mean(-1)
        return walls_loss, objects_se, walls_acc


@torch.no_grad()
def features(wm, batch, state):
    data = wm.preprocess(wm_batch(batch))
    embed = wm.encoder(data)
    post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"], state)
    feats = {"state": state_feat(wm, post), "no_memory": embed_feat(embed)}
    mems = [post[k].float() for k in sorted(post) if k.startswith("mem_")]
    if mems:
        feats["full"] = torch.cat([feats["state"], *mems], -1)
    return feats, last_state(post)


@torch.no_grad()
def evaluate(wm, probes, episodes, idx, config, pool):
    out = {name: {"objects": [], "walls": []} for name in probes}
    for c in range(0, len(idx), 32):
        eps = list(pool.map(episodes.load, idx[c: c + 32]))
        T, state = len(eps[0]["reward"]), None
        per = {name: {"objects": [], "walls": []} for name in probes}
        for p in range(0, T, config.batch_length):
            batch = {k: np.stack([ep[k][p: p + config.batch_length] for ep in eps])
                     for k in ("image", "action", "reward", "targets_vec", "agent_pos", "agent_dir", "is_first")}
            batch["maze_layout"] = np.stack([ep["maze_layout"] for ep in eps])
            feats, state = features(wm, batch, state)
            for name, probe in probes.items():
                _, objects_se, walls_acc = probe.losses(feats[name], batch, config.device)
                per[name]["objects"].append(objects_se.cpu().numpy())
                per[name]["walls"].append(walls_acc.cpu().numpy())
        for name in probes:
            for k in ("objects", "walls"):
                out[name][k].append(np.concatenate(per[name][k], 1))
    log = {}
    for name in probes:
        for k in ("objects", "walls"):
            m = np.concatenate(out[name][k], 0)[:, 1:]  # the 1000 steps after the initial image
            key = PREFIX[name] + ("eval_objects_mse" if k == "objects" else "eval_walls_acc")
            log[key] = float(m[:, m.shape[1] // 2:].mean())
            log[key + "_curve"] = [round(float(m[:, b].mean()), 4) for b in np.array_split(np.arange(m.shape[1]), 10)]
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/memory_maze_9x9_official")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--probe_updates", type=int, default=100000)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--eval_eps", type=int, default=200)
    parser.add_argument("--probe_lr", type=float, default=3e-4)
    args, rest = parser.parse_known_args()
    config, _ = probe_wm.parse_config(rest + ["--train_episodes", "1"])
    tools.set_seed_everywhere(config.seed)
    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train_src, eval_src = split_sources(args.data)
    train, evals = Episodes(train_src), Episodes(eval_src)
    obs_space = gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)})
    config.num_actions = 6
    wm = models.WorldModel(obs_space, gym.spaces.Discrete(6), 0, config).to(config.device)
    ckpt = torch.load(args.ckpt, map_location=config.device)
    wm.load_state_dict(ckpt["wm"])
    wm.requires_grad_(False)
    print(f"world model from {args.ckpt} at update {ckpt['update']}", flush=True)
    feat_dim = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    probes = {"state": Probes(feat_dim).to(config.device), "no_memory": Probes(wm.embed_size).to(config.device)}
    mem_dim = sum(getattr(wm.dynamics._cell, "mem_sizes", {}).get(c, config.dyn_deter) for c in wm.dynamics._mem_keys)
    if mem_dim:
        probes["full"] = Probes(feat_dim + mem_dim).to(config.device)
    opts = {k: torch.optim.Adam(p.parameters(), lr=args.probe_lr, eps=1e-5) for k, p in probes.items()}
    pool = concurrent.futures.ThreadPoolExecutor(16)
    eval_all = np.arange(len(evals))
    eval_sub = eval_all[:: max(1, len(evals) // args.eval_eps)][: args.eval_eps]
    streams = Streams(train, config.batch_size, config.batch_length, config.seed)
    metrics = (logdir / "metrics.jsonl").open("a")
    state, running, t0 = None, {}, time.time()
    for update in range(1, args.probe_updates + 1):
        batch = streams.next()
        feats, state = features(wm, batch, state)
        for name, probe in probes.items():
            walls_loss, objects_se, walls_acc = probe.losses(feats[name], batch, config.device)
            opts[name].zero_grad()
            (walls_loss + objects_se.mean()).backward()  # separate networks: the sum trains each on its own loss
            opts[name].step()
            prefix = PREFIX[name]
            running.setdefault(prefix + "train_objects_mse", []).append(objects_se.mean().item())
            running.setdefault(prefix + "train_walls_acc", []).append(walls_acc.mean().item())
        if update % args.eval_every == 0 or update == args.probe_updates:
            final = update == args.probe_updates
            log = {"step": update, "wm_update": ckpt["update"], **{k: float(np.mean(v)) for k, v in running.items()},
                   "sec_per_update": (time.time() - t0) / len(running["train_walls_acc"])}
            log.update(evaluate(wm, probes, evals, eval_all if final else eval_sub, config, pool))
            print(json.dumps(log), flush=True)
            metrics.write(json.dumps(log) + "\n")
            metrics.flush()
            running, t0 = {}, time.time()
    (logdir / "results.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
