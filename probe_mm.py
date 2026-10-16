"""Memory Maze offline probing benchmark (Pasukonis et al., ICLR 2023), research plan E3.

Follows the protocol of the benchmark (paper §3.3, §5, Tables E.1 and E.3):

  data     the official Memory Maze 9x9 (30M) dataset: 29k training and 1k evaluation trajectories
           of 1000 steps (64x64 first-person images), recorded with a scripted policy. Read directly
           from the released zip archives (data/memory_maze_9x9_official/*.zip).
  model    the world model is trained on the images with its usual loss and truncated
           backpropagation through time (TBTT): 32 streams walk through trajectories in consecutive
           chunks of 48 steps and the state is carried from one chunk to the next.
  probe    an MLP (4 hidden layers of 1024 units, LayerNorm, ELU) reads the stop-gradient model
           state [stoch, deter] concatenated with the agent position and orientation, and predicts
             walls    maze_layout, the 9x9 wall map in global coordinates (BCE; metric: accuracy
                      averaged over the tiles)
             objects  targets_vec, the positions of the 3 objects relative to the agent (MSE;
                      metric: squared distance averaged over the objects)
           It is trained alongside the world model on the same batches (Adam, lr 3e-4, eps 1e-5).
  score    evaluation trajectories are filtered from their start; the score averages the second
           half of each trajectory (last 500 steps), after the initial exploration.

Reference numbers of the paper (9x9, 1M gradient steps): walls 80.8 (constant), 84.1 (VAE, no
memory), 89.0 (RSSM), 94.0 (RSSM TBTT); objects 23.9, 17.2, 8.3, 6.2.

Also trained, with the same MLP: a no-memory probe on the current image embedding (+ position and
orientation), the counterpart of the paper's VAE baseline for the same encoder. The difference to
the state probe is the information the state carries from the past.

usage: python probe_mm.py --configs mmprobe ours --logdir runs/mmaze/memory_maze_9x9/PD2L_tau1000/s0
"""
import argparse
import concurrent.futures
import io
import json
import pathlib
import shutil
import sys
import threading
import time
import zipfile

sys.path.append(str(pathlib.Path(__file__).parent))

import gym
import numpy as np
import torch
from torch import nn

import models
import probe_wm
import tools

KEYS = ("image", "action", "reward", "targets_vec", "maze_layout", "agent_pos", "agent_dir")


class Episodes:
    """Trajectories of an offline Memory Maze dataset: npz files inside zip archives (the official
    release) or in a directory."""

    def __init__(self, sources):
        self.items = []
        for src in sources:
            src = pathlib.Path(src)
            if src.suffix == ".zip":
                with zipfile.ZipFile(src) as zf:
                    self.items += [(src, n) for n in sorted(zf.namelist()) if n.endswith(".npz")]
            else:
                self.items += [(None, f) for f in sorted(src.glob("*.npz"))]
        self._local = threading.local()  # one ZipFile handle per thread

    def __len__(self):
        return len(self.items)

    def load(self, i):
        archive, member = self.items[i]
        if archive is None:
            raw = np.load(member)
        else:
            handles = self._local.__dict__.setdefault("zips", {})
            if archive not in handles:
                handles[archive] = zipfile.ZipFile(archive)
            raw = np.load(io.BytesIO(handles[archive].read(member)))
        ep = {k: raw[k] for k in KEYS}
        if ep["maze_layout"].ndim == 3:  # constant within the trajectory
            ep["maze_layout"] = ep["maze_layout"][0]
        ep["is_first"] = np.zeros(len(ep["reward"]), bool)
        ep["is_first"][0] = True
        return ep


def split_sources(data):
    data = pathlib.Path(data)
    train = sorted(data.glob("train*.zip")) or [data / "train"]
    evals = sorted(data.glob("eval*.zip")) or [data / "eval"]
    return train, evals


class Streams:
    """B streams of consecutive chunks through random training trajectories (TBTT). Each stream
    starts at a random offset of its first trajectory so that episode starts are not synchronised."""

    def __init__(self, episodes, batch, length, seed, workers=8):
        self.eps, self.B, self.L = episodes, batch, length
        self.rng = np.random.RandomState(seed)
        self.pool = concurrent.futures.ThreadPoolExecutor(workers)
        self.cur = [f.result() for f in [self._prefetch() for _ in range(batch)]]
        self.next_ep = [self._prefetch() for _ in range(batch)]
        self.pos = [self.rng.randint(0, len(ep["reward"]) - length) for ep in self.cur]

    def _prefetch(self):
        return self.pool.submit(self.eps.load, self.rng.randint(len(self.eps)))

    def next(self):
        for b in range(self.B):
            if self.pos[b] + self.L > len(self.cur[b]["reward"]):
                self.cur[b], self.pos[b] = self.next_ep[b].result(), 0
                self.next_ep[b] = self._prefetch()
        batch = {k: np.stack([ep[k][p: p + self.L] for ep, p in zip(self.cur, self.pos)])
                 for k in (*KEYS, "is_first") if k != "maze_layout"}
        batch["maze_layout"] = np.stack([ep["maze_layout"] for ep in self.cur])
        self.pos = [p + self.L for p in self.pos]
        return batch


def wm_batch(batch):
    n = batch["reward"].shape
    return {"image": batch["image"], "action": batch["action"].astype(np.float32),
            "reward": batch["reward"].astype(np.float32), "is_first": batch["is_first"].astype(np.float32),
            "is_terminal": np.zeros(n, np.float32), "discount": np.ones(n, np.float32)}


def last_state(post):
    return {k: (v[:, :, -1] if k == "deter" else v[:, -1]).detach() for k, v in post.items()}


class ProbeMLP(nn.Module):
    """Table E.3: 4 hidden layers of 1024 units, LayerNorm (eps 1e-3), ELU."""

    def __init__(self, dim, units=1024, layers=4):
        super().__init__()
        body = []
        for _ in range(layers):
            body += [nn.Linear(dim, units), nn.LayerNorm(units, eps=1e-3), nn.ELU()]
            dim = units
        self.body = nn.Sequential(*body)
        self.walls = nn.Linear(units, 81)
        self.objects = nn.Linear(units, 6)

    def forward(self, x):
        h = self.body(x)
        return self.walls(h), self.objects(h)


def pose(batch, device):
    return torch.tensor(np.concatenate([batch["agent_pos"], batch["agent_dir"]], -1), dtype=torch.float32, device=device)


def probe_losses(probe, feat, batch, device):
    """Per-step losses and metrics; feat (B, L, D) without the pose."""
    walls_logit, objects = probe(torch.cat([feat, pose(batch, device)], -1))
    B, L = feat.shape[:2]
    walls = torch.tensor(batch["maze_layout"].reshape(B, 1, 81), dtype=torch.float32, device=device).expand(B, L, 81)
    target = torch.tensor(batch["targets_vec"].reshape(B, L, 3, 2), dtype=torch.float32, device=device)
    walls_loss = nn.functional.binary_cross_entropy_with_logits(walls_logit, walls)
    objects_se = ((objects.view(B, L, 3, 2) - target) ** 2).sum(-1).mean(-1)  # squared distance, mean over objects
    walls_acc = ((walls_logit > 0).float() == walls).float().mean(-1)
    return walls_loss, objects_se, walls_acc


def state_feat(wm, post):
    return wm.dynamics.get_policy_feat(post).float().mean(0).detach()  # what the actor reads, averaged over the T SNN steps


def embed_feat(embed):
    return (embed.mean(0) if embed.dim() == 4 else embed).float().detach()


@torch.no_grad()
def evaluate(wm, probes, episodes, idx, config, length, pool):
    """Filter the trajectories idx from their start; per-step metrics of both probes."""
    out = {name: {"objects": [], "walls": []} for name in probes}
    for c in range(0, len(idx), 32):
        eps = list(pool.map(episodes.load, idx[c: c + 32]))
        T = len(eps[0]["reward"])
        state = None
        per = {name: {"objects": [], "walls": []} for name in probes}
        for p in range(0, T, length):
            batch = {k: np.stack([ep[k][p: p + length] for ep in eps]) for k in (*KEYS, "is_first") if k != "maze_layout"}
            batch["maze_layout"] = np.stack([ep["maze_layout"] for ep in eps])
            data = wm.preprocess(wm_batch(batch))
            embed = wm.encoder(data)
            post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"], state)
            state = last_state(post)
            feats = {"state": state_feat(wm, post), "no_memory": embed_feat(embed)}
            for name, probe in probes.items():
                _, objects_se, walls_acc = probe_losses(probe, feats[name], batch, config.device)
                per[name]["objects"].append(objects_se.cpu().numpy())
                per[name]["walls"].append(walls_acc.cpu().numpy())
        for name in probes:
            for k in ("objects", "walls"):
                out[name][k].append(np.concatenate(per[name][k], 1))
    log = {}
    for name in probes:
        for k in ("objects", "walls"):
            m = np.concatenate(out[name][k], 0)[:, 1:]  # the 1000 steps after the initial image
            half = m.shape[1] // 2
            key = f"eval_{k}_mse" if k == "objects" else f"eval_{k}_acc"
            prefix = "" if name == "state" else "nomem_"
            log[prefix + key] = float(m[:, half:].mean())
            log[prefix + key + "_curve"] = [round(float(m[:, b].mean()), 4) for b in np.array_split(np.arange(m.shape[1]), 10)]
    return log


def constant_baseline(train, evals, idx, pool, n=500):
    """Training-set mean prediction, scored like the models (second half of the evaluation data)."""
    tr = list(pool.map(train.load, range(0, len(train), max(1, len(train) // n))))
    mean_vec = np.concatenate([ep["targets_vec"][1:] for ep in tr]).reshape(-1, 3, 2).mean(0)
    wall_freq = np.stack([ep["maze_layout"] for ep in tr]).reshape(-1, 81).mean(0)
    ev = list(pool.map(evals.load, idx))
    vec = np.stack([ep["targets_vec"][501:] for ep in ev]).reshape(-1, 3, 2)
    walls = np.stack([ep["maze_layout"] for ep in ev]).reshape(-1, 81)
    return {"const_objects_mse": float(((vec - mean_vec) ** 2).sum(-1).mean()),
            "const_walls_acc": float(((wall_freq > 0.5) == walls).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/memory_maze_9x9_official")
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--mm_updates", type=int, default=200000)
    parser.add_argument("--mm_eval_every", type=int, default=10000)
    parser.add_argument("--mm_eval_eps", type=int, default=200)
    parser.add_argument("--probe_lr", type=float, default=3e-4)
    parser.add_argument("--keep", type=int, nargs="*", default=[],
                        help="also keep the checkpoint of these updates as model_<N>k.pt (scripts/diag/probe_mm_frozen.py)")
    args, rest = parser.parse_known_args()
    config, _ = probe_wm.parse_config(rest + ["--train_episodes", "1"])
    tools.set_seed_everywhere(config.seed)
    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    done = logdir / "results.json"
    if done.exists() and json.loads(done.read_text())["step"] >= args.mm_updates:
        print("already finished")
        return
    train_src, eval_src = split_sources(args.data)
    train, evals = Episodes(train_src), Episodes(eval_src)
    print(f"{len(train)} training and {len(evals)} evaluation trajectories", flush=True)
    obs_space = gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)})
    config.num_actions = 6
    wm = models.WorldModel(obs_space, gym.spaces.Discrete(6), 0, config).to(config.device)
    wm.requires_grad_(False)
    feat_dim = config.dyn_stoch * config.dyn_discrete + config.dyn_deter + wm.dynamics.policy_extra
    probes = {"state": ProbeMLP(feat_dim + 4).to(config.device),
              "no_memory": ProbeMLP(wm.embed_size + 4).to(config.device)}
    opts = {k: torch.optim.Adam(p.parameters(), lr=args.probe_lr, eps=1e-5) for k, p in probes.items()}
    pool = concurrent.futures.ThreadPoolExecutor(16)
    eval_all = np.arange(len(evals))
    eval_sub = eval_all[:: max(1, len(evals) // args.mm_eval_eps)][: args.mm_eval_eps]
    start, state = 0, None
    if (logdir / "model.pt").exists():  # resume after an interruption or to extend the budget
        ckpt = torch.load(logdir / "model.pt", map_location=config.device)
        wm.load_state_dict(ckpt["wm"])
        for k, p in probes.items():
            p.load_state_dict(ckpt["probes"][k])
        if "opts" in ckpt:  # checkpoints before 2026-10-01 hold no optimizer state: Adam restarts once
            wm._model_opt._opt.load_state_dict(ckpt["opts"]["wm"])
            wm._model_opt._scaler.load_state_dict(ckpt["opts"]["scaler"])
            for k, o in opts.items():
                o.load_state_dict(ckpt["opts"][k])
        start = ckpt["update"]
        print(f"resumed at update {start} ({'with' if 'opts' in ckpt else 'without'} optimizer state; "
              f"the TBTT state restarts)", flush=True)
    metrics = (logdir / "metrics.jsonl").open("a")
    if start == 0:
        log = {"step": 0, "train_trajectories": len(train), "eval_trajectories": len(evals),
               **constant_baseline(train, evals, eval_all, pool)}
        print(json.dumps(log), flush=True)
        metrics.write(json.dumps(log) + "\n")
    streams = Streams(train, config.batch_size, config.batch_length, config.seed + start)
    running, t0 = {}, time.time()
    for update in range(start + 1, args.mm_updates + 1):
        batch = streams.next()
        post, context, mets = wm._train(wm_batch(batch), state)
        state = last_state(post)
        feats = {"state": state_feat(wm, post), "no_memory": embed_feat(context["embed"])}
        for name, probe in probes.items():
            walls_loss, objects_se, walls_acc = probe_losses(probe, feats[name], batch, config.device)
            opts[name].zero_grad()
            (walls_loss + objects_se.mean()).backward()
            opts[name].step()
            if name == "state":
                running.setdefault("train_objects_mse", []).append(objects_se.mean().item())
                running.setdefault("train_walls_acc", []).append(walls_acc.mean().item())
        for k in ("model_loss", "image_loss", "kl", "deter_rate"):
            running.setdefault(k, []).append(float(mets.get(k, 0)))
        if update % args.mm_eval_every == 0 or update == args.mm_updates:
            log = {"step": update, **{k: float(np.mean(v)) for k, v in running.items()},
                   "sec_per_update": (time.time() - t0) / len(running["kl"])}
            final = update == args.mm_updates
            log.update(evaluate(wm, probes, evals, eval_all if final else eval_sub, config, config.batch_length, pool))
            torch.save({"wm": wm.state_dict(), "probes": {k: p.state_dict() for k, p in probes.items()},
                        "opts": {"wm": wm._model_opt._opt.state_dict(), "scaler": wm._model_opt._scaler.state_dict(),
                                 **{k: o.state_dict() for k, o in opts.items()}},
                        "update": update}, logdir / "model.pt")
            if update in args.keep:
                shutil.copy(logdir / "model.pt", logdir / f"model_{update // 1000}k.pt")
            print(json.dumps(log), flush=True)
            metrics.write(json.dumps(log) + "\n")
            metrics.flush()
            running, t0 = {}, time.time()
    (logdir / "results.json").write_text(json.dumps(log, indent=1))
    (logdir / f"results_{args.mm_updates // 1000}k.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
