"""Memory Maze offline memory probe (research plan E3).

Trains a world model on the offline Memory Maze data of gen_memory_maze.py (9x9 mazes, 1000-step
episodes of 64x64 first-person images) and, on its state, a probe that reads out what the agent
must remember about the maze:

  objects  targets_vec: the positions of the 3 objects relative to the agent (MSE)
  walls    maze_layout: the 9x9 layout of walls and corridors (per-cell accuracy)

The probe is an MLP on the stop-gradient features [stoch, deter] and is trained alongside the world
model on the same batches, as in the Memory Maze offline probing benchmark. Training uses truncated
backpropagation through time: B streams walk through episodes in consecutive chunks of
batch_length steps and the posterior state is carried from one chunk to the next, so the model can
use the whole episode. Evaluation filters held-out episodes from the start in the same way and
reports the probe metrics per step; the headline numbers average the second half of the
episode, when most of the maze has been seen.

usage:
  python probe_mm.py --prepare --data data/memory_maze_9x9            # once: memmap cache
  python probe_mm.py --configs mmprobe ours --data data/memory_maze_9x9 --logdir runs/mmaze/...
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(__file__).parent))

import gym
import numpy as np
import torch
from torch import nn

import models
import probe_wm
import tools

LENGTH = 1001  # steps per episode, including the reset step
STEP_KEYS = ("image", "action", "reward", "is_first", "targets_vec")


def prepare(data):
    """npz episodes -> one uint8/float32 memmap per key and split, shared by all runs."""
    for split in ("train", "eval"):
        files = sorted((data / split).glob("*-1001.npz"))
        cache = data / f"{split}_cache"
        if (cache / "done").exists():
            print(split, "cache exists")
            continue
        cache.mkdir(exist_ok=True)
        first = np.load(files[0])
        arrays = {}
        for k in STEP_KEYS:
            arrays[k] = np.lib.format.open_memmap(cache / f"{k}.npy", "w+", first[k].dtype, (len(files),) + first[k].shape)
        arrays["maze_layout"] = np.lib.format.open_memmap(cache / "maze_layout.npy", "w+", np.uint8, (len(files), 9, 9))
        for i, f in enumerate(files):
            ep = np.load(f)
            for k in STEP_KEYS:
                arrays[k][i] = ep[k]
            arrays["maze_layout"][i] = ep["maze_layout"][0]  # constant within an episode
            if (i + 1) % 200 == 0:
                print(split, i + 1, "/", len(files), flush=True)
        for a in arrays.values():
            a.flush()
        (cache / "done").write_text(str(len(files)))


def load(data, split):
    cache = data / f"{split}_cache"
    assert (cache / "done").exists(), f"run --prepare first ({cache})"
    return {k: np.load(cache / f"{k}.npy", mmap_mode="r") for k in (*STEP_KEYS, "maze_layout")}


class Streams:
    """B streams of consecutive chunks through randomly chosen episodes."""

    def __init__(self, arrays, batch, length, seed):
        self.a, self.B, self.L = arrays, batch, length
        self.rng = np.random.RandomState(seed)
        self.ep = self.rng.randint(len(arrays["image"]), size=batch)
        self.pos = np.zeros(batch, int)

    def next(self):
        for b in range(self.B):
            if self.pos[b] + self.L > LENGTH:
                self.ep[b], self.pos[b] = self.rng.randint(len(self.a["image"])), 0
        idx = [(e, slice(p, p + self.L)) for e, p in zip(self.ep, self.pos)]
        batch = {k: np.stack([self.a[k][e, s] for e, s in idx]) for k in STEP_KEYS}
        batch["maze_layout"] = np.stack([self.a["maze_layout"][e] for e in self.ep])
        self.pos += self.L
        return batch


def wm_batch(batch):
    n = batch["reward"].shape
    return {"image": batch["image"], "action": batch["action"].astype(np.float32),
            "reward": batch["reward"].astype(np.float32), "is_first": batch["is_first"].astype(np.float32),
            "is_terminal": np.zeros(n, np.float32), "discount": np.ones(n, np.float32)}


def last_state(post):
    return {k: (v[:, :, -1] if k == "deter" else v[:, -1]).detach() for k, v in post.items()}


class Probe(nn.Module):
    def __init__(self, dim, units=512):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim, units), nn.LayerNorm(units), nn.SiLU(),
                                  nn.Linear(units, units), nn.LayerNorm(units), nn.SiLU())
        self.walls = nn.Linear(units, 81)
        self.objects = nn.Linear(units, 6)

    def forward(self, feat):
        h = self.body(feat)
        return self.walls(h), self.objects(h)


def features(wm, post):
    return wm.dynamics.get_feat(post).float().mean(0).detach()  # average over the T SNN steps


def probe_losses(probe, feat, batch, device):
    walls_logit, objects = probe(feat)
    walls = torch.tensor(batch["maze_layout"].reshape(len(feat), 1, 81), dtype=torch.float32, device=device)
    walls = walls.expand_as(walls_logit)
    target = torch.tensor(batch["targets_vec"].reshape(*feat.shape[:2], 6), dtype=torch.float32, device=device)
    walls_loss = nn.functional.binary_cross_entropy_with_logits(walls_logit, walls)
    objects_mse = ((objects - target) ** 2).mean(-1)
    walls_acc = ((walls_logit > 0).float() == walls).float().mean(-1)
    return walls_loss, objects_mse, walls_acc


@torch.no_grad()
def evaluate(wm, probe, arrays, config, eps, length):
    """Filter the held-out episodes eps from their start; probe metrics per step."""
    steps = (LENGTH // length) * length
    mse = np.zeros((len(eps), steps))
    acc = np.zeros((len(eps), steps))
    for chunk in range(0, len(eps), 32):
        sel = eps[chunk: chunk + 32]
        state = None
        for p in range(0, steps, length):
            batch = {k: np.stack([arrays[k][e, p: p + length] for e in sel]) for k in STEP_KEYS}
            batch["maze_layout"] = np.stack([arrays["maze_layout"][e] for e in sel])
            data = wm.preprocess(wm_batch(batch))
            post, _ = wm.dynamics.observe(wm.encoder(data), data["action"], data["is_first"], state)
            state = last_state(post)
            _, objects_mse, walls_acc = probe_losses(probe, features(wm, post), batch, config.device)
            mse[chunk: chunk + len(sel), p: p + length] = objects_mse.cpu().numpy()
            acc[chunk: chunk + len(sel), p: p + length] = walls_acc.cpu().numpy()
    half = steps // 2
    bins = np.array_split(np.arange(steps), 10)
    return {
        "eval_objects_mse": float(mse[:, half:].mean()),
        "eval_walls_acc": float(acc[:, half:].mean()),
        "eval_objects_mse_curve": [round(float(mse[:, b].mean()), 3) for b in bins],
        "eval_walls_acc_curve": [round(float(acc[:, b].mean()), 4) for b in bins],
    }


def unseen_mazes(train, evals):
    """Held-out episodes whose maze layout does not occur in the training data (186 of 200)."""
    seen = {layout.tobytes() for layout in np.asarray(train["maze_layout"])}
    return np.array([i for i, layout in enumerate(np.asarray(evals["maze_layout"])) if layout.tobytes() not in seen])


def baselines(train, evals, eps):
    """Constant predictors from the training data: the probe must beat these."""
    mean_vec = np.asarray(train["targets_vec"][:200, ::10]).reshape(-1, 6).mean(0)
    ev = np.asarray(evals["targets_vec"][eps, 500:]).reshape(-1, 6)
    freq = np.asarray(train["maze_layout"]).reshape(-1, 81).mean(0)
    walls = np.asarray(evals["maze_layout"][eps]).reshape(-1, 81)
    return {"baseline_objects_mse": float(((ev - mean_vec) ** 2).mean()),
            "baseline_walls_acc": float(((freq > 0.5) == walls).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/memory_maze_9x9")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--logdir")
    parser.add_argument("--mm_updates", type=int, default=20000)
    parser.add_argument("--mm_eval_every", type=int, default=2000)
    parser.add_argument("--mm_eval_eps", type=int, default=64)
    parser.add_argument("--probe_lr", type=float, default=3e-4)
    args, rest = parser.parse_known_args()
    data = pathlib.Path(args.data)
    if args.prepare:
        prepare(data)
        return
    config, _ = probe_wm.parse_config(rest + ["--train_episodes", "1"])
    tools.set_seed_everywhere(config.seed)
    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    if (logdir / "results.json").exists():
        print("already finished")
        return
    train, evals = load(data, "train"), load(data, "eval")
    obs_space = gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)})
    act_space = gym.spaces.Discrete(6)
    config.num_actions = 6
    wm = models.WorldModel(obs_space, act_space, 0, config).to(config.device)
    wm.requires_grad_(False)
    feat_dim = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    probe = Probe(feat_dim).to(config.device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.probe_lr)
    streams = Streams(train, config.batch_size, config.batch_length, config.seed)
    metrics = (logdir / "metrics.jsonl").open("a")
    eval_eps = unseen_mazes(train, evals)
    log = {"step": 0, "eval_episodes": len(eval_eps), **baselines(train, evals, eval_eps)}
    print(json.dumps(log))
    metrics.write(json.dumps(log) + "\n")
    state, running, t0 = None, {}, time.time()
    for update in range(1, args.mm_updates + 1):
        batch = streams.next()
        post, _, mets = wm._train(wm_batch(batch), state)
        state = last_state(post)
        walls_loss, objects_mse, walls_acc = probe_losses(probe, features(wm, post), batch, config.device)
        opt.zero_grad()
        (walls_loss + objects_mse.mean()).backward()
        opt.step()
        for k, v in (("model_loss", mets["model_loss"]), ("image_loss", mets.get("image_loss", 0)),
                     ("kl", mets["kl"]), ("deter_rate", mets["deter_rate"]),
                     ("train_objects_mse", objects_mse.mean().item()), ("train_walls_acc", walls_acc.mean().item())):
            running.setdefault(k, []).append(float(v))
        if update % args.mm_eval_every == 0 or update == args.mm_updates:
            log = {"step": update, **{k: float(np.mean(v)) for k, v in running.items()}}
            log.update(evaluate(wm, probe, evals, config, eval_eps[:args.mm_eval_eps], config.batch_length))
            log["sec_per_update"] = (time.time() - t0) / args.mm_eval_every
            torch.save({"wm": wm.state_dict(), "probe": probe.state_dict()}, logdir / "model.pt")
            print(json.dumps(log), flush=True)
            metrics.write(json.dumps(log) + "\n")
            metrics.flush()
            running, t0 = {}, time.time()
    log.update({f"final_{k}": v for k, v in evaluate(wm, probe, evals, config, eval_eps, config.batch_length).items()})
    (logdir / "results.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
