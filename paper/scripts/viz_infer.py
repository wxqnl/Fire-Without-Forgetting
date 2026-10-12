"""Inference for the qualitative figures of the paper, on the CPU, from checkpoints in runs/ (read only).

Writes small arrays to paper/fig/data/, which scripts/make_figures.py turns into figures:

  mmaze    Memory Maze 9x9, the first 100 evaluation trajectories of the official dataset: the object positions
           that the probe of each model predicts at every step (probe_mm.py, probe on the state the policy reads),
           converted to maze coordinates, with the true positions, the agent pose and the maze layout. The images
           stay in the dataset; make_figures.py reads the few frames it shows from there.
  walker   DMC walker walk from pixels: open-loop prediction of the world model (models.WorldModel.video_pred
           protocol: 5 observed frames, then imagination from the actions of the episode).
  dendrites  BSuite memory length: the dendritic channels of SD-Dreamer for two episodes that differ only
           in the cue, and the spikes that carry the state of the recurrent core in Spiking-WM.

usage: nice -n 19 python paper/scripts/viz_infer.py mmaze [--episodes 0 1 2]
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gym
import numpy as np
import torch

import models
import probe_mm
import probe_wm

torch.set_num_threads(8)
RUNS = ROOT / "runs"
OUT = ROOT / "paper" / "fig" / "data"

# Memory Maze runs (scripts/make_jobs.py; final model as in scripts/main_tables.py OURS["mmaze"])
MM = {"ours": ("PD2Lv2thermofbch2_tau1000", "mmprobe ours resonanthalf policymem somathermo pdfeedback pdch2"),
      "base": ("base_tau2", "mmprobe"),
      "lif": ("LIF_tau2", "mmprobe lifcell"),
      "grsn": ("GRSN_tau2", "mmprobe grsn"),
      "ann": ("ann_tau2", "mmprobe grucell annwm")}


def config_of(cfgs, extra=()):
    config, _ = probe_wm.parse_config(["--configs", *cfgs.split(), "--device", "cpu", "--train_episodes", "1", *extra])
    return config


def ego_to_maze(vec, pos, direction):
    """Memory Maze target vectors are in the agent frame, vec = [(p - a)·(dy, -dx), (p - a)·(dx, dy)]."""
    dx, dy = direction[..., None, 0], direction[..., None, 1]
    lat, fwd = vec[..., 0], vec[..., 1]
    return np.stack([pos[..., None, 0] + lat * dy + fwd * dx, pos[..., None, 1] - lat * dx + fwd * dy], -1)


@torch.no_grad()
def mmaze(args):
    evals = probe_mm.Episodes(probe_mm.split_sources(ROOT / "data" / "memory_maze_9x9_official")[1])
    idx = list(args.episodes)
    raw = [evals_raw(evals, i) for i in idx]
    out = {"episodes": np.array(idx), "targets_pos": np.stack([r["targets_pos"][0] for r in raw]).astype(np.float32)}
    preds = {name: [] for name in MM}
    for c in range(0, len(idx), args.batch):
        eps = [evals.load(i) for i in idx[c: c + args.batch]]
        for k in ("agent_pos", "agent_dir", "maze_layout"):
            out.setdefault(k, []).append(np.stack([ep[k] for ep in eps]).astype(np.float32 if k != "maze_layout" else np.uint8))
        # check the frame convention on the data before using it for the predictions
        tv, ap, ad = (np.stack([ep[k] for ep in eps]) for k in ("targets_vec", "agent_pos", "agent_dir"))
        err = np.abs(ego_to_maze(tv, ap, ad) - np.stack([r["targets_pos"] for r in raw[c: c + args.batch]])).max()
        assert err < 1e-3, err
        for name, (run, cfgs) in MM.items():
            wm, probe, config, update = mm_model(run, cfgs)
            out[f"update_{name}"] = update
            T, L = len(eps[0]["reward"]), config.batch_length
            state, vec = None, []
            for p in range(0, T, L):
                batch = {k: np.stack([ep[k][p: p + L] for ep in eps]) for k in (*probe_mm.KEYS, "is_first") if k != "maze_layout"}
                data = wm.preprocess(probe_mm.wm_batch(batch))
                embed = wm.encoder(data)
                post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"], state)
                state = probe_mm.last_state(post)
                feat = probe_mm.state_feat(wm, post)
                _, objects = probe(torch.cat([feat, probe_mm.pose(batch, "cpu")], -1))
                vec.append(objects.view(len(eps), -1, 3, 2).numpy())
            vec = np.concatenate(vec, 1)
            preds[name].append(ego_to_maze(vec, ap, ad).astype(np.float32))
            se = ((vec - tv) ** 2).sum(-1).mean(-1)[:, 1:]
            print(f"episodes {idx[c]}..{idx[c] + len(eps) - 1} {name}: update {update}, "
                  f"objects MSE on the second half {se[:, se.shape[1] // 2:].mean():.2f}", flush=True)
    for k in ("agent_pos", "agent_dir", "maze_layout"):
        out[k] = np.concatenate(out[k])
    for name in MM:
        out[f"pred_{name}"] = np.concatenate(preds[name])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "mmaze_viz.npz", **out)
    print("wrote", OUT / "mmaze_viz.npz")


_MODELS = {}


def mm_model(run, cfgs):
    """World model and state probe of a Memory Maze run (cached), from its latest checkpoint."""
    if run not in _MODELS:
        logdir = RUNS / "mmaze_official" / "memory_maze_9x9" / run / "s0"
        config = config_of(cfgs)
        config.num_actions = 6
        obs_space = gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)})
        wm = models.WorldModel(obs_space, gym.spaces.Discrete(6), 0, config)
        feat_dim = config.dyn_stoch * config.dyn_discrete + config.dyn_deter + wm.dynamics.policy_extra
        probe = probe_mm.ProbeMLP(feat_dim + 4)
        ckpt = torch.load(logdir / "model.pt", map_location="cpu")
        wm.load_state_dict(ckpt["wm"])
        probe.load_state_dict(ckpt["probes"]["state"])
        _MODELS[run] = (wm.eval(), probe.eval(), config, ckpt["update"])
    return _MODELS[run]


def evals_raw(evals, i):
    """Fields of an evaluation trajectory that probe_mm.Episodes does not load."""
    import io
    import zipfile
    archive, member = evals.items[i]
    raw = np.load(io.BytesIO(zipfile.ZipFile(archive).read(member))) if archive else np.load(member)
    return {k: raw[k] for k in ("targets_pos",)}


# DMC walker walk from pixels (scripts/make_jobs.py; ours: group dmcpix1m, the baselines: dmcpix, as paper_tables.py)
WALKER = {"ours": ("dmcpix1m", "PD2Lv2r64thermofbch2_tau1000",
                   "dmc_vision ours resonanthalf policymem aclowrank somathermo pdfeedback pdch2"),
          "base": ("dmcpix", "base_tau2", "dmc_vision"),
          "ann": ("dmcpix", "ann_tau2", "dmc_vision grucell annwm")}


def walker_model(group, run, cfgs):
    config = config_of(cfgs, ["--task", "dmc_walker_walk"])
    config.num_actions = 6
    obs_space = gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)})
    wm = models.WorldModel(obs_space, gym.spaces.Box(-1, 1, (6,), dtype=np.float32), 0, config)
    sd = torch.load(RUNS / group / "dmc_walker_walk" / run / "s0" / "latest_model.pt", map_location="cpu")
    prefix = "_wm._orig_mod." if any(k.startswith("_wm._orig_mod.") for k in sd) else "_wm."
    wm.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)})
    return wm.eval()


@torch.no_grad()
def openloop(wm, ep, starts, context, horizon):
    """Observe `context` frames from each start, then predict `horizon` frames from the actions alone."""
    sl = lambda k: np.stack([ep[k][s: s + context + horizon] for s in starts])
    n = (len(starts), context + horizon)
    data = wm.preprocess({"image": sl("image"), "action": sl("action"), "reward": sl("reward"),
                          "is_first": np.zeros(n, np.float32), "is_terminal": np.zeros(n, np.float32),
                          "discount": np.ones(n, np.float32)})
    data["is_first"][:, 0] = 1.0  # the state starts at the first observed frame
    embed = wm.encoder(data)
    post, _ = wm.dynamics.observe(embed[:, :, :context], data["action"][:, :context], data["is_first"][:, :context])
    recon = wm.heads["decoder"](wm.dynamics.get_feat(post))["image"].mode()
    state, frames = probe_mm.last_state(post), [recon]
    for h in range(horizon):
        state = wm.dynamics.img_step(state, data["action"][:, context + h])
        frames.append(wm.heads["decoder"](wm.dynamics.get_feat(state).unsqueeze(2))["image"].mode())
    return np.clip(torch.cat(frames, 1).numpy() + 0.5, 0, 1)


def walker(args):
    """The same evaluation episode of SD-Dreamer for every model (the latest one), and open-loop error over
    `--segments` starts in it."""
    import glob
    path = sorted(glob.glob(str(RUNS / "dmcpix1m" / "dmc_walker_walk" / WALKER["ours"][1] / "s0" / "eval_eps" / "*.npz")))[-1]
    ep = dict(np.load(path))
    C, H = args.context, args.horizon
    starts = list(range(args.start, len(ep["reward"]) - C - H, (len(ep["reward"]) - C - H - args.start) // args.segments))[: args.segments]
    out = {"episode": pathlib.Path(path).name, "truth": ep["image"][args.start: args.start + C + H],
           "starts": np.array(starts), "context": C}
    truth = np.stack([ep["image"][s: s + C + H] for s in starts]) / 255.0
    for name, (group, run, cfgs) in WALKER.items():
        wm = walker_model(group, run, cfgs)
        pred = openloop(wm, ep, starts, C, H)
        out[f"pred_{name}"] = (pred[0] * 255).round().astype(np.uint8)  # the segment that the figure shows
        out[f"mse_{name}"] = ((pred - truth) ** 2).mean((0, 2, 3, 4))  # per predicted step, over the segments
        print(f"{name}: open-loop MSE at steps 1, 15, 45 after the context "
              f"{out[f'mse_{name}'][C]:.4f} {out[f'mse_{name}'][C + 14]:.4f} {out[f'mse_{name}'][-1]:.4f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "walker_viz.npz", **out)
    print("wrote", OUT / "walker_viz.npz")


# BSuite memory length 101 (R2I protocol), the reported runs (scripts/main_tables.py seed())
BSUITE = {"ours": ("PD2Lv2r64thermofbch2_tau1000",
                   "bsuite ours resonanthalf policymem aclowrank somathermo pdfeedback pdch2 rewardbyact"),
          "base": ("base_tau2", "bsuite rewardbyact")}


@torch.no_grad()
def dendrites(args):
    """Two evaluation episodes that differ only in the cue (rows 0 and 1 of the observation): the difference
    of the dendritic channels of SD-Dreamer, and of the spikes that carry the state of Spiking-WM, at every step."""
    import glob
    sys.path.insert(0, str(ROOT / "scripts"))
    import main_tables as mt
    out = {}
    for name, (run, cfgs) in BSUITE.items():
        rel = f"r2i/bsuite_memlen-101/{run}"
        logdir = RUNS / rel / mt.seed(rel)
        config = config_of(cfgs, ["--task", "bsuite_memlen-101", "--batch_length", "128", "--reward_head", "symlog_mse"])
        config.num_actions = 2
        obs_space = gym.spaces.Dict({"vector": gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)})
        wm = models.WorldModel(obs_space, gym.spaces.Discrete(2), 0, config)
        sd = torch.load(logdir / "latest_model.pt", map_location="cpu")
        prefix = "_wm._orig_mod." if any(k.startswith("_wm._orig_mod.") for k in sd) else "_wm."
        wm.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)})
        wm.eval()
        if name == "ours":  # the episode: the latest evaluation episode of SD-Dreamer, used for both models
            path = sorted(glob.glob(str(logdir / "eval_eps" / "*.npz")))[-1]
            ep = dict(np.load(path))
            pair = np.stack([ep["vector"], ep["vector"]])
            pair[1, :2, 2] *= -1  # the other cue
            out["episode"], out["vector"] = pathlib.Path(path).name, pair
        n = (2, len(ep["reward"]))
        data = wm.preprocess({"vector": pair, "action": np.stack([ep["action"]] * 2).astype(np.float32),
                              "reward": np.zeros(n, np.float32), "is_first": np.stack([ep["is_first"]] * 2).astype(np.float32),
                              "is_terminal": np.zeros(n, np.float32), "discount": np.ones(n, np.float32)})
        # the mode of the stochastic state instead of a sample, so that the two episodes differ only through the cue
        obs_step = wm.dynamics.obs_step  # passes `sample` on to the prior (img_step)
        wm.dynamics.obs_step = lambda *a, **k: obs_step(*a, **{**k, "sample": False})
        posts = []
        for b in range(2):
            torch.manual_seed(0)
            post, _ = wm.dynamics.observe(wm.encoder({k: v[b: b + 1] for k, v in data.items()}),
                                          data["action"][b: b + 1], data["is_first"][b: b + 1])
            posts.append(post)
        rate = [p["deter"].float().mean(0)[0].numpy() for p in posts]  # spike rate of the recurrent core, (t, deter)
        out[f"rate_{name}"] = np.stack(rate)
        if name == "ours":
            cell = wm.dynamics._cell
            for key in [k for k in posts[0] if k.startswith("mem_d")]:
                out[key] = np.stack([p[key][0].numpy() for p in posts])  # (2, t, [a; b] of every channel)
            for i in range(len(cell.log_tau)):
                out[f"tau{i}"] = torch.exp(cell.log_tau[i]).numpy()
                out[f"theta{i}"] = (cell.theta[i] * cell.res_mask).numpy()
            out["channels"] = cell._channels
        print(f"{name}: {logdir}, mean |difference| of the core spike rate at steps 1, 5, 50, 100 after the cue "
              + " ".join(f"{np.abs(rate[0][t] - rate[1][t]).mean():.3f}" for t in (2, 6, 51, 101)), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "dendrites_viz.npz", **out)
    print("wrote", OUT / "dendrites_viz.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["mmaze", "walker", "dendrites"])
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(100)))
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=45)
    parser.add_argument("--segments", type=int, default=8)
    args = parser.parse_args()
    {"mmaze": mmaze, "walker": walker, "dendrites": dendrites}[args.what](args)


if __name__ == "__main__":
    main()
