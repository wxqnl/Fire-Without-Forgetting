"""World-model memory probe (E1/E2 of the Phase 0 plan).

Trains only the world model on random-policy data from a memory task and measures
whether its reward prediction uses information from k steps back:

  bsuite_memlen-L          reward at the last step = +1/-1 depending on whether the
                           random action matches the context bit shown L steps earlier
  popgym_RepeatPrevious*   reward at every step t > k depends on whether the random
                           action matches the card shown k steps earlier

A memoryless model cannot beat chance (AUC 0.5) on these steps, so reward-prediction
AUC on held-out full episodes is a direct read-out of how far back the world model
remembers. No actor-critic is trained, which removes exploration as a confound.

usage: python probe_wm.py --configs bsuite persist --task bsuite_memlen-16 --logdir ...
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(__file__).parent))

import numpy as np
import ruamel.yaml as yaml
import torch

import dreamer
import models
import tools


def collect(config, episodes, seed, mode):
    """Random-policy episodes in the replay format used by tools.simulate."""
    cfg = argparse.Namespace(**vars(config))
    cfg.seed = seed
    env = dreamer.make_env(cfg, mode)
    rng = np.random.RandomState(seed)
    n_act = env.action_space.shape[0]
    out = {}
    for i in range(episodes):
        obs = env.reset()
        ep = {k: [v] for k, v in obs.items()}
        ep["action"] = [np.zeros(n_act, np.float32)]
        ep["reward"] = [0.0]
        ep["discount"] = [1.0]
        done = False
        while not done:
            action = np.eye(n_act, dtype=np.float32)[rng.randint(n_act)]
            obs, reward, done, info = env.step({"action": action})
            for k, v in obs.items():
                ep[k].append(v)
            ep["action"].append(action)
            ep["reward"].append(reward)
            ep["discount"].append(float(info["discount"]))
        out[f"{mode}-{i:06d}"] = {k: tools.convert(v) for k, v in ep.items()}
    return out, env


def auc(pos, neg):
    """Mann-Whitney AUC: P(score of a positive-reward step > score of a negative one)."""
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([pos, neg]))  # ties get the average rank
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def logistic_acc(x, y):
    """Held-out accuracy of an L2-regularised logistic regression (first half trains)."""
    n = len(y) // 2
    x = (x - x[:n].mean(0)) / (x[:n].std(0) + 1e-6)
    w = torch.zeros(x.shape[1] + 1, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = x[:n] @ w[:-1] + w[-1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y[:n]) + 1e-3 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return float((((x[n:] @ w[:-1] + w[-1]) > 0).float() == y[n:]).float().mean())


@torch.no_grad()
def evaluate(wm, eval_eps, config, chunk=128):
    keys = list(next(iter(eval_eps.values())).keys())
    batch = {k: np.stack([ep[k] for ep in eval_eps.values()]) for k in keys}
    preds, rates, states, mems = [], [], [], []
    for s in range(0, len(eval_eps), chunk):
        data = wm.preprocess({k: v[s : s + chunk] for k, v in batch.items()})
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feat = wm.dynamics.get_feat(post)
        preds.append(wm.reward(feat, data["action"]).mode()[..., 0].cpu().numpy())
        rates.append(post["deter"].mean().item())
        states.append(post["deter"].float().mean(0))  # [B, rows, deter]: what the heads read
        mem_keys = sorted(k for k in post if k.startswith("mem_"))
        if mem_keys:
            mems.append(torch.cat([post[k] for k in mem_keys], -1))
    pred = np.concatenate(preds)
    true = batch["reward"]
    pos, neg = pred[true > 0], pred[true < 0]
    out = {
        "eval_auc": float(auc(pos, neg)),
        "eval_bal_acc": float(((pos > 0).mean() + (neg < 0).mean()) / 2),
        "eval_reward_mse": float(np.mean((pred - true)[true != 0] ** 2)),
        "eval_deter_rate": float(np.mean(rates)),
        "eval_pos_steps": int(len(pos)),
        "eval_neg_steps": int(len(neg)),
    }
    if config.task.startswith("bsuite") and len({len(ep["reward"]) for ep in eval_eps.values()}) == 1:
        # memory read out without the reward head: linear decodability of the cue (context
        # bit, shown in rows 0-1) from the recurrent state at every row
        cue = torch.tensor(batch["vector"][:, 0, 2] > 0, dtype=torch.float32, device=config.device)
        for name, xs in (("h", states), ("mem", mems)):
            if xs:
                x = torch.cat(xs)
                acc = [logistic_acc(x[:, t], cue) for t in range(x.shape[1])]
                out[f"eval_cue_{name}"] = [round(a, 3) for a in acc]
                out[f"eval_cue_{name}_end"] = acc[-1]
    return out


def grad_profile(wm, eval_eps, n=64):
    """|d terminal reward prediction / d embedding| per row, relative to the terminal row.

    Only for fixed-length episodes (BSuite memory_len, where the cue is in rows 0-1): how
    much of the gradient from the memory-dependent reward reaches the cue step."""
    eps = list(eval_eps.values())[:n]
    if len({len(ep["reward"]) for ep in eps}) != 1:
        return {}
    data = wm.preprocess({k: np.stack([ep[k] for ep in eps]) for k in eps[0]})
    with torch.no_grad():
        embed = wm.encoder(data)
    embed = embed.detach().requires_grad_(True)
    post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
    pred = wm.reward(wm.dynamics.get_feat(post), data["action"]).mode()[..., 0]
    pred[:, -1].sum().backward()
    g = embed.grad.abs().sum(dim=(0, 1, 3)).cpu().numpy()  # embed is [T, B, rows, dim]
    if g[-1] == 0:  # reward head is zero-initialised
        return {}
    g = g / g[-1]
    return {"eval_grad_cue": float(g[:2].max()), "eval_grad_rows": [float(f"{x:.3g}") for x in g]}


def main(config, args):
    tools.set_seed_everywhere(config.seed)
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    done_file = logdir / "results.json"
    if done_file.exists():
        print("already finished:", done_file)
        return

    # identical data for every model seed / condition of a task
    t0 = time.time()
    train_eps, env = collect(config, args.train_episodes, args.data_seed, "train")
    eval_eps, _ = collect(config, args.eval_episodes, args.data_seed + 1, "eval")
    # --raw_reward keeps the task's reward scale, as in the RL runs
    scale = 1.0 if args.raw_reward else max(np.abs(ep["reward"]).max() for ep in train_eps.values())
    for eps in (train_eps, eval_eps):  # unit-magnitude rewards, e.g. POPGym's +-1/(N-k)
        for ep in eps.values():
            ep["reward"] = (ep["reward"] / scale).astype(np.float32)
    steps = sum(len(ep["reward"]) - 1 for ep in train_eps.values())
    print(f"data: {len(train_eps)} train eps ({steps} steps), {len(eval_eps)} eval eps, "
          f"reward scale {scale:.4f}, {time.time() - t0:.0f}s")

    config.num_actions = env.action_space.shape[0]
    dataset = dreamer.make_dataset(train_eps, config)
    wm = models.WorldModel(env.observation_space, env.action_space, 0, config).to(config.device)
    wm.requires_grad_(False)

    metrics_file = (logdir / "metrics.jsonl").open("a")
    log = {"step": 0, **evaluate(wm, eval_eps, config)}
    print(json.dumps(log))
    metrics_file.write(json.dumps(log) + "\n")
    metrics_file.flush()
    t0 = time.time()
    running = {}
    for update in range(1, args.updates + 1):
        _, _, mets = wm._train(next(dataset))
        for k in ("model_loss", "reward_loss", "vector_loss", "kl", "deter_rate", "rate_loss"):
            if k in mets:
                running.setdefault(k, []).append(float(mets[k]))
        if update % args.eval_every == 0 or update == args.updates:
            log = {"step": update, **{k: float(np.mean(v)) for k, v in running.items()}}
            log.update(evaluate(wm, eval_eps, config))
            log.update(grad_profile(wm, eval_eps))
            torch.save(wm.state_dict(), logdir / "wm.pt")  # survives the hard stop
            log["sec_per_update"] = (time.time() - t0) / args.eval_every
            log["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            t0 = time.time()
            running = {}
            print(json.dumps(log))
            metrics_file.write(json.dumps(log) + "\n")
            metrics_file.flush()
    torch.save(wm.state_dict(), logdir / "wm.pt")
    done_file.write_text(json.dumps(log, indent=1))


def parse_config(argv=None):
    """(config, args) from the command line: configs.yaml blocks named by --configs, then
    overrides of any config key, as in dreamer.py."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--train_episodes", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=512)
    parser.add_argument("--train_steps", type=int, default=200000)
    parser.add_argument("--data_seed", type=int, default=12345)
    parser.add_argument("--raw_reward", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    configs = yaml.safe_load((pathlib.Path(__file__).parent / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    defaults = {}
    for name in ["defaults", *(args.configs or [])]:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args(remaining)
    config.time_limit = int(config.time_limit)
    if not args.train_episodes:
        # fixed step budget regardless of episode length
        probe_env = dreamer.make_env(config, "train")
        obs, n, done = probe_env.reset(), 0, False
        while not done:
            _, _, done, _ = probe_env.step({"action": np.eye(probe_env.action_space.shape[0], dtype=np.float32)[0]})
            n += 1
        args.train_episodes = max(1, args.train_steps // n)
    return config, args


if __name__ == "__main__":
    main(*parse_config())
