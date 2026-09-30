"""Configuration review of the main-table runs (EXPERIMENTS.md §14).

For every benchmark of the paper, resolves the full configuration of each model's job exactly as
dreamer.py / probe_mm.py do (configs.yaml defaults, the job's config blocks, its command-line
overrides) and prints
  - the protocol settings shared by all models of the benchmark (they must be identical), and
  - every key in which the models differ (it must only be the recurrent cell and the spiking /
    ANN choice of the model itself).
With --params also the parameter counts of the world model and of the actor-critic per model.

usage: python scripts/review_configs.py [--params]
"""
import argparse
import json
import pathlib
import shlex
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)  # not scripts/, whose queue.py would shadow the standard library's

import ruamel.yaml as yaml

import tools

BENCHMARKS = {  # name: job-name prefix of the main-table runs
    "BSuite L8-L64": "rl2/bsuite_memlen-",
    "BSuite L51-L101 (R2I)": "r2i/bsuite_memlen-",
    "POPGym (R2I)": "popgym/",
    "Memory Maze (offline probe)": "mmaze_official/",
    "DMC proprio": "dmc/",
}
SHOW = ["steps", "train_ratio", "envs", "batch_size", "batch_length", "action_repeat", "time_limit",
        "reward_head", "reward_by_action", "reward_scale", "value_head", "discount", "discount_lambda",
        "imag_horizon", "model_lr", "actor_lr", "value_lr", "dataset_size", "prefill", "dyn_stoch",
        "dyn_discrete", "dyn_deter", "dyn_hidden", "units", "reward_layers", "actor_layers",
        "value_layers", "encoder", "decoder", "eval_episode_num", "eval_every", "env_seed_per_env",
        "actor_entropy", "kl_free", "dyn_scale", "rep_scale", "grad_clip", "unimix_ratio"]
PROBE_ARGS = ("--data", "--mm_updates", "--mm_eval_every", "--mm_eval_eps", "--probe_lr")


def resolve(argv):
    """The config dreamer.py builds from argv (without the script name)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args(argv)
    configs = yaml.safe_load((ROOT / "configs.yaml").read_text())

    def update(base, new):
        for key, value in new.items():
            if isinstance(value, dict) and key in base:
                update(base[key], value)
            else:
                base[key] = value

    defaults = {}
    for name in ["defaults", *(args.configs or [])]:
        update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items()):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config, unknown = parser.parse_known_args(remaining)
    return vars(config), unknown


def job_config(job):
    argv = shlex.split(job["cmd"])
    script = next(a for a in argv if a.endswith(".py"))
    argv = argv[argv.index(script) + 1:]
    extra = {}
    for flag in PROBE_ARGS:  # probe_mm.py's own arguments
        if flag in argv:
            i = argv.index(flag)
            extra[flag[2:]] = argv[i + 1]
            del argv[i: i + 2]
    config, unknown = resolve(argv)
    config.update(extra)
    config["_script"] = script
    return config, unknown


def params(config, job):
    import gym
    import numpy as np
    import torch

    import models

    cfg = argparse.Namespace(**config)
    cfg.device = "cpu"
    task = cfg.task
    if job["name"].startswith("mmaze"):
        obs = gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)})
        n_act = 6
    else:
        import dreamer
        env = dreamer.make_env(cfg, "eval")
        obs, n_act = env.observation_space, env.action_space.shape[0]
    cfg.num_actions = n_act
    act_space = gym.spaces.Discrete(n_act) if job["name"].startswith("mmaze") else env.action_space
    wm = models.WorldModel(obs, act_space, 0, cfg)
    count = lambda m: sum(p.numel() for p in m.parameters())
    out = {"wm": count(wm), "core": count(wm.dynamics._cell)}
    if not job["name"].startswith("mmaze"):
        beh = models.ImagBehavior(cfg, wm)
        own = {id(p) for p in wm.parameters()}
        out["actor_critic"] = sum(p.numel() for p in beh.parameters() if id(p) not in own)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", action="store_true")
    args = parser.parse_args()
    jobs = [json.loads(line) for line in (ROOT / "runs" / "jobs.jsonl").open()]
    for bench, prefix in BENCHMARKS.items():
        group = [j for j in jobs if j["name"].startswith(prefix)]
        tasks = sorted({j["name"].split("/")[1] for j in group})
        print(f"\n## {bench}: {len(group)} jobs, tasks {', '.join(tasks)}")
        by_task = {}
        for job in group:
            config, unknown = job_config(job)
            if unknown:
                print(f"  !! {job['name']}: arguments not in the config: {unknown}")
            by_task.setdefault(job["name"].split("/")[1], []).append((job, config))
        for task, items in by_task.items():
            keys = sorted(set().union(*[c.keys() for _, c in items]) - {"logdir", "_script"})
            differ = [k for k in keys if len({json.dumps(c.get(k), sort_keys=True, default=str) for _, c in items}) > 1]
            cells = [j["name"].split("/")[2] for j, _ in items]
            print(f"\n### {task} ({', '.join(cells)})")
            first = items[0][1]
            shared = {k: first.get(k) for k in SHOW if k in first and k not in differ}
            print("  shared: " + "; ".join(f"{k}={v}" for k, v in shared.items()))
            print("  differ:")
            for k in differ:
                vals = {cell: c.get(k) for cell, (_, c) in zip(cells, items)}
                print(f"    {k}: " + "; ".join(f"{cell}={json.dumps(v, default=str)}" for cell, v in vals.items()))
            if args.params and task == next(iter(by_task)):  # sizes do not depend on the task length
                for cell, (job, c) in zip(cells, items):
                    try:
                        print(f"  params {cell}: {params(c, job)}")
                    except Exception as e:  # report, keep going
                        print(f"  params {cell}: failed ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
