"""Times world-model + actor-critic updates for a config, without env interaction.

usage: python scripts/bench_update.py --configs popgym persist --task popgym_RepeatPreviousEasy
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.argv[0] = str(pathlib.Path(__file__).resolve().parents[1] / "dreamer.py")

import argparse

import numpy as np
import ruamel.yaml as yaml
import torch

import dreamer
import tools


def build_config(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--updates", type=int, default=30)
    args, remaining = parser.parse_known_args(argv)
    configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text())

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
    return parser.parse_args(remaining), args.updates


def main():
    config, updates = build_config(sys.argv[1:])
    env = dreamer.make_env(config, "train")
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    episodes = {}
    for i in range(8):
        obs = env.reset()
        ep = {k: [v] for k, v in obs.items()}
        ep["action"] = [np.zeros(config.num_actions, np.float32)]
        ep["reward"] = [0.0]
        ep["discount"] = [1.0]
        done = False
        while not done:
            a = np.eye(config.num_actions, dtype=np.float32)[np.random.randint(config.num_actions)]
            obs, r, done, info = env.step({"action": a})
            for k, v in obs.items():
                ep[k].append(v)
            ep["action"].append(a)
            ep["reward"].append(r)
            ep["discount"].append(float(info["discount"]))
        episodes[str(i)] = {k: tools.convert(v) for k, v in ep.items()}
    logger = tools.Logger(pathlib.Path("/tmp") / f"bench_{time.time()}", 0)
    dataset = dreamer.make_dataset(episodes, config)
    agent = dreamer.Dreamer(env.observation_space, env.action_space, config, logger, dataset).to(config.device)
    agent.requires_grad_(False)
    for _ in range(3):
        agent._train(next(dataset))
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(updates):
        agent._train(next(dataset))
    torch.cuda.synchronize()
    dt = (time.time() - t0) / updates
    batch_steps = config.batch_size * config.batch_length
    per_env_step = config.train_ratio / batch_steps
    print(
        f"sec/update {dt:.3f}  updates/env-step {per_env_step:.2f}  "
        f"-> hours per 1e5 env steps {dt * per_env_step * 1e5 / 3600:.2f}  "
        f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
    )


if __name__ == "__main__":
    main()
