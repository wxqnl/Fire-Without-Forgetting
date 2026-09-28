"""Offline Memory Maze data for the memory probe (research plan E3, 'Memory Maze offline probe').

Episodes of MemoryMaze-9x9 (1000 steps, 64x64 first-person RGB, 6 discrete actions) collected by
a scripted exploration policy: breadth-first search on the maze layout to a random free cell,
turning towards the next cell of the path and moving forward, with 10% random actions and a new
goal when stuck. This covers the maze the way the scripted exploration of the Memory Maze
offline datasets does; the policy never looks for the targets, rewards are incidental.

Per episode (Dreamer episode format, one npz): image, action (one-hot, zeros at the first
step), reward, is_first, is_terminal, discount, plus the probe targets of every step: agent_pos,
agent_dir, targets_vec (the 3 objects relative to the agent), targets_pos, target_color and
maze_layout (9x9, 1 = free), which the world model does not see.

usage: MUJOCO_GL=osmesa python gen_memory_maze.py --out data/memory_maze_9x9 \
           --train 2000 --eval 200 --workers 24
"""
import argparse
import multiprocessing
import os
import pathlib

import numpy as np

EVAL_SEED_OFFSET = 1_000_000


def heading_error(agent_pos, agent_dir, point):
    """Signed angle (degrees) from the agent's heading to point; positive = turn left."""
    v = point - agent_pos
    err = np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(agent_dir[1], agent_dir[0]))
    return (err + 180.0) % 360.0 - 180.0


class Explorer:
    def __init__(self, rng):
        self.rng = rng
        self.path, self.last_pos, self.stuck = [], None, 0

    def new_goal(self, maze, cell):
        from memory_maze.oracle import breadth_first_search

        free = np.argwhere(maze == 1)  # rows are (y, x)
        for _ in range(20):
            y, x = free[self.rng.randint(len(free))]
            path = breadth_first_search(maze, cell, (int(x), int(y)))
            if path and len(path) > 1:
                self.path = path[1:]
                return
        self.path = []

    def act(self, obs):
        maze, pos, direction = obs["maze_layout"], obs["agent_pos"], obs["agent_dir"]
        cell = (int(pos[0]), int(pos[1]))
        if self.last_pos is not None and np.linalg.norm(pos - self.last_pos) < 1e-3:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_pos = pos.copy()
        while self.path and self.path[0] == cell:
            self.path.pop(0)
        if not self.path or self.stuck > 12:
            self.new_goal(maze, cell)
            self.stuck = 0
        if self.rng.rand() < 0.1 or not self.path:
            return self.rng.randint(6)
        waypoint = np.array(self.path[0], dtype=float) + 0.5
        err = heading_error(pos, direction, waypoint)
        if abs(err) > 35:
            return 2 if err > 0 else 3  # turn in place
        if abs(err) > 10:
            return 4 if err > 0 else 5  # forward while turning
        return 1


PROBE_KEYS = ("agent_pos", "agent_dir", "targets_vec", "targets_pos", "target_color", "maze_layout")


def episode(args):
    index, seed, out = args
    path = out / f"{index:06d}-1001.npz"
    if path.exists():
        return str(path)
    from memory_maze import tasks

    env = tasks.memory_maze_9x9(global_observables=True, seed=seed)
    rng = np.random.RandomState(seed)
    policy = Explorer(rng)
    ts = env.reset()
    ep = {k: [] for k in ("image", "action", "reward", "is_first", "is_terminal", "discount", *PROBE_KEYS)}

    def add(ts, action, first):
        obs = ts.observation
        ep["image"].append(obs["image"])
        ep["action"].append(np.eye(6, dtype=np.float32)[action] if action is not None else np.zeros(6, np.float32))
        ep["reward"].append(np.float32(ts.reward or 0.0))
        ep["is_first"].append(first)
        ep["is_terminal"].append(False)  # time limit only
        ep["discount"].append(np.float32(1.0))
        for k in PROBE_KEYS:
            ep[k].append(np.asarray(obs[k]))

    add(ts, None, True)
    while not ts.last():
        action = policy.act(ts.observation)
        ts = env.step(action)
        add(ts, action, False)
    arrays = {k: np.stack(v) for k, v in ep.items()}
    for k in ("agent_pos", "agent_dir", "targets_vec", "targets_pos", "target_color"):
        arrays[k] = arrays[k].astype(np.float32)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path.with_name(f"{index:06d}-{len(arrays['reward'])}.npz"))
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/memory_maze_9x9")
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--eval", type=int, default=200)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    jobs = []
    for split, n, offset in (("eval", args.eval, EVAL_SEED_OFFSET), ("train", args.train, 0)):
        (out / split).mkdir(parents=True, exist_ok=True)
        done = {p.name.split("-")[0] for p in (out / split).glob("*.npz") if ".tmp" not in p.name}
        jobs += [(i, offset + i, out / split) for i in range(n) if f"{i:06d}" not in done]
    print(f"{len(jobs)} episodes to generate", flush=True)
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(episode, jobs), 1):
            if k % 50 == 0:
                print(f"{k}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
