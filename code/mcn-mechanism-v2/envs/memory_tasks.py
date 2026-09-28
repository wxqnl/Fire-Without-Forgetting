"""Official BSuite/POPGym tasks adapted to the upstream Dreamer interaction API."""
import importlib
import inspect
import gym
import numpy as np


class MemoryTask(gym.Env):
    def __init__(self, suite, task, seed):
        self._suite = suite
        self._seed = seed
        if suite == "popgym":
            import gymnasium
            import popgym
            self._gymnasium = gymnasium
            name = task if task.startswith("popgym-") else "popgym-" + task
            if not name.endswith("-v0"):
                name += "-v0"
            self._env = gymnasium.make(name)
            self._obs_space = self._env.observation_space
            size = gymnasium.spaces.flatdim(self._obs_space)
            if not isinstance(self._env.action_space, gymnasium.spaces.Discrete):
                raise ValueError("Phase-0 tasks require discrete actions")
            actions = self._env.action_space.n
        else:
            from bsuite import sweep
            name, setting = task.rsplit("/", 1)
            module = importlib.import_module("bsuite.experiments." + name + "." + name)
            kwargs = dict(sweep.SETTINGS[task])
            if "seed" in inspect.signature(module.load).parameters:
                kwargs["seed"] = seed
            self._env = module.load(**kwargs)
            size = int(np.prod(self._env.observation_spec().shape))
            actions = self._env.action_spec().num_values
        self.action_space = gym.spaces.Discrete(actions)
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(-np.inf, np.inf, (size,), np.float32),
            "is_first": gym.spaces.Box(0, 1, (), bool),
            "is_last": gym.spaces.Box(0, 1, (), bool),
            "is_terminal": gym.spaces.Box(0, 1, (), bool),
        })

    def _obs(self, obs, first=False, last=False, terminal=False):
        if self._suite == "popgym":
            obs = self._gymnasium.spaces.flatten(self._obs_space, obs)
        return dict(observation=np.asarray(obs, np.float32).reshape(-1),
                    is_first=first, is_last=last, is_terminal=terminal)

    def reset(self):
        if self._suite == "popgym":
            obs, _ = self._env.reset(seed=self._seed)
            self._seed = None
        else:
            obs = self._env.reset().observation
        return self._obs(obs, first=True)

    def step(self, action):
        if self._suite == "popgym":
            obs, reward, terminated, truncated, _ = self._env.step(int(action))
            done = terminated or truncated
            discount = 0.0 if terminated else 1.0
        else:
            timestep = self._env.step(int(action))
            obs, reward = timestep.observation, timestep.reward
            done = timestep.last()
            discount = float(timestep.discount)
            terminated = done and discount == 0.0
        return (self._obs(obs, last=done, terminal=terminated), float(reward),
                done, {"discount": np.float32(discount)})

    def close(self):
        close = getattr(self._env, "close", None)
        if close:
            close()
