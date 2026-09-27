import gym
import numpy as np


def _flat_dim(space):
    import gymnasium

    if isinstance(space, gymnasium.spaces.Discrete):
        return int(space.n)
    if isinstance(space, gymnasium.spaces.MultiDiscrete):
        return int(np.sum(space.nvec))
    if isinstance(space, gymnasium.spaces.Tuple):
        return sum(_flat_dim(s) for s in space.spaces)
    if isinstance(space, gymnasium.spaces.Box):
        return int(np.prod(space.shape))
    raise NotImplementedError(space)


def _flatten(space, obs):
    """One-hot encodes discrete parts, flattens continuous parts."""
    import gymnasium

    if isinstance(space, gymnasium.spaces.Discrete):
        out = np.zeros(space.n, np.float32)
        out[int(obs) - space.start] = 1.0
        return out
    if isinstance(space, gymnasium.spaces.MultiDiscrete):
        parts = []
        for n, o in zip(space.nvec.flatten(), np.asarray(obs).flatten()):
            part = np.zeros(n, np.float32)
            part[int(o)] = 1.0
            parts.append(part)
        return np.concatenate(parts)
    if isinstance(space, gymnasium.spaces.Tuple):
        return np.concatenate([_flatten(s, o) for s, o in zip(space.spaces, obs)])
    if isinstance(space, gymnasium.spaces.Box):
        return np.asarray(obs, np.float32).flatten()
    raise NotImplementedError(space)


class PopGym:
    """POPGym task (e.g. RepeatPreviousEasy) behind the Dreamer env interface."""

    metadata = {}

    def __init__(self, name, seed=0):
        import popgym.envs

        classes = {cls.__name__: cls for cls in popgym.envs.ALL}
        self._env = classes[name]()
        self._seed = seed
        self._first_reset = True
        self._dim = _flat_dim(self._env.observation_space)
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "vector": gym.spaces.Box(-np.inf, np.inf, (self._dim,), dtype=np.float32),
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )

    @property
    def action_space(self):
        space = self._env.action_space
        assert hasattr(space, "n"), space
        return gym.spaces.Discrete(int(space.n))

    def _obs(self, obs, is_first=False, is_last=False, is_terminal=False):
        return {
            "vector": _flatten(self._env.observation_space, obs),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        done = terminated or truncated
        info = dict(discount=np.array(0.0 if terminated else 1.0, np.float32))
        return self._obs(obs, False, done, terminated), float(reward), done, info

    def reset(self):
        if self._first_reset:
            obs, _ = self._env.reset(seed=self._seed)
            self._first_reset = False
        else:
            obs, _ = self._env.reset()
        return self._obs(obs, is_first=True)
