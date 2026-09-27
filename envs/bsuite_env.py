import gym
import numpy as np


class MemoryChain:
    """bsuite memory_len task behind the Dreamer env interface.

    task: "memlen-<L>" (one context bit, recalled after L steps) or
    "memlen-<L>-bits-<B>" (memory_size variant).
    """

    metadata = {}

    def __init__(self, task, seed=0):
        from bsuite.environments import memory_chain

        parts = task.split("-")
        assert parts[0] == "memlen", task
        length = int(parts[1])
        bits = int(parts[3]) if len(parts) > 3 else 1
        self._env = memory_chain.MemoryChain(memory_length=length, num_bits=bits, seed=seed)
        self._dim = bits + 2
        self.reward_range = [-1.0, 1.0]

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
        return gym.spaces.Discrete(2)

    def step(self, action):
        ts = self._env.step(int(action))
        done = ts.last()
        terminal = done and ts.discount == 0
        obs = {
            "vector": ts.observation.reshape(-1).astype(np.float32),
            "is_first": False,
            "is_last": done,
            "is_terminal": terminal,
        }
        info = dict(discount=np.array(0.0 if terminal else 1.0, np.float32))
        return obs, float(ts.reward), done, info

    def reset(self):
        ts = self._env.reset()
        return {
            "vector": ts.observation.reshape(-1).astype(np.float32),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
