import gym
import numpy as np


class MemoryChain:
    """bsuite memory_len task behind the Dreamer env interface.

    task: "memlen-<L>" (one context bit, recalled after L steps) or
    "memlen-<L>-bits-<B>" (memory_size variant). Trailing flags:
      -cue     append the context bits to every observation, which removes the need for
               memory (control);
      -recall  the final reward is the queried context bit itself (+1/-1), independent of
               the action. The original reward is +1 iff action == context, an XOR of cue
               and action with no first-order signal, which a world model learns slowly
               even when its state holds the cue.
    """

    metadata = {}

    def __init__(self, task, seed=0):
        from bsuite.environments import memory_chain

        parts = task.split("-")
        self._cue = "cue" in parts
        self._recall = "recall" in parts
        parts = [x for x in parts if x not in ("cue", "recall")]
        assert parts[0] == "memlen", task
        length = int(parts[1])
        bits = int(parts[3]) if len(parts) > 3 else 1
        self._env = memory_chain.MemoryChain(memory_length=length, num_bits=bits, seed=seed)
        self._dim = bits + 2 + (bits if self._cue else 0)
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

    def _obs(self, ts):
        vec = ts.observation.reshape(-1).astype(np.float32)
        if self._cue:
            vec = np.concatenate([vec, 2 * self._env._context.astype(np.float32) - 1])
        return vec

    def step(self, action):
        ts = self._env.step(int(action))
        done = ts.last()
        terminal = done and ts.discount == 0
        obs = {
            "vector": self._obs(ts),
            "is_first": False,
            "is_last": done,
            "is_terminal": terminal,
        }
        info = dict(discount=np.array(0.0 if terminal else 1.0, np.float32))
        reward = float(ts.reward)
        if self._recall and done:
            reward = 2.0 * float(self._env._context[self._env._query]) - 1.0
        return obs, reward, done, info

    def reset(self):
        ts = self._env.reset()
        return {
            "vector": self._obs(ts),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
