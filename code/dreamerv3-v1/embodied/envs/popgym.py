"""Gymnasium POPGym adapter; no access to privileged benchmark state."""
import functools
import elements
import embodied
import gymnasium as gym
import numpy as np
import popgym  # Register the official environments.

class POPGym(embodied.Env):
  def __init__(self, task, seed=0):
    self._env = gym.make(f"popgym-{task}-v0")
    assert isinstance(self._env.observation_space, gym.spaces.Discrete)
    assert isinstance(self._env.action_space, gym.spaces.Discrete)
    self._seed = seed
    self._done = True
    self._eye = np.eye(self._env.observation_space.n, dtype=np.float32)

  @functools.cached_property
  def obs_space(self):
    return {
        'vector': elements.Space(np.float32, (len(self._eye),), 0, 1),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool)}

  @functools.cached_property
  def act_space(self):
    return {'action': elements.Space(np.int32, (), 0, self._env.action_space.n),
            'reset': elements.Space(bool)}

  def step(self, action):
    first = bool(action['reset']) or self._done
    if first:
      obs, _ = self._env.reset(seed=self._seed)
      self._seed = None
      reward, terminal, truncated = 0.0, False, False
    else:
      obs, reward, terminal, truncated, _ = self._env.step(int(action['action']))
    self._done = bool(terminal or truncated)
    return dict(vector=self._eye[obs].copy(), reward=np.float32(reward),
                is_first=np.bool_(first), is_last=np.bool_(self._done),
                is_terminal=np.bool_(terminal))

  def close(self):
    self._env.close()
