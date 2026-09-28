"""Official BSuite experiment loader with seed-separated, unscaled observations."""
import importlib
import functools
import elements
import embodied
import numpy as np
from bsuite import sweep

class BSuite(embodied.Env):
  def __init__(self, task, seed=0):
    experiment, _ = task.rsplit('/', 1)
    module = importlib.import_module(f'bsuite.experiments.{experiment}.{experiment}')
    settings = dict(sweep.SETTINGS[task])
    settings['seed'] = seed
    self._env = module.load(**settings)
    self._done = True

  @functools.cached_property
  def obs_space(self):
    size = int(np.prod(self._env.observation_spec().shape))
    return dict(vector=elements.Space(np.float32, (size,)),
                reward=elements.Space(np.float32),
                is_first=elements.Space(bool), is_last=elements.Space(bool),
                is_terminal=elements.Space(bool))

  @functools.cached_property
  def act_space(self):
    return dict(action=elements.Space(np.int32, (), 0, self._env.action_spec().num_values),
                reset=elements.Space(bool))

  def step(self, action):
    ts = self._env.reset() if action['reset'] or self._done else self._env.step(int(action['action']))
    self._done = ts.last()
    return dict(vector=np.asarray(ts.observation, np.float32).reshape(-1),
                reward=np.float32(0 if ts.first() else ts.reward),
                is_first=np.bool_(ts.first()), is_last=np.bool_(ts.last()),
                is_terminal=np.bool_(not ts.first() and ts.discount == 0))

  def close(self):
    pass
