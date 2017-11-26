#Virtual Environment

import gym
from gym import spaces
import numpy as np


class scrEnv(gym.Env):
    def __init__(self):
        
        self.action_space = spaces.Tuple((
            spaces.Box(-1.0, 100.0, 1),  # Ammonia.Return.Line.Pre.Control.V.V
            spaces.Box(-1.0, 100.0, 1),  # Ammonia.Supply.Line.Flow.Control.V.V
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(-5.0, 150.0, 1),  # Ammonia.Consumption
            spaces.Box(-5.0, 150.0, 1),  # Flue.Gas.Stack.Nox
            spaces.Box(-5.0, 50.0, 1),   # NH3.Slip
        ))

    def _step(self, action):

        return np.array(self.state), reward, done, {}

    def _reset(self):
        
        return np.array(self.state)