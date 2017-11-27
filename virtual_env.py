#Virtual Environment

import gym
from gym import spaces
import numpy as np


class scrEnv(gym.Env):
    def __init__(self):

        self.seq_len = 10
        self.x_dim = 39
        self.y_dim = 3
        self.state = None
        
        self.m_ = lstm.build_model(1, seq_len, x_dim, 100, 1, y_dim)
        self.m_.load_weights("./save_model/env.h5")
        #m_.fit(x_train, y_train, batch_size=1, nb_epoch=100)
        #m_.save_weights("./save_model/env.h5")
        
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
        
        y_pred = lstm.predict_sequence(m_, action_vector, batch_size=1)  # (1,1,3) tensor output
        self.state = y_pred[0,0,:]
        return self.state, reward, done, {}

    def _reset(self):
        
        return np.array(self.state)