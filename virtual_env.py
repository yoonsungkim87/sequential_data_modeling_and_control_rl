#Virtual Environment

import gym
from gym import spaces
import lstm
import numpy as np


class scrEnv(gym.Env):
    def __init__(self):

        self.seq_len = 10
        self.x_dim = 39
        self.y_dim = 3
        self.state = None
        
        self.m_ = lstm.build_model(1, self.seq_len, self.x_dim, 100, 1, self.y_dim)
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
        
        action_vector = np.random.randn(1,10,39)
        y_pred = lstm.predict_sequence(self.m_, action_vector, batch_size=1)  # (1,1,3) tensor output
        self.state = y_pred[0,0,:]
        reward = None
        done = None
        return self.state, reward, done, {}

    def _reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(3,))
        return self.state