#Virtual Environment

import gym
from gym import spaces
import numpy as np


class scrEnv(gym.Env):
    def __init__(self):
        seq_len = 10
        train_samples = 10000
        test_samples = 200

        x_raw, y_raw, info = lstm.load_data(
            path="./data.csv", 
            sequence_length = seq_len, 
            row_start_ind=1, 
            in_column_ind=[18,22,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,28,30,31,32,33,34,35,36,37,38,39,40,41,42,43,48,49,50,51],
            out_column_ind=[21,55,58], 
            do_normalize=True)

        #print(x_raw.shape, y_raw.shape)

        x_dim = x_raw.shape[2]
        y_dim = y_raw.shape[2]
        x_train, y_train = x_raw[:train_samples,:,:], y_raw[:train_samples,:,:]
        x_test, y_test = x_raw[-test_samples:,:,:], y_raw[-test_samples:,:,:]

        self.m_ = lstm.build_model(1, seq_len, x_dim, 100, 1, y_dim)
        self.m_.load_weights("./save_model/env.h5")
        self.m_.fit(x_train, y_train, batch_size=1, nb_epoch=1)
        self.m_.save_weights("./save_model/env.h5")
        self.y_pred = lstm.predict_sequence(m_, x_test, batch_size=1)

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