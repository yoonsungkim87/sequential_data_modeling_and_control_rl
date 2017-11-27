#Virtual Environment

import gym
from gym import spaces
import lstm
import numpy as np


class scrEnv(gym.Env):
    def __init__(self):

        self.seq_len = 10
        self.state = None
        x_raw, _, _ = lstm.load_data(
            path="./data.csv", 
            sequence_length = self.seq_len, 
            row_start_ind=1, 
            in_column_ind=[ 2, 3, 4, 5, 6, 7, 8, 9,10,11,
                           12,13,14,15,16,17,19,20,28,30,
                           31,32,33,34,35,36,37,38,39,40,
                           41,42,43,48,49,50,51],
            out_column_ind=[ 2], 
            do_normalize=True)
        self.temp = []
        self.temp.append(x_test[1,:,:])
        
        self.m1 = lstm.build_model(1, self.seq_len, 39, 100, 1, 3)
        self.m1.load_weights("./save_model/env.h5")
        
        self.m2 = lstm.build_model(1, self.seq_len, 37, 100, 1, 37)
        self.m2.load_weights("./save_model/supplementary_env.h5")
        
        self.action_space = spaces.Box(np.array([-1.0,-1.0]), np.array([100.0,100.0]))
        # Ammonia.Return.Line.Pre.Control.V.V
        # Ammonia.Supply.Line.Flow.Control.V.V
        self.observation_space = spaces.Box(np.array([-5.0,-5.0,-5.0]), np.array([150.0,150.0,50.0]))
        # Ammonia.Consumption
        # Flue.Gas.Stack.Nox
        # NH3.Slip
        
    def sppl_env_count_up(self, model):
        
        temp = []
        result = []
        temp.append(x_test[1,:,:])
        for i in range(x_test.shape[0]):
            element = temp[-1].reshape(-1,seq_len,37)
            y_pred = lstm.predict_sequence(m_, element, batch_size=1)
            temp.append(np.concatenate((element[0,1:,:], y_pred[0,:,:]), axis=0))
            result.append(y_pred[0,:,:])
        result = np.array(result)

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