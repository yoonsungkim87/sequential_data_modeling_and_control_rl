#Virtual Environment

import gym
from gym import spaces
import lstm
import numpy as np


class scrEnv(gym.Env):
    def __init__(self):

        self.seq_len = 10
        self.state = None
        self.jump = 5
        self.adj = 1
        
        self.i1m = 4.16589241e+01
        self.i2m = 8.76622772e+00
        
        self.i1s = 3.39985733e+01
        self.i2s = 1.20738525e+01
        
        self.o1m = 2.92767677e+01
        self.o2m = 1.02022934e+01
        self.o3m = 6.67613029e-01
        
        self.o1s = 1.37448368e+01
        self.o2s = 4.67279196e+00
        self.o3s = 1.12606144e+00
        
        #Return            Supply
        #4.16589241e+01,   8.76622772e+00
        #3.39985733e+01,   1.20738525e+01
        
        #consumption       stack NOx         NH3 Slip
        #2.92767677e+01,   1.02022934e+01,   6.67613029e-01
        #1.37448368e+01,   4.67279196e+00,   1.12606144e+00
        
        self.m1 = lstm.build_model(1, self.seq_len, 39, 100, 1, 3, False)
        self.m1.load_weights("./save_model/env.h5")
        
        self.m2 = lstm.build_model(1, self.seq_len, 37, 100, 1, 37, True)
        self.m2.load_weights("./save_model/supplementary_env.h5")
        
        self.action_space = spaces.Discrete(25)
        # Ammonia.Return.Line.Pre.Control.V.V
        # Ammonia.Supply.Line.Flow.Control.V.V
        self.observation_space = spaces.Box(np.array([-5.0,-5.0,-5.0]), np.array([150.0,150.0,50.0]))
        # Ammonia.Consumption
        # Flue.Gas.Stack.Nox
        # NH3.Slip
        
    def mover(state, action, high_limit, low_limit):
        if action == 0:
            result = state - self.jump
        elif action == 1:
            result = state - self.adj
        elif action == 2:
            result = state
        elif action == 3:
            result = state + self.adj
        elif action == 4:
            result = state + self.jump
        
        if result > high_limit:
            result = high_limit
        elif result < low_limit:
            result = low_limit
            
        return result
        

    def _step(self, action):
        
        element = self.sppl_env_stack.reshape(-1,self.seq_len,37)
        sppl_env = lstm.predict_sequence(self.m2, element, batch_size=1)  # (1,10,37) tensor input (1,1,37) tensor output
        self.sppl_env_stack = np.concatenate((element[0,1:,:], sppl_env[0,:,:]), axis=0)
        
        a1 = int(action/5) #Ammonia.Return.Line.Pre.Control.V.V
        a2 = action%5  # Ammonia.Supply.Line.Flow.Control.V.V
        t1 = float(self.action_state_stack[9,0])
        t2 = float(self.action_state_stack[9,1])
        self.action_state_stack = np.concatenate( \
            (self.action_state_stack[1:,:], \
             np.concatenate( \
                 (self.mover(state = t1, action = a1, high_limit = 100.0, low_limit = -1.0), \
                  self.mover(state = t2, action = a2, high_limit = 100.0, low_limit = -1.0) \
                 ),axis=1)), axis=0)
        
        
        action_vector = np.concatenate((self.action_state_stack, self.sppl_env_stack), axis=1).reshape(1,10,39)
        y_pred = lstm.predict_sequence(self.m1, action_vector, batch_size=1)  # (1,10,39) tensor input (1,1,3) tensor output
        self.state = y_pred[0,0,:]
        
        if not done:
            if self.state[1] < 7.0:
                reward = 1.0
            else:
                reward = 0.5
        else:
            reward = 0.0
        
        done =  self.state[1] > 15.0
        done = bool(done)
        
        return self.state, reward, done, {}

    def _reset(self):
        
        x_raw, _, _ = lstm.load_data(
            path="./data.csv", 
            sequence_length = self.seq_len, 
            row_start_ind=1, 
            in_column_ind=[ 2, 3, 4, 5, 6, 7, 8, 9,10,11,
                           12,13,14,15,16,17,19,20,28,30,
                           31,32,33,34,35,36,37,38,39,40,
                           41,42,43,48,49,50,51],
            out_column_ind=[18,22], 
            do_normalize=True)
        self.sppl_env_stack = x_raw[1,:,:]
        
        self.action_state_stack = np.zeros((10,2))
        
        self.state = np.random.uniform(low=-0.01, high=0.01, size=(3,))
        return self.state