#Virtual Environment

import gym
from gym import spaces
import lstm
import numpy as np


class scrEnv(gym.Env):
    def __init__(self, num_episode):

        self.seq_len = 10
        self.jump = 5
        self.adj = 1
        
        self.i1m = 4.16589241e+01
        self.i2m = 8.76622772e+00
        
        self.i1s = 3.39985733e+01
        self.i2s = 1.20738525e+01
        
        self.m = []
        self.s = []
        for i in range(10):
            self.m.append([self.i1m,self.i2m])
            self.s.append([self.i1s,self.i2s])
        self.m = np.array(self.m)
        self.s = np.array(self.s)
        
        self.o1m = 2.92767677e+01
        self.o2m = 1.02022934e+01
        self.o3m = 6.67613029e-01
        
        self.o1s = 1.37448368e+01
        self.o2s = 4.67279196e+00
        self.o3s = 1.12606144e+00
        
        self.mu = np.array([self.o1m,self.o2m,self.o3m])
        self.sigma = np.array([self.o1s,self.o2s,self.o3s])
        
        self.x_raw, _, _ = lstm.load_data(
            path="./data.csv", 
            sequence_length = self.seq_len, 
            row_start_ind=1, 
            in_column_ind=[ 2, 3, 4, 5, 6, 7, 8, 9,10,11,
                           12,13,14,15,16,17,19,20,28,30,
                           31,32,33,34,35,36,37,38,39,40,
                           41,42,43,48,49,50,51],
            out_column_ind=[18,22], 
            do_normalize=True)
        
        self.count = 0
        self.sppl_env_stack = self.x_raw[:num_episode,:,:]        
        self.action_state_stack = np.concatenate((90*np.ones((10,1)),50*np.ones((10,1))),axis = 1)
        self.state = np.array([-self.o1m/self.o1s,-self.o2m/self.o2s,-self.o3m/self.o3s])
        
        #Return            Supply
        #4.16589241e+01,   8.76622772e+00
        #3.39985733e+01,   1.20738525e+01
        
        #consumption       stack NOx         NH3 Slip
        #2.92767677e+01,   1.02022934e+01,   6.67613029e-01
        #1.37448368e+01,   4.67279196e+00,   1.12606144e+00
        
        self.m1 = lstm.build_model(1, self.seq_len, 39, 100, 1, 3, False)
        self.m1.load_weights("./save_model/env.h5")
        '''
        self.m2 = lstm.build_model(1, self.seq_len, 37, 100, 1, 37, True)
        self.m2.load_weights("./save_model/supplementary_env.h5")
        '''
        
        self.action_space = spaces.Discrete(9)
        # Ammonia.Return.Line.Pre.Control.V.V
        # Ammonia.Supply.Line.Flow.Control.V.V
        self.observation_space = spaces.Box(np.array([-5.0,-5.0,-5.0]), np.array([150.0,150.0,50.0]))
        # Ammonia.Consumption
        # Flue.Gas.Stack.Nox
        # NH3.Slip
        
    def mover(self, state, action, high_limit, low_limit):
        if action == 3:
            result = state - self.jump
        elif action == 0:
            result = state - self.adj
        elif action == 1:
            result = state
        elif action == 2:
            result = state + self.adj
        elif action == 4:
            result = state + self.jump
        
        if result > high_limit:
            result = high_limit
        elif result < low_limit:
            result = low_limit
            
        return result
        

    def _step(self, action):
        '''
        element = self.sppl_env_stack.reshape(-1,self.seq_len,37)
        sppl_env = lstm.predict_sequence(self.m2, element, batch_size=1)  # (1,10,37) tensor input (1,1,37) tensor output
        self.sppl_env_stack = np.concatenate((element[0,1:,:], sppl_env[0,:,:]), axis=0)
        '''
        a1 = int(action/3) #Ammonia.Return.Line.Pre.Control.V.V
        a2 = action%3  # Ammonia.Supply.Line.Flow.Control.V.V
        self.action_state_stack = np.concatenate((self.action_state_stack[1:,:],
             np.array([[self.mover(self.action_state_stack[9,0], a1, 100.0, -1.0), \
                      self.mover(self.action_state_stack[9,1], a2, 100.0, -1.0)]])), axis=0)
        
        
        self.action_state_stack_norm = (self.action_state_stack - self.m) / self.s
        
        
        action_vector = np.concatenate((self.action_state_stack_norm, self.sppl_env_stack[self.count,:,:]), axis=1).reshape(1,10,39)
        y_pred = lstm.predict_sequence(self.m1, action_vector, batch_size=1)  # (1,10,39) tensor input (1,1,3) tensor output
        self.state = self.sigma*y_pred[0,0,:]+self.mu
        self.count += 1
        
        
        done =  self.state[1] > 20.0
        done = bool(done)
        
        if not done:
            reward = 1.0
        else:
            reward = 0.0
        
        return self.state, reward, done, {}

    def _reset(self):
        self.count = 0
        #self.sppl_env_stack = self.x_raw[1,:,:]        
        self.action_state_stack =  np.concatenate((70*np.ones((10,1)),60*np.ones((10,1))),axis = 1)
        self.state = np.array([-self.o1m/self.o1s,-self.o2m/self.o2s,-self.o3m/self.o3s])
        return self.state