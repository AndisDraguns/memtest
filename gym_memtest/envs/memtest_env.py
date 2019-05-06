import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import numpy as np

class MemTestEnv(gym.Env):  
    metadata = {'render.modes': ['human']}   
    def __init__(self):
        self.time = 0
        self.max_time = 100
        self.cell_history = [-1]*self.max_time

        self.state = None
        self.action_space = 2
        self.observation_space = 1
 
    def step(self, action):
        time = self.time
        cell_history = self.cell_history

        if(cell_history[time - 1] == action): # last action guessed
            reward = 1.0
        else:
            reward = -1.0

        if(time < self.max_time):
            self.state = random.randint(0,1) # 0 or 1
            cell_history[time] = self.state
            self.time += 1
            done = False
        else:
            done = True

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = random.randint(0,1)
        self.time = 0
        return np.array(self.state)
 
    def render(self, mode='human', close=False):
        pass