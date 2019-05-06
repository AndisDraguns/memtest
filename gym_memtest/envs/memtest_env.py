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
        self.offset = 2 # how many back should be guessed

        self.state = None
        self.action_space = 2
        self.observation_space = 1

    def step(self, action):
        time = self.time
        cell_history = self.cell_history
        offset = self.offset
        max_time = self.max_time

        if(time < offset):
            reward = 0.0
        elif(cell_history[time - offset] == action): # last action guessed
            reward = 1.0
        else:
            reward = -1.0

        if(time < max_time):
            self.time += 1
            self.state = random.randint(0,1) # 0 or 1
            self.cell_history[time] = self.state
            done = False
        else:
            done = True

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.time = 0
        self.state = random.randint(0,1)
        self.cell_history[0] = self.state
        return np.array(self.state)
 
    def render(self, mode='human', close=False):
        pass