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
        self.offset = 2 # how many back should be guessed. (1 is trivial)

        self.neutral_reward  = 0.0
        self.negative_reward = 0.0
        self.positive_reward = 1.0

        self.state = None # saves the current roll of the dice
        self.action_space = 2 # how many sides the dice has
        self.observation_space = 1

    def step(self, action):
        time = self.time
        cell_history = self.cell_history
        offset = self.offset
        max_time = self.max_time

        if(time < offset): # if too early for guessing
            reward = self.neutral_reward
        elif(cell_history[time - offset] == action): # if guessed correctly
            reward = self.positive_reward
        else: # if guessed incorrectly
            reward = self.negative_reward

        if(time < max_time): # if time has not run out
            self.time += 1
            self.state = random.randint(0,self.action_space-1) # roll a dice
            self.cell_history[time] = self.state
            done = False
        else:
            done = True

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.time = 0
        self.state = random.randint(0,self.action_space-1) # roll a dice
        self.cell_history[0] = self.state
        return np.array(self.state)
 
    def render(self, mode='human', close=False):
        pass