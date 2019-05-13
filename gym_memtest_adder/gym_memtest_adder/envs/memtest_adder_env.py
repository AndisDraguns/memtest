import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import numpy as np

class MemTestAdderEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.time = 0
        self.max_time = 100 # maximum amount of guesses (excluding warmup)
        self.cell_history = [0]*self.max_time # including warmup

        self.negative_reward = 0.0
        self.positive_reward = 1.0

        self.state = None # state represents the current roll of the dice
        self.n_acts = self.max_time # how many sides the dice has
        self.action_dim = 1 # how many dice there are
        self.observation_dim = 2 # for compatibility can tile state for observation
        self.seed()

    def step(self, action):
        time = self.time
        cell_history = self.cell_history
        offset = self.offset
        max_time = self.max_time

        if(sum(cell_history) == action): # if guessed correctly
            reward = self.positive_reward
        else: # if guessed incorrectly
            reward = self.negative_reward

        if(time < max_time): # if time has not run out (including warmup)
            self.time += 1
            self.state = self.np_random.randint(low=0, high=2) # roll a dice
            self.cell_history[time] = self.state
            done = False
        else:
            done = True

        return np.full(shape=self.observation_dim, fill_value=self.state), reward, done, {}

    def reset(self):
        self.time = 0
        self.state = self.np_random.randint(low=0, high=2) # roll a dice
        self.cell_history[0] = self.state
        return np.full(shape=self.observation_dim, fill_value=self.state)
 
    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reinit(self, max_time=None):
        if max_time != None:
            self.max_time = max_time
        self.cell_history = [0]*self.max_time # including warmup
        self.n_acts = self.max_time