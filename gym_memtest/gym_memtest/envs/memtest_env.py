import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import numpy as np

class MemTestEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.offset = 2 # how many back should be guessed. (1 is trivial)
        self.time = 0
        self.max_time = 100 # maximum amount of guesses (excluding warmup)
        self.cell_history = [-1]*(self.max_time+(self.offset-1)) # including warmup

        self.neutral_reward  = 0.0
        self.negative_reward = 0.0
        self.positive_reward = 1.0

        self.act_dim = 2 # how many sides the dice has
        self.obs_dim = 2 # for compatibility can tile state for observation

        # for compatibility with algorithms for generic Gym environments:
        self.state = None # state represents the current roll of the dice
        self.action_space = spaces.Discrete(act_dim)
        self.observation_space = spaces.Discrete(obs_dim)
        self.seed()

    def step(self, action):
        time = self.time
        cell_history = self.cell_history
        offset = self.offset
        max_time = self.max_time

        if(time < offset): # if too early for guessing (still in warmup)
            reward = self.neutral_reward
        elif(cell_history[time - offset] == action): # if guessed correctly
            reward = self.positive_reward
        else: # if guessed incorrectly
            reward = self.negative_reward

        if(time < max_time + (offset-1)): # if time has not run out (including warmup)
            self.time += 1
            self.state = self.np_random.randint(low=0, high=self.act_dim) # roll a dice
            self.cell_history[time] = self.state
            done = False
        else:
            done = True

        return np.full(shape=self.obs_dim, fill_value=self.state), reward, done, {}

    def reset(self):
        self.time = 0
        self.cell_history = [-1]*(self.max_time+(self.offset-1))
        self.state = self.np_random.randint(low=0, high=self.act_dim) # roll a dice
        self.cell_history[0] = self.state
        return np.full(shape=self.obs_dim, fill_value=self.state)
 
    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reinit(self, offset=None, max_time=None):
        if offset != None:
            self.offset = offset
        if max_time != None:
            self.max_time = max_time
        self.cell_history = [-1.0]*(self.max_time+(self.offset-1)) # including warmup