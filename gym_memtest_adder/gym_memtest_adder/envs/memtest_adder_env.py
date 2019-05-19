import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import numpy as np

class MemTestAdderEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.time = 1
        self.max_time = 100 # maximum amount of guesses
        self.summed = 0

        self.negative_reward = 0.0
        self.positive_reward = 1.0

        self.act_dim = self.max_time+1 # +1 because "0" is an action
        self.obs_dim = 2 # for compatibility can tile state for observation

        # for compatibility with algorithms for generic Gym environments:
        self.state = 0 # state represents the current roll of the dice
        self.action_space = spaces.Discrete(act_dim)
        self.observation_space = spaces.Discrete(obs_dim)
        self.seed()

    def step(self, action):
        if(self.summed == action): # if guessed correctly
            reward = self.positive_reward
        else: # if guessed incorrectly
            reward = self.negative_reward

        if(self.time < self.max_time): # if time has not run out
            self.time += 1
            self.state = self.np_random.randint(low=0, high=2) # roll a dice
            self.summed += self.state
            done = False
        else:
            done = True

        return np.full(shape=self.obs_dim, fill_value=self.state), reward, done, {}

    def reset(self):
        self.time = 1
        self.summed = 0
        self.state = self.np_random.randint(low=0, high=2) # roll a dice
        self.summed += self.state
        return np.full(shape=self.obs_dim, fill_value=self.state)
 
    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reinit(self, max_time=None):
        if max_time != None:
            self.max_time = max_time
        self.act_dim = self.max_time+1