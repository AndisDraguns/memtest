import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import numpy as np

class MemTestContinuousEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.offset = 2 # how many back should be guessed. (1 is trivial)
        self.time = 0
        self.max_time = 100 # maximum amount of guesses (excluding warmup)
        self.cell_history = [-1.0]*(self.max_time+(self.offset-1)) # including warmup

        self.neutral_reward = 0.0 # reward during the warmup
        self.min_state = 0.0 # highest number on the dice
        self.max_state = 1.0 # lowest  number on the dice

        self.state = None # state represents the current roll of the dice
        self.n_acts = float("Inf") # how many sides the dice has
        self.action_dim = 1 # how many dice there are 
        self.observation_dim = 2 # for compatibility can tile state for observation

        # for compatibility with algorithms for MountainCarContinuous-v0:
        self.low_state_array  = np.array([self.min_state]*self.observation_dim)
        self.high_state_array = np.array([self.max_state]*self.observation_dim)
        self.action_space = spaces.Box(low=self.min_state, high=self.max_state,
            shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state_array,
            high=high_state_array, dtype=np.float32)
        self.viewer = None

        self.seed()

    def step(self, action):
        time = self.time
        cell_history = self.cell_history
        offset = self.offset
        max_time = self.max_time

        if(time < offset): # if too early for guessing (still in warmup)
            reward = self.neutral_reward
        else reward = abs(cell_history[time - offset] - action)

        if(time < max_time + (offset-1)): # if time has not run out (including warmup)
            self.time += 1
            self.state = self.np_random.uniform(low=self.min_state, high=self.max_state) # roll a dice
            self.cell_history[time] = self.state
            done = False
        else:
            done = True

        return np.full(shape=self.observation_dim, fill_value=self.state), reward, done, {}

    def reset(self):
        self.time = 0
        self.state = self.np_random.uniform(low=self.min_state, high=self.max_state) # roll a dice
        self.cell_history[0] = self.state
        return np.full(shape=self.observation_dim, fill_value=self.state)
 
    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reinit(self, offset=None, max_time=None, min_state=None, max_state=None, observation_dim=None):
        if offset != None:
            self.offset = offset
        if max_time != None:
            self.max_time = max_time
        if min_state != None:
            self.min_state = min_state
        if observation_dim != None:
            self.observation_dim = observation_dim

        self.cell_history = [-1.0]*(self.max_time+(self.offset-1)) # including warmup
        self.low_state_array  = np.array([self.min_state]*self.observation_dim)
        self.high_state_array = np.array([self.max_state]*self.observation_dim)
        self.action_space = spaces.Box(low=self.min_state, high=self.max_state,
            shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state_array,
            high=high_state_array, dtype=np.float32)
    return True