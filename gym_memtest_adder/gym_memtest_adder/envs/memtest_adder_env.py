"""
A class for a memory testing binary adder environment called MemTestAdder-v0.

Author: Andis Draguns.
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class MemTestAdderEnv(gym.Env):
    """
    OpenAI Gym type of enviromnent class.

    At each step a coin is flipped and the agent is rewarded if it
    guesses correctly what the total amount of tails so far is.
    To solve this problem close to maximum reward, the agent requires
    some form of memory.

    Changeable variables:
        This environment is customisable by changing some of its parameters.
        max_time (int): controls how many guesses there can be. For example,
        if set to 100, the agent guesses 100 times before the episode ends.
        obs_dim (int) by default it 1 - corresponding to the one boolean value
            of the current coin flip - 0 for heads and 1 for tails. Change
            this to some other positive integer to return an array of the
            current result filling all cells of the array. This might be
            useful for easier compatibility with algorithms working with
            observation space dimensions differring from 1.
        negative_reward (float): is the reward that the agent receives if
            guess is incorrect.
        positive_reward (float): is the reward that the agent receives if
            guess is correct.

    Observation (by default):
        Type: Box(1)
        0 - coin comes up as heads
        1 - coin comes up as tails

    Actions (by default):
        Type: Discrete(101)
        0 - guess the number of tails so far is 0
        1 - guess the number of tails so far is 1
        ...
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """
        Initialises the class with the default variables.

        Customise the environment by changing the changeable variables below.
        To change max_time or obs_dim, use the reinit
        function below as these variables have other variables that should
        change depending on them. The other changeable variables can be freely
        changed.
        """

        # Changeable variables:
        self.max_time = 100
        self.obs_dim = 1
        self.summed = 0
        self.negative_reward = 0.0
        self.positive_reward = 1.0

        # Other variables:
        self.time = 1
        self.act_dim = self.max_time + 1  # +1 because "0" is an action

        # For gym's "spaces.Box" definition of observation_space:
        self.low = np.array([0] * self.obs_dim)
        self.high = np.array([self.act_dim] * self.obs_dim)

        # For compatibility with algorithms for generic Gym environments:
        self.state = 0  # State represents the current flip of the coin
        self.action_space = spaces.Discrete(self.act_dim)
        self.observation_space = spaces.Box(self.low, self.high, dtype=int)
        self.seed()

    def step(self, action):
        """
        Takes an action in the environment.

        This function performs the action from the input in the
        environment. Then it updates the inner state of the environment
        by generating a new random number and adding it to the sum.
        It returns the current coin flip, the calculated reward,
        "done" flag, and an empty dictionary that can be alternatively
        be used for diagnostic information.

        Input:
            action (int): the action that the agent takes in the environment
                at the current step

        Outputs:
            observation (int np.array): current flip of the coin repeated
                obs_dim times
            reward (float): reward obtained during this step
            done (bool): True if episode ended, False otherwise
            info (dict): diagnostic information
        """

        if(self.summed == action):  # If guessed correctly
            reward = self.positive_reward
        else:  # If guessed incorrectly
            reward = self.negative_reward

        if(self.time < self.max_time):
            self.time += 1
            # Flip a coin:
            self.state = self.np_random.randint(low=0, high=2)
            self.summed += self.state
            done = False
        else:
            done = True

        observation = np.full(shape=self.obs_dim, fill_value=self.state)
        info = {}
        return observation, reward, done, info

    def reset(self):
        """
        Resets the environment to the starting state

        Used when starting a new episode.
        Returns the observation from the starting state.

        Output:
            observation (int np.array): current flip of the coin repeated
                obs_dim times
        """

        self.time = 1
        self.summed = 0
        self.state = self.np_random.randint(low=0, high=2)
        self.summed += self.state
        observation = np.full(shape=self.obs_dim, fill_value=self.state)
        return observation

    def render(self, mode="human", close=False):
        """
        Renders a visualisation of the environment.

        Added here as it is a standard Gym environment function,
        but not used as the environment is simple enough to visualise
        by printing the sum at each step.
        """

        pass

    def seed(self, seed=None):
        """
        Initialises the environmment's random number generator (RNG) seed.
        Input:
            seed (int): seed for the RNG
        Output:
            seed_array (int array): array containing the seed
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reinit(self, max_time=None, obs_dim=None):
        """
        Reinitialises the environmment's variables.

        Changes variables that have other variables dependent on them.

        Input:
            max_time (int): the new maximum time limit
            obs_dim (int): the new observation dimension
        """

        if max_time is not None:
            self.max_time = max_time
        if obs_dim is not None:
            self.obs_dim = obs_dim

        self.act_dim = self.max_time + 1
        self.low = np.array([0] * self.obs_dim)
        self.high = np.array([self.act_dim] * self.obs_dim)
        self.action_space = spaces.Discrete(self.act_dim)
        self.observation_space = spaces.Box(self.low, self.high, dtype=int)

    def get_correct(self):
        """
        Returns the correct action at the current step.

        This is useful for expressivity testing of the neural networks. If a
        network can not be trained to a sufficient degree on this problem,
        it might be due to the network used not having enough model
        expressivity. If correct answers are given to it as labels in
        supervised learning, it can reveal problems with expressivity.

        Output:
            correct_action (int): the action that would have earned positive
                reward at the current step
        """
        correct_action = self.summed
        return correct_action
