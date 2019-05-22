"""
A class for a simple memory testing environment called MemTest-v0.

Author: Andis Draguns.
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class MemTestEnv(gym.Env):
    """
    OpenAI Gym type of enviromnent class.

    At each step a coin is flipped and the agent is rewarded if it
    guesses correctly what the result of the coin flip was two steps
    ago. To solve this problem better than guessing at random, the
    agent requires some form of memory.

    Customisability:
        By default the environment fits the simple description above,
        but it can also be customised by changing some of its parameters.
        The coin can be changed into a dice, allowing more than two
        possible outcomes at each step. The number of steps the agent
        should look back for the correct guess can be changed as well.

    Definitions:
        Dice roll - the outcome of a coin flip where number of actions
            in the environment is changed to some integer amount
            corresponding to the sides of a dice.
        Warmup phase - the first few steps where the agent has not yet
            taken enough steps for there to be a dice roll enough steps
            in the past for the agent to guess. For example, the agent
            could not guess what was the result of the dice roll 5 steps
            ago while being at the first step.
        Game phase - the steps that are not in the warmup phase.

    Changeable variables:
        This environment is customisable by changing some of its parameters.
        offset (int): controls how many steps ago was the coin flip that the
            agent currently has to guess.  If set to 1, the agent just has to
            output the action corresponding to the observation - this requires
            no memory.
        max_time (int): controls how many guesses there can be, excluding the
            warmup phase. For example, if set to 100, the sum of the rewards
            in an episode corresponds to the percent of correct guesses in
            the game phase.
        act_dim (int): controls how many outcomes of the dice roll can there
            be - it can change the coin into a dice. This also corresponds to
            the number of actions that the agent can take - its guess can be
            any side of the dice.
        obs_dim (int): by default it is 1 - corresponding to the one boolean
            value of the current coin flip - 0 for heads and 1 for tails.
            Change this to some other positive integer to return an array of
            the current result filling all cells of the array. This might be
            useful for easier compatibility with algorithms working with
            observation space dimensions differring from 1.
        neutral_reward (float): is the reward that the agent receives at each
            step while in the warmup phase.
        negative_reward (float): is the reward that the agent receives if
            guess is incorrect and the game phase has already started.
        positive_reward (float): is the reward that the agent receives if
            guess is correct and the game phase has already started.

    Observation (by default):
        Type: Box(1)
        0 - coin comes up as heads
        1 - coin comes up as tails

    Actions (by default):
        Type: Discrete(2)
        0 - guess that coin came up heads two turns ago
        1 - guess that coin came up tails two turns ago
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """
        Initialises the class with the default variables.

        Customise the environment by changing the changeable variables below.
        To change offset, max_time, act_dim or obs_dim, use the reinit
        function below as these variables have other variables that should
        change depending on them. The three types of reward can be freely
        changed.
        """

        # Changeable variables:
        self.offset = 2
        self.max_time = 100
        self.act_dim = 2
        self.obs_dim = 1
        self.neutral_reward = 0.0
        self.negative_reward = 0.0
        self.positive_reward = 1.0

        # Variables dependent on the changeable variables:
        self.time = 0
        # Cell history keeps the record of all the coin flips so far
        # The sum is maximum time + steps in the warmup phase
        self.cell_history = [-1] * (self.max_time + (self.offset - 1))

        # For gym's "spaces.Box" definition of observation_space
        # Smallest and largest valued observation arrays
        self.low = np.array([0] * self.obs_dim)
        self.high = np.array([self.act_dim] * self.obs_dim)

        # For compatibility with algorithms for generic Gym environments:
        self.state = None  # State represents the current roll of the dice
        self.action_space = spaces.Discrete(self.act_dim)
        self.observation_space = spaces.Box(self.low, self.high, dtype=int)
        self.seed()

    def step(self, action):
        """
        Takes an action in the environment.

        This function performs the action from the input in the
        environment. Then it updates the inner state of the environment
        by rolling a dice and saving it in the cell history. It returns
        the the dice roll, the calculated reward, "done" flag, and an
        empty dictionary that can be alternatively be used for diagnostic
        information.

        Input:
            action (int): the action that the agent takes in the environment
                at the current step

        Outputs:
            observation (int np.array): current roll of the dice repeated
                obs_dim times
            reward (float): reward obtained during this step
            done (bool): True if episode ended, False otherwise
            info (dict): diagnostic information
        """

        time = self.time
        offset = self.offset

        if(time < offset):  # If too early for guessing (in warmup phase)
            reward = self.neutral_reward
        elif(self.cell_history[time - offset] == action):  # if correct
            reward = self.positive_reward
        else:  # If incorrect
            reward = self.negative_reward

        # If time has not run out yet (including the warmup phase):
        if(time < self.max_time + (offset - 1)):
            self.time += 1
            # Roll a dice:
            self.state = self.np_random.randint(low=0, high=self.act_dim)
            self.cell_history[time] = self.state
            done = False
        else:
            done = True

        info = {}
        observation = np.full(shape=self.obs_dim, fill_value=self.state)
        return observation, reward, done, info

    def reset(self):
        """
        Resets the environment to the starting state

        Used when starting a new episode.
        Returns the observation from the starting state.

        Output:
            observation (int np.array): current roll of the dice repeated
                obs_dim times
        """

        self.time = 0
        self.cell_history = [-1] * (self.max_time + (self.offset - 1))
        self.state = self.np_random.randint(low=0, high=self.act_dim)
        self.cell_history[0] = self.state
        observation = np.full(shape=self.obs_dim, fill_value=self.state)
        return observation

    def render(self, mode="human", close=False):
        """
        Renders a visualisation of the environment.

        Added here as it is a standard Gym environment function,
        but not used as the environment is simple enough to visualise
        by printing the cell history.
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

    def reinit(self, offset=None, max_time=None, act_dim=None, obs_dim=None):
        """
        Reinitialises the environmment's variables.

        Changes variables that have other variables dependent on them.

        Input:
            offset (int): the new offset
            max_time (int): the new maximum time limit
            act_dim (int): the new number of actions
            obs_dim (int): the new observation dimension
        """

        if offset is not None:
            self.offset = offset
        if max_time is not None:
            self.max_time = max_time
        if act_dim is not None:
            self.act_dim = act_dim
        if obs_dim is not None:
            self.obs_dim = obs_dim

        self.cell_history = [-1.0] * (self.max_time + (self.offset - 1))
        self.action_space = spaces.Discrete(self.act_dim)
        self.observation_space = spaces.Discrete(self.obs_dim)


def get_correct(self):
    """
    Returns the correct action if in game phase, otherwise a random action.

    This is useful for expressivity testing of the neural networks. If a
    network can not be trained to a sufficient degree on this problem,
    it might be due to the network used not having enough model expressivity.
    If correct answers are given to it as labels in supervised learning,
    it can reveal problems with expressivity. Random actions are given in
    the warmup phase to mask the gradients in the average for the warmup
    phase in which all actions earn the same neutral reward.

    Output:
        correct_action (int): the action that would have earned positive
            reward at the current step
    """
    time = self.time
    offset = self.offset

    if(time < offset):  # If too early for guessing (in warmup phase)
        correct_action = self.np_random.randint(low=0, high=self.act_dim)
    else:
        correct_action = self.cell_history[time - offset]

    return correct_action
