"""
A class for continuous memory testing environment called MemTestContinuous-v0.

Author: Andis Draguns.
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class MemTestContinuousEnv(gym.Env):
    """
    OpenAI Gym type of enviromnent class.

    At each step a random floating point number is generated.
    The agent is rewarded in proportion to how close it guesses
    what the result of the generation was two steps ago. To do
    better than guessing at random, the agent requires some form
    of memory.

    Customisability:
        By default the environment fits the simple description above,
        but it can also be customised by changing some of its parameters.
        For example, the number of steps the agent should look back for
        the correct guess can be changed.

    Definitions:
        Generation - the product of the random number generator that
            is generated at each step.
        Warmup phase - the first few steps where the agent has not yet
            taken enough steps for there to be a generation enough steps
            in the past for the agent to guess. For example, the agent
            could not guess what was the result of the generation 5 steps
            ago while being at the first step.
        Game phase - the steps that are not in the warmup phase.

    Changeable variables:
        This environment is customisable by changing some of its parameters.
        offset (int): controls how many steps ago was the generation that the
            agent currently has to guess. If set to 1, the agent just has to
            output the action corresponding to the observation - this requires
            no memory.
        max_time (int): controls how many guesses there can be, excluding the
            warmup phase.
        obs_dim (int) by default it 1 - corresponding to the one float value
            of the current generation. Change this to some other positive
            integer to return an array of the current generation filling all
            cells of the array. This might be useful for easier compatibility
            with algorithms working with observation space dimensions
            differring from 1.
        neutral_reward (float): is the reward that the agent receives at each
            step while in the warmup phase.
        min_state (float): the lowest value that can be generated
        max_state (float): the highest value that can be generated

    Observation (by default):
        Type: Box(1)
        min: 0.0
        max: 1.0

    Actions (by default):
        Type: Box(1)
        min: 0.0
        max: 1.0
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """
        Initialises the class with the default variables.

        Customise the environment by changing the changeable variables below.
        To change offset, max_time, obs_dim, min_state or max_state, use the
        reinit function below as these variables have other variables that
        should change depending on them. The neutral reward can be freely
        changed.
        """

        # Changeable variables:
        self.offset = 2
        self.max_time = 100
        self.obs_dim = 2
        self.min_state = 0.0
        self.max_state = 1.0
        self.neutral_reward = 0.0

        # Other variables:
        self.time = 0
        # Cell history keeps the record of all the generations so far
        # The sum is maximum time + steps in the warmup phase
        self.cell_history = [-1.0] * (self.max_time + (self.offset - 1))

        # For compatibility with algorithms for MountainCarContinuous-v0:
        self.low_state_array = np.array([self.min_state] * self.obs_dim)
        self.high_state_array = np.array([self.max_state] * self.obs_dim)
        self.viewer = None

        # For compatibility with algorithms for generic Gym environments:
        self.state = 0.0  # State represents the current generation
        self.action_space = spaces.Box(
            low=self.min_state, high=self.max_state,
            shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=self.low_state_array,
            high=self.high_state_array, dtype=np.float32)
        self.seed()

    def step(self, action):
        """
        Takes an action in the environment.

        This function performs the action from the input in the
        environment. Then it updates the inner state of the environment
        by generating a new random number and saving it in the cell
        history. It returns the generation, the calculated reward,
        "done" flag, and an empty dictionary that can be alternatively
        be used for diagnostic information.

        Input:
            action (float): the action that the agent takes in the environment
                at the current step

        Outputs:
            observation (float np.array): current roll of the dice repeated
                obs_dim times
            reward (float): reward obtained during this step
            done (bool): True if episode ended, False otherwise
            info (dict): diagnostic information
        """

        time = self.time
        offset = self.offset

        if(time < offset):  # If too early for guessing (in warmup phase)
            reward = self.neutral_reward
        else:
            # Rewards are higher if the guess is closer:
            reward = 1.0 - abs(self.cell_history[time - offset] - action * 1.0)

        # If time has not run out yet (including the warmup phase):
        if(time < self.max_time + (offset - 1)):
            self.time += 1
            # Create a new generation:
            self.state = self.np_random.uniform(
                low=self.min_state, high=self.max_state)
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
            observation (float np.array): current generation repeated
                obs_dim times
        """

        self.time = 0
        self.cell_history = [-1.0] * (self.max_time + (self.offset - 1))
        self.state = self.np_random.uniform(
            low=self.min_state, high=self.max_state)
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

    def reinit(
            self, offset=None, max_time=None,
            obs_dim=None, min_state=None, max_state=None):
        """
        Reinitialises the environmment's variables.

        Changes variables that have other variables dependent on them.

        Input:
            offset (int): the new offset
            max_time (int): the new maximum time limit
            obs_dim (int): the new observation dimension
            min_state (float): the new minimum generation
            max_state (float): the new maximum generation
        """

        if offset is not None:
            self.offset = offset
        if max_time is not None:
            self.max_time = max_time
        if obs_dim is not None:
            self.obs_dim = obs_dim
        if min_state is not None:
            self.min_state = min_state
        if max_state is not None:
            self.max_state = max_state

        self.cell_history = [-1.0] * (self.max_time + (self.offset - 1))
        self.low_state_array = np.array([self.min_state] * self.obs_dim)
        self.high_state_array = np.array([self.max_state] * self.obs_dim)
        self.action_space = spaces.Box(
            low=self.min_state, high=self.max_state,
            shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=self.low_state_array,
            high=self.high_state_array, dtype=np.float32)

    def get_correct(self):
        """
        Returns the correct action if in game phase, otherwise a random action.

        This is useful for expressivity testing of the neural networks. If a
        network can not be trained to a sufficient degree on this problem,
        it might be due to the network used not having enough model
        expressivity. If correct answers are given to it as labels in
        supervised learning, it can reveal problems with expressivity.
        Random actions are given in the warmup phase to mask the gradients
        in the average for the warmup phase in which all actions earn the
        same neutral reward.

        Output:
            correct_action (float): the action that would have earned the
                highest reward at the current step
        """
        time = self.time
        offset = self.offset

        if(time < offset):  # If too early for guessing (in warmup phase)
            correct_action = self.state = self.np_random.uniform(
                low=self.min_state, high=self.max_state)
        else:
            correct_action = self.cell_history[time - offset]

        return correct_action
