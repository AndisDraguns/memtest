"""
A class for a simple memory testing environment called Mastermind-v0.

Author: Andis Draguns.
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class MastermindEnv(gym.Env):
    """
    OpenAI Gym type of enviromnent class.

    This is an implementation of the board game "Mastermind":
    https://en.wikipedia.org/wiki/Mastermind_(board_game)

    At each step the agent tries to guess the pattern of 4 hidden peg colours
    and order by putting up 4 code pegs. The environment outputs observation
    with white key peg for each correct colour but wrong place code peg and a
    black key peg for each correct colout and right place code peg.

    Action: a guess for the hidden code pegs
    Observation: counters for white and black pegs

    Customisability:
        By default the environment fits the simple description above,
        but it can also be customised by changing some of its parameters.
        By defauly there are 6 code peg colours and 12 guesses. Also
        the reward by default is given only when the game is won.
        All of this can be changed.

    Changeable variables:
        This environment is customisable by changing some of its parameters.

        max_guesses (int): controls how many guesses there can be.
        n_colours (int): controls how many colours of code pegs there are.
        n_pegs (int): controls how many code peg slots there are.
        white_reward (float): is the reward that the agent receives for each
            white peg observed.
        black_reward (float): is the reward that the agent receives for each
            black peg observed.
        win_reward (float): is the reward that the agent receives if
            the guess is correct.

    Observation (by default):
        Type: Box(2)

    Actions (by default):
        Type: Discrete(1296)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """
        Initialises the class with the default variables.

        Customise the environment by changing the changeable variables below.
        To change n_colours or n_pegs use the reinit function below as these
        variables have other variables that should change depending on them.
        """

        # Changeable variables:
        self.max_guesses = 12
        self.n_colours = 6  # Change via reinit() method
        self.n_pegs = 4  # Change via reinit() method
        self.white_reward = 1.0
        self.black_reward = 2.0
        self.win_reward = 10.0

        self.act_dim = self.n_colours * self.n_pegs
        self.obs_dim = 2  # The two counters for key pegs
        self.time = 0

        self.pattern = np.array([-1] * self.n_pegs)  # The pattern array

        # For gym's "spaces.Box" definition of observation_space
        # Smallest and largest valued observation arrays
        self.low = np.array([0] * self.obs_dim)
        self.high = np.array([self.n_pegs] * self.obs_dim)

        # For compatibility with algorithms for generic Gym environments:
        self.action_space = spaces.Discrete(self.act_dim)
        self.observation_space = spaces.Box(self.low, self.high, dtype=int)
        self.seed()

    def step(self, action):
        """
        Takes an action in the environment.

        This function performs the action from the input in the
        environment. It returns key peg counters, the calculated reward,
        "done" flag, and an empty dictionary that can be alternatively be
        used for diagnostic information.

        Input:
            action (int): the action that the agent takes in the environment
                at the current step

        Outputs:
            observation (int np.array): array consisting of white key peg
                counter and black key peg counter.
            reward (float): reward obtained during this step
            done (bool): True if episode ended, False otherwise
            info (dict): diagnostic information
        """

        #  Convert the action int to an array of peg colours by base conversion
        guess = [-1] * self.n_pegs
        for i in range(self.n_pegs):
            guess[i] = action % self.n_colours
            action = action // self.n_colours
        guess.reverse()

        reward = 0.0
        black_pegs = 0
        white_pegs = 0
        to_check = [1] * self.n_pegs
        for i in range(self.n_pegs):
            if guess[i] == self.pattern[i] and to_check[i] == 1:
                black_pegs += 1
                to_check[i] = 0
                break

            for j in range(self.n_pegs):
                if guess[i] == self.pattern[j] and to_check[j] == 1:
                    white_pegs += 1
                    to_check[j] = 0
                    break

        if black_pegs == self.n_pegs:
            done = True
            reward += self.win_reward
        elif self.time >= self.max_guesses:
            done = True
        else:
            done = False

        reward += (
            white_pegs * self.white_reward + black_pegs * self.black_reward)
        self.time += 1

        info = {}
        observation = np.array([white_pegs, black_pegs])
        return observation, reward, done, info

    def reset(self):
        """
        Resets the environment to the starting state

        Used when starting a new episode.
        Returns the observation from the starting state.

        Output:
            observation (int np.array): key peg counters set to zero.
        """

        self.time = 0
        self.pattern = np.random.randint(self.n_colours, size=self.n_pegs)
        observation = np.array([0, 0])
        return observation

    def render(self, mode="human", close=False):
        """
        Renders a visualisation of the environment.

        Added here as it is a standard Gym environment function,
        but not used as the environment is simple enough to visualise
        by printing the relevant variables.
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

    def reinit(self, n_colours=None, n_pegs=None):
        """
        Reinitialises the environmment's variables.

        Changes variables that have other variables dependent on them.

        Input:
            n_colours (int): the new number of code peg colours
            n_pegs (int): the new number of code peg slots
        """

        if n_colours is not None:
            self.n_colours = n_colours
        if n_pegs is not None:
            self.n_pegs = n_pegs

        self.act_dim = self.n_colours * self.n_pegs
        self.pattern = np.array([-1] * self.n_pegs)

        # For gym's "spaces.Box" definition of observation_space/action_space:
        self.low = np.array([0] * self.obs_dim)
        self.high = np.array([self.n_pegs] * self.obs_dim)
        self.action_space = spaces.Discrete(self.act_dim)
        self.observation_space = spaces.Box(self.low, self.high, dtype=int)
