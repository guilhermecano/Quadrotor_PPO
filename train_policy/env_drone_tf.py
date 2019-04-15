import numpy as np
from tensorforce import TensorForceError
from tensorforce.environments import Environment
from drone_vrep_api import DroneVrepEnv
import vrep_env
import vrep
import vrepConst
import tensorflow as tf

class EnvArDrone(Environment):
    def __init__(self):
        self.sim = DroneVrepEnv()

    def __str__(self):
        return 'Vrep-Env drone environment for Tensorforce'

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """
        obs = self.sim._reset(2)
        return obs

    def execute(self, action):
        """
        Executes action, observes next state(s) and reward.

        Args:
            action: action to execute.

        Returns:
            (Dict of) next state(s), boolean indicating terminal, and reward signal.
        """
        state, reward, done,_ = self.sim._step(action)
        return state, done, reward

    @property
    def states(self):
        """
        Return the state space. Might include subdicts if multiple states are available simultaneously.

        Returns: dict of state properties (shape and type).

        """
        return dict(shape=tuple(self.sim.observation_space.shape), type='float')

    @property
    def actions(self):
        """
        Return the action space. Might include subdicts if multiple actions are available simultaneously.

        Returns: dict of action properties (continuous, number of actions)

        """
        return dict(type='float', shape=self.sim.action_space.low.shape,
                            min_value=np.float(self.sim.action_space.low[0]),
                            max_value=np.float(self.sim.action_space.high[0]))

