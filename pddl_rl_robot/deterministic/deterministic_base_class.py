import numpy as np
from abc import ABC, abstractmethod


class DeterministicBaseClass(ABC):
    """
    Base class for deterministic action policies.
    
    This class provides a similar interface to RL models from stable_baselines3,
    particularly implementing a predict method that can be used in the same way
    as RL model predictions in the main execution loop.
    """
    
    def __init__(self, env):
        """
        Initialize the deterministic action policy.
        
        Args:
            env: The environment the policy will interact with
        """
        self.env = env
        self._previous_action = None
        
    @abstractmethod
    def _get_action(self, observation):
        """
        Abstract method that should be implemented by subclasses to determine
        the action based on the current observation.
        
        Args:
            observation: The current observation from the environment
            
        Returns:
            action: The action to take
        """
        pass
    
    def perform(self, observation, deterministic=True):
        """
        Perform the action based on the current observation.
        
        This method mimics the predict method of stable_baselines3 models
        to maintain compatibility with the main execution loop.
        
        Args:
            observation: The current observation from the environment
            deterministic: Flag for deterministic behavior (included for API compatibility)
            
        Returns:
            action: The action to take
        """
        action = self._get_action(observation)
        return action
    
    def get_parameters(self):
        """
        Get the parameters of the policy.
        
        Returns:
            dict: A dictionary of policy parameters
        """
        return {
            "type": "deterministic",
            "name": self.__class__.__name__
        }
        
    def set_previous_action(self, action):
        self._previous_action = action
        
    def get_previous_action(self):
        return self._previous_action
        