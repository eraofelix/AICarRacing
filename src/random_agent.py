import numpy as np
import torch
from gymnasium import spaces

class RandomAgent:
    """
    Random agent that samples actions uniformly from the action space.
    
    Implements the same `act` interface as PPOAgent for compatibility with
    the evaluation script.
    """
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box, device: str = 'cpu'):
        """
        Initialize the random agent.
        
        Args:
            observation_space: The environment observation space (not used).
            action_space: The environment action space.
            device: The device to use (not used for random agent).
        """
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        
    def act(self, observation):
        """
        Sample a random action from the action space.
        
        Args:
            observation: The current observation (ignored).
            
        Returns:
            A tuple containing:
                - action: Random action sampled from action space
                - value: Dummy value (zeros)
                - log_prob: Dummy log probability (zeros)
        """
        # Get batch size from observation
        if isinstance(observation, torch.Tensor):
            batch_size = observation.shape[0]
        else:
            # If numpy array
            batch_size = observation.shape[0] if len(observation.shape) > 3 else 1
        
        # Sample random actions from uniform distribution in [-1, 1]
        # CarRacing action space is typically normalized to this range
        actions = np.random.uniform(-1, 1, size=(batch_size, self.action_dim))
        
        # Create dummy values and log probs (not used for random agent)
        values = np.zeros(batch_size)
        log_probs = np.zeros(batch_size)
        
        return actions, values, log_probs 