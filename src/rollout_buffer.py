import numpy as np
import torch
from gymnasium import spaces
from typing import Generator, Tuple, Optional

class RolloutBuffer:
    """
    Stores trajectories collected by the PPO agent and computes advantages and returns.
    Uses Generalized Advantage Estimation (GAE) with improved numerical stability.
    """
    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Box,
                 action_space: spaces.Box,
                 num_envs: int,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: str = 'cpu'):

        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.total_buffer_size = buffer_size * num_envs

        self.obs_shape = observation_space.shape
        self.action_dim = action_space.shape[0]
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        assert len(self.obs_shape) == 3, f"Expected obs shape (k, H, W), got {self.obs_shape}"

        # Pre-allocate NumPy arrays for efficiency
        self.observations = np.zeros((self.buffer_size, self.num_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.num_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.episode_starts = np.ones((self.buffer_size, self.num_envs), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            terminated: np.ndarray,
            truncated: np.ndarray,
            value: np.ndarray,
            log_prob: np.ndarray):
        """Add transitions from all parallel environments to the buffer."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = np.clip(reward, -10.0, 10.0)
        self.dones[self.pos] = (terminated | truncated).astype(np.float32)
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        
        # Handle episode starts
        if self.pos > 0:
            self.episode_starts[self.pos] = self.dones[self.pos - 1]
        else:
            self.episode_starts[self.pos] = 1.0

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def compute_returns_and_advantages(self, last_values: np.ndarray, last_dones: np.ndarray):
        """
        Compute returns and advantages using GAE after a rollout is finished.
        Handles multiple environments with improved numerical stability.

        :param last_values: Value estimate of the last state for each environment. Shape (num_envs,)
        :param last_dones: Whether each environment terminated/truncated at the last step. Shape (num_envs,)
        """
        assert last_values.shape == (self.num_envs,), f"Expected last_values shape {(self.num_envs,)}, got {last_values.shape}"
        assert last_dones.shape == (self.num_envs,), f"Expected last_dones shape {(self.num_envs,)}, got {last_dones.shape}"

        last_gae_lam = np.zeros(self.num_envs, dtype=np.float32)
        
        # Iterate backwards through the buffer steps
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            # Calculate the TD error (delta) for all envs
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]

            # Calculate the GAE advantage
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # Calculate the returns (value targets)
        self.returns = self.advantages + self.values
        
        # Special handling for episode boundaries
        for env_idx in range(self.num_envs):
            for step in range(self.buffer_size):
                if self.episode_starts[step, env_idx]:
                    # At episode start, reset advantage calculation
                    if step > 0:
                        self.advantages[step-1, env_idx] = self.returns[step-1, env_idx] - self.values[step-1, env_idx]

    def get_batches(self, batch_size: int) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generates minibatches of experiences from the buffer.
        Ensures proper normalization of advantages globally across all environments.

        :param batch_size: The size of each minibatch.
        :return: A generator yielding tuples of tensors:
                 (observations, actions, old_log_probs, advantages, returns)
        """
        steps_to_use = self.buffer_size if self.full else self.pos
        num_samples = steps_to_use * self.num_envs

        if num_samples == 0:
            return

        # Normalize advantages globally across all environments
        all_advantages = self.advantages[:steps_to_use].reshape(-1)
        if len(all_advantages) > 0 and all_advantages.std() > 0:
            adv_mean = all_advantages.mean()
            adv_std = all_advantages.std()
            normalized_advantages = (self.advantages[:steps_to_use] - adv_mean) / (adv_std + 1e-8)
        else:
            normalized_advantages = self.advantages[:steps_to_use].copy()
                    
        # Flatten data across environments and steps before shuffling
        observations = self.observations[:steps_to_use].reshape((-1,) + self.obs_shape)
        actions = self.actions[:steps_to_use].reshape((-1, self.action_dim))
        old_log_probs = self.log_probs[:steps_to_use].reshape(-1)
        advantages = normalized_advantages[:steps_to_use].reshape(-1)
        returns = self.returns[:steps_to_use].reshape(-1)
        
        # Create episode masks to avoid shuffling across episode boundaries
        episode_masks = np.ones(num_samples, dtype=bool)
        episode_ends = np.where(self.dones[:steps_to_use].reshape(-1))[0]
        if len(episode_ends) > 0:
            for end_pos in episode_ends:
                if end_pos + 1 < num_samples:
                    episode_masks[end_pos + 1] = False
                    
        # Create shuffled indices respecting episode boundaries
        continuous_indices = np.arange(num_samples)
        episode_starts = np.where(~episode_masks)[0]
        valid_indices = []
        
        # Add indices for each continuous episode segment
        start_idx = 0
        for end_idx in episode_starts:
            if end_idx > start_idx:
                segment_indices = continuous_indices[start_idx:end_idx]
                np.random.shuffle(segment_indices)
                valid_indices.extend(segment_indices)
            start_idx = end_idx + 1
            
        # Add last segment if needed
        if start_idx < num_samples:
            segment_indices = continuous_indices[start_idx:num_samples]
            np.random.shuffle(segment_indices)
            valid_indices.extend(segment_indices)
            
        indices = np.array(valid_indices)
        
        # If we don't have enough indices, fall back to regular shuffling
        if len(indices) < 0.5 * num_samples:
            indices = np.random.permutation(num_samples)

        start_idx = 0
        while start_idx < len(indices):
            batch_indices = indices[start_idx : start_idx + batch_size]
            end_idx = start_idx + batch_size

            # Extract batch data using shuffled indices
            obs_batch = torch.as_tensor(observations[batch_indices]).to(self.device)
            actions_batch = torch.as_tensor(actions[batch_indices]).to(self.device)
            log_probs_batch = torch.as_tensor(old_log_probs[batch_indices]).to(self.device)
            advantages_batch = torch.as_tensor(advantages[batch_indices]).to(self.device)
            returns_batch = torch.as_tensor(returns[batch_indices]).to(self.device)

            yield obs_batch, actions_batch, log_probs_batch, advantages_batch, returns_batch

            start_idx = end_idx

    def size(self) -> int:
        """Returns the number of elements currently stored in the buffer."""
        return self.total_buffer_size if self.full else self.pos * self.num_envs 