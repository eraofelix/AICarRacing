import numpy as np
import torch
from gymnasium import spaces
from typing import Generator, Tuple, Optional

class RolloutBuffer:
    """
    Stores trajectories collected by the PPO agent from parallel environments.

    This buffer collects experiences (observations, actions, rewards, dones, values,
    log probabilities) from multiple environments run in parallel.
    It calculates advantages using Generalized Advantage Estimation (GAE)
    and provides an iterator to yield shuffled minibatches for training.

    Handles vectorized environments and ensures advantages are normalized globally
    across all collected samples before batching.
    """
    def __init__(self, buffer_size: int, observation_space: spaces.Box,
                 action_space: spaces.Box, num_envs: int, gamma: float = 0.99,
                 gae_lambda: float = 0.95, device: str = 'cpu'):
        """
        Initializes the RolloutBuffer.

        Args:
            buffer_size: The number of steps to collect *per environment* before calculating advantages.
            observation_space: The observation space of a single environment.
            action_space: The action space of a single environment.
            num_envs: The number of parallel environments.
            gamma: The discount factor for reward calculation.
            gae_lambda: The lambda factor for Generalized Advantage Estimation (GAE).
            device: The device ('cpu' or 'cuda') to store tensors for batching.
        """
        self.buffer_size = buffer_size # Steps per environment
        self.num_envs = num_envs
        self.total_buffer_size = buffer_size * num_envs # Total storage capacity

        self.obs_shape = observation_space.shape
        self.action_dim = action_space.shape[0]
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        # Ensure observation shape is as expected (e.g., stacked frames: k, H, W)
        assert len(self.obs_shape) == 3, f"RolloutBuffer expects obs shape (k, H, W), got {self.obs_shape}"

        # Pre-allocate NumPy arrays for storing trajectory data
        # Shape: (buffer_size, num_envs, *data_shape)
        self.observations = np.zeros((self.buffer_size, self.num_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.num_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32) # Combined terminated/truncated
        self.values = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32) # Critic value estimates
        self.log_probs = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32) # Log probability of taken actions
        # Fields calculated after rollout
        self.advantages = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32) # Targets for value function
        self.episode_starts = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32) # Track episode boundaries

        self.pos = 0 # Current index in the buffer
        self.full = False # Flag indicating if the buffer has wrapped around

    def reset(self): # Added reset method
        """Resets the buffer position and the full flag. Does not clear data."""
        self.pos = 0
        self.full = False
        # Optionally reset calculated fields if needed, but advantage/return recalculation handles it
        # self.advantages.fill(0)
        # self.returns.fill(0)
        # self.episode_starts.fill(0)

    def add(self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            terminated: np.ndarray,
            truncated: np.ndarray,
            value: np.ndarray,
            log_prob: np.ndarray):
        """
        Adds a transition (from all parallel environments) to the buffer.

        Args:
            obs: Observations from all environments. Shape (num_envs, *obs_shape).
            action: Actions taken in all environments. Shape (num_envs, action_dim).
            reward: Rewards received in all environments. Shape (num_envs,).
            terminated: Termination flags from all environments. Shape (num_envs,).
            truncated: Truncation flags from all environments. Shape (num_envs,).
            value: Value estimates from the critic for the current obs. Shape (num_envs,).
            log_prob: Log probabilities of the taken actions. Shape (num_envs,).
        """
        # Store data at the current position
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        # Clip rewards for stability (optional but common)
        self.rewards[self.pos] = np.clip(reward, -10.0, 10.0)
        self.dones[self.pos] = (terminated | truncated).astype(np.float32) # Store combined done signal
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob

        # Determine if this step is the start of a new episode for each env
        # An episode starts at the beginning (pos=0) or if the previous step was a 'done' step
        if self.pos > 0:
            self.episode_starts[self.pos] = self.dones[self.pos - 1]
        else:
            # The very first step in the buffer is always considered an episode start
            self.episode_starts[self.pos] = 1.0 # Should be np.ones(self.num_envs) ideally, but broadcast works

        # Increment buffer position and handle wrap-around
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0 # Wrap around

    def compute_returns_and_advantages(self, last_values: np.ndarray, last_dones: np.ndarray):
        """
        Computes the returns (targets for the value function) and advantages
        using Generalized Advantage Estimation (GAE).

        This should be called *after* a rollout is complete and before `get_batches`.

        Args:
            last_values: Value estimate of the state reached *after* the last step
                         in the buffer, for each environment. Shape (num_envs,).
            last_dones: Done flags corresponding to the `last_values`. Shape (num_envs,).
        """
        assert last_values.shape == (self.num_envs,), f"Expected last_values shape {(self.num_envs,)}, got {last_values.shape}"
        assert last_dones.shape == (self.num_envs,), f"Expected last_dones shape {(self.num_envs,)}, got {last_dones.shape}"

        # Initialize GAE advantage accumulator
        last_gae_lam = np.zeros(self.num_envs, dtype=np.float32)

        # Iterate backwards through the collected steps
        for step in reversed(range(self.buffer_size)):
            # Determine the value and non-terminal status of the *next* state
            if step == self.buffer_size - 1:
                # If it's the last step in the buffer, the next state is outside the buffer
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values = last_values # Use the provided last_values
            else:
                # Otherwise, the next state is the next step in the buffer
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            # Calculate the TD error (delta) for the current step
            # delta = R_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]

            # Calculate the GAE advantage using the recursive formula
            # A_t = delta_t + gamma * lambda * A_{t+1} * (1 - done_{t+1})
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # Calculate the returns (targets for the value function)
        # R_t = A_t + V(s_t)
        self.returns = self.advantages + self.values

        # Note: GAE calculation implicitly handles episode boundaries because
        # next_non_terminal becomes 0 when an episode ends, stopping the
        # propagation of future advantages/rewards.
        # The `episode_starts` array is primarily for the batch generator.

    def get_batches(self, batch_size: int) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generates minibatches of experiences from the buffer for PPO updates.

        Flattens the data across environments and steps, normalizes advantages
        globally, shuffles the data (while trying to respect episode boundaries
        if possible), and yields batches as PyTorch tensors on the specified device.

        Args:
            batch_size: The desired size of each minibatch.

        Yields:
            A tuple containing tensors for: (observations, actions, old_log_probs,
            advantages, returns).
        """
        # Determine how many steps are actually available in the buffer
        steps_to_use = self.buffer_size if self.full else self.pos
        num_samples = steps_to_use * self.num_envs

        if num_samples == 0:
            print("Warning: RolloutBuffer.get_batches called with zero samples.")
            return # Stop iteration if buffer is empty

        # Ensure advantages and returns are computed before batching
        assert np.any(self.advantages), "Advantages not computed. Call compute_returns_and_advantages first."
        assert np.any(self.returns), "Returns not computed. Call compute_returns_and_advantages first."

        # --- Data Preparation ---
        # Flatten data arrays from (buffer_size, num_envs, ...) to (total_samples, ...)
        observations = self.observations[:steps_to_use].reshape((-1,) + self.obs_shape)
        actions = self.actions[:steps_to_use].reshape((-1, self.action_dim))
        old_log_probs = self.log_probs[:steps_to_use].reshape(-1)
        advantages = self.advantages[:steps_to_use].reshape(-1)
        returns = self.returns[:steps_to_use].reshape(-1)

        # Normalize advantages globally (across all samples in the buffer)
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # --- Index Shuffling ---
        # Create indices for shuffling
        indices = np.arange(num_samples)
        np.random.shuffle(indices) # Simple random shuffling
        # Note: More complex shuffling respecting episode boundaries was removed for simplicity,
        # as standard PPO often shuffles randomly across the entire batch.
        # If respecting boundaries is critical, the previous boundary-aware shuffling logic
        # using `episode_starts` could be reinstated here.

        # --- Batch Generation ---
        start_idx = 0
        while start_idx < num_samples:
            # Get indices for the current minibatch
            batch_indices = indices[start_idx : start_idx + batch_size]

            # Extract data for the batch using the shuffled indices
            # Convert numpy arrays to PyTorch tensors and move to the target device
            obs_batch = torch.as_tensor(observations[batch_indices], dtype=torch.float32).to(self.device)
            actions_batch = torch.as_tensor(actions[batch_indices]).to(self.device)
            log_probs_batch = torch.as_tensor(old_log_probs[batch_indices]).to(self.device)
            advantages_batch = torch.as_tensor(advantages[batch_indices]).to(self.device)
            returns_batch = torch.as_tensor(returns[batch_indices]).to(self.device)

            yield obs_batch, actions_batch, log_probs_batch, advantages_batch, returns_batch

            # Move to the start of the next batch
            start_idx += batch_size

    def size(self) -> int:
        """Returns the total number of transitions currently stored across all environments."""
        return self.total_buffer_size if self.full else self.pos * self.num_envs 