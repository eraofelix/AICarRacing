import numpy as np
import torch
from gymnasium import spaces
from typing import Generator, Tuple, Optional

class RolloutBuffer:
    """
    Stores trajectories collected by the PPO agent and computes advantages and returns.
    Uses Generalized Advantage Estimation (GAE).
    """
    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Box,
                 action_space: spaces.Box,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: str = 'cpu'):

        self.buffer_size = buffer_size
        self.obs_shape = observation_space.shape
        self.action_dim = action_space.shape[0]
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        # Ensure observation space is what we expect from wrappers (k, H, W)
        assert len(self.obs_shape) == 3, f"Expected obs shape (k, H, W), got {self.obs_shape}"

        # Pre-allocate NumPy arrays for efficiency
        self.observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32) # Use float for calculations
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            done: bool,
            value: float,
            log_prob: float):
        """Add a transition to the buffer."""
        if len(log_prob.shape) > 0: # Ensure log_prob is scalar
           log_prob = log_prob.item()
        if len(value.shape) > 0: # Ensure value is scalar
            value = value.item()

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0 # Wrap around

    def compute_returns_and_advantages(self, last_value: float, last_done: bool):
        """
        Compute returns and advantages using GAE after a rollout is finished.

        :param last_value: Value estimate of the state after the last step in the rollout.
        :param last_done: Whether the episode terminated after the last step.
        """
        last_value = last_value.item() if isinstance(last_value, np.ndarray) and last_value.ndim > 0 else last_value
        last_gae_lam = 0
        # Iterate backwards through the buffer
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            # Calculate the TD error (delta)
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]

            # Calculate the GAE advantage
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # Calculate the returns (TD(lambda) estimate)
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generates minibatches of experiences from the buffer.

        :param batch_size: The size of each minibatch.
        :return: A generator yielding tuples of tensors:
                 (observations, actions, old_log_probs, advantages, returns)
        """
        # Determine number of samples currently stored
        num_samples = self.buffer_size if self.full else self.pos
        indices = np.random.permutation(num_samples)

        # Flatten data (remove time dimension, each step is a sample)
        # No need to flatten observations as CNN expects (k, H, W)
        observations = self.observations[:num_samples]
        actions = self.actions[:num_samples]
        old_log_probs = self.log_probs[:num_samples]
        advantages = self.advantages[:num_samples]
        returns = self.returns[:num_samples]

        start_idx = 0
        while start_idx < num_samples:
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
        return self.buffer_size if self.full else self.pos

# Example Usage (Conceptual)
if __name__ == '__main__':
    # Dummy spaces
    k = 4
    h, w = 96, 96
    action_dim = 3
    obs_space = spaces.Box(low=0, high=255, shape=(k, h, w), dtype=np.uint8)
    act_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)

    buffer_sz = 2048
    batch_sz = 64

    buffer = RolloutBuffer(buffer_size=buffer_sz, observation_space=obs_space, action_space=act_space)

    print(f"Buffer initialized. Size: {buffer.size()}")

    # Simulate adding data (normally from agent interaction)
    dummy_obs = obs_space.sample()
    dummy_action = act_space.sample()
    dummy_reward = 0.5
    dummy_done = False
    dummy_value = np.array([0.1]) # Example value from critic
    dummy_log_prob = np.array([-0.5]) # Example log_prob from actor

    for i in range(buffer_sz + 10): # Fill buffer and wrap around a bit
        buffer.add(dummy_obs, dummy_action, dummy_reward, i % 100 == 0, dummy_value, dummy_log_prob)

    print(f"Buffer after adding data. Size: {buffer.size()}, Pos: {buffer.pos}, Full: {buffer.full}")

    # Simulate computing returns/advantages
    last_val = np.array([0.05])
    last_done = True
    buffer.compute_returns_and_advantages(last_val, last_done)
    print("Computed returns and advantages.")
    print(f"Sample Advantages: {buffer.advantages[:5]}...")
    print(f"Sample Returns: {buffer.returns[:5]}...")

    # Simulate getting batches
    batch_count = 0
    print(f"\nIterating through batches of size {batch_sz}:")
    for batch in buffer.get_batches(batch_sz):
        obs_b, act_b, logp_b, adv_b, ret_b = batch
        if batch_count < 2: # Print first couple
            print(f"  Batch {batch_count+1}: obs={obs_b.shape}, act={act_b.shape}, logp={logp_b.shape}, adv={adv_b.shape}, ret={ret_b.shape}, device={obs_b.device}")
        batch_count += 1
    print(f"Total batches generated: {batch_count}")
    expected_batches = (buffer_sz // batch_sz) if buffer_sz % batch_sz == 0 else (buffer_sz // batch_sz + 1)
    print(f"Expected batches: {expected_batches}")
    assert batch_count == expected_batches, "Batch count mismatch!"

    print("\nRolloutBuffer seems functional.") 