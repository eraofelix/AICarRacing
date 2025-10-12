import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
import argparse
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import gymnasium.vector
from contextlib import nullcontext # Import nullcontext for mixed precision handling
from torch.distributions import Normal
import itertools
from typing import Generator, Tuple, Optional
import cv2
from gymnasium import spaces
from gymnasium.spaces import Box
from datetime import datetime

# Constants for clamping the log standard deviation
LOG_STD_MAX = 2
LOG_STD_MIN = -20

LAUNCH_DATE = datetime.now().strftime('%Y%m%d_%H%M%S')

# Ryzen 7 7800X3D (8 Cores), 32 GB RAM, and RTX 3080 GPU.
config = {
    # Environment
    "env_id": "CarRacing-v3",           # ID for the Gymnasium environment
    "frame_stack": 4,                   # Number of consecutive frames to stack as input
    "num_envs": 32,                      # Number of parallel environments for vectorized training (Change based on your CPU/GPU)
    "max_episode_steps": 1000,          # Maximum steps allowed per episode
    "seed": 42,                         # Seed used for all evaluations and model training

    # PPO Core Parameters
    "total_timesteps": 6_000_000,       # Total number of training steps across all environments
    "learning_rate": 3e-4,              # Learning rate for the optimizers
    "buffer_size": 32768,                # Size of the rollout buffer per environment before updates
    "batch_size": 2048,                  # Minibatch size for PPO updates
    "ppo_epochs": 3,                    # Number of optimization epochs per rollout
    "gamma": 0.99,                      # Discount factor for future rewards
    "gae_lambda": 0.95,                  # Factor for Generalized Advantage Estimation (GAE)
    "clip_epsilon": 0.15,               # Clipping parameter for the PPO policy loss
    "vf_coef": 0.5,                     # Coefficient for the value function loss in the total loss
    "ent_coef": 0.008,                  # Coefficient for the entropy bonus in the total loss
    "max_grad_norm": 0.75,              # Maximum norm for gradient clipping
    "target_kl": 0.2,                  # Target KL divergence threshold (for monitoring, not early stopping)
    "features_dim": 256,                # Dimensionality of features extracted by the CNN

    # Agent specific hyperparameters (previously defaults in PPOAgent)
    "initial_action_std": 0.75,          # Initial standard deviation for the action distribution
    "weight_decay": 1e-5,               # Weight decay (L2 regularization) for optimizers
    "fixed_std": False,                 # Whether to use a fixed or learned action standard deviation
    "lr_warmup_steps": 10000,            # Number of steps for learning rate warmup
    "min_learning_rate": 1e-6,          # Minimum learning rate allowed by the scheduler

    # Reward shaping
    "use_reward_shaping": True,         # Flag to enable custom reward shaping
    "velocity_reward_weight": 0.005,    # Weight for the velocity component of the reward
    "survival_reward": 0.05,            # Constant reward added at each step for surviving
    "track_penalty": 5.0,               # Penalty for going off-track
    "steering_smooth_weight": 0.3,      # Weight for the penalty encouraging smooth steering
    "acceleration_while_turning_penalty_weight": 0.8, # Weight for penalizing acceleration during sharp turns

    # Performance optimizations
    "torch_num_threads": 1,             # Number of threads for PyTorch CPU operations
    "mixed_precision": True,           # Flag to enable/disable mixed precision training (requires CUDA)
    "pin_memory": True,                 # Flag to use pinned memory for faster CPU-GPU data transfer
    "async_envs": True,                # Flag to use asynchronous vectorized environments

    # Logging and saving
    "log_interval": 1,                  # Number of rollouts between logging summary statistics
    "save_interval": 100,                # Number of rollouts between saving model checkpoints
    "save_dir": f"./models/ppo_carracing_{LAUNCH_DATE}",  # Directory to save model checkpoints
    "log_dir": f"./logs/ppo_carracing_{LAUNCH_DATE}",     # Directory to save TensorBoard logs
    "checkpoint_path": "./models/ppo_carracing/restart515.pth", # Path to load a pre-trained model checkpoint
    "device": "cuda" if torch.cuda.is_available() else "cpu", # Automatically select CUDA if available, else CPU
}

# Set performance-enhancing environment variables for multi-threading
os.environ['OMP_NUM_THREADS'] = str(config["torch_num_threads"])
os.environ['MKL_NUM_THREADS'] = str(config["torch_num_threads"])
torch.set_num_threads(config["torch_num_threads"])
if config["device"] == "cuda":
    torch.backends.cudnn.benchmark = True # Enable cuDNN auto-tuner for best performance
    if config["mixed_precision"]:
        # Enable TensorFloat-32 for faster matrix multiplications on compatible hardware
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

class RewardShapingWrapper(gym.Wrapper):
    """
    Applies custom reward shaping to the CarRacing environment.

    Adds rewards for velocity, survival, staying on track, and smooth driving.
    Adds penalties for going off-track, jerky steering, and accelerating while turning sharply.
    """
    def __init__(self, env, velocity_weight: float = 0.005, survival_reward: float = 0.05,
                 track_penalty: float = 2.0, steering_smooth_weight: float = 0.1,
                 acceleration_while_turning_penalty_weight: float = 0.5):
        """
        Initializes the RewardShapingWrapper.

        Args:
            env: The Gymnasium environment to wrap.
            velocity_weight: Weight for the speed-based reward component.
            survival_reward: Constant reward added at each step.
            track_penalty: Penalty applied for each step off-track.
            steering_smooth_weight: Weight for the steering change penalty.
            acceleration_while_turning_penalty_weight: Weight for the penalty for accelerating during sharp turns.
        """
        super().__init__(env)
        self.velocity_weight = velocity_weight
        self.survival_reward = survival_reward
        self.track_penalty = track_penalty
        self.steering_smooth_weight = steering_smooth_weight
        self.acceleration_while_turning_penalty_weight = acceleration_while_turning_penalty_weight

        # Track previous action for smoothness calculations
        self.last_steering = 0.0
        self.last_speed = 0.0

        # Track cumulative reward components per episode
        self.episode_velocity_rewards = 0.0
        self.episode_survival_rewards = 0.0
        self.episode_track_penalties = 0.0
        self.episode_steering_penalties = 0.0
        self.episode_acceleration_while_turning_penalties = 0.0
        self.steps_off_track = 0

        # Additional reward components
        self.centerline_reward_weight = 0.5 # Reward for staying near the track center
        self.track_return_weight = 0.3    # Reward for steering back towards the track when off-track
        self.speed_consistency_weight = 0.05 # Penalty for large speed changes

    def reset(self, **kwargs):
        """Resets the environment and internal state trackers."""
        self.last_steering = 0.0
        self.last_speed = 0.0

        # Reset episode trackers
        self.episode_velocity_rewards = 0.0
        self.episode_survival_rewards = 0.0
        self.episode_track_penalties = 0.0
        self.episode_steering_penalties = 0.0
        self.episode_acceleration_while_turning_penalties = 0.0
        self.steps_off_track = 0

        obs, info = self.env.reset(**kwargs)

        # Initialize reward components in the info dictionary
        info['velocity_rewards'] = 0.0
        info['survival_rewards'] = 0.0
        info['track_penalties'] = 0.0
        info['steering_penalties'] = 0.0
        info['acceleration_while_turning_penalties'] = 0.0
        info['steps_off_track'] = 0

        return obs, info

    def step(self, action):
        """
        Steps the environment and applies reward shaping.

        Args:
            action: The action taken by the agent.

        Returns:
            A tuple containing (observation, shaped_reward, terminated, truncated, info).
            The info dictionary includes detailed reward components.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Ensure info is a dictionary
        if info is None:
             info = {}

        # Extract action components
        steering = action[0]
        gas = action[1]
        # brake = action[2] # Brake is not used in current reward shaping

        # Initialize reward components for this step
        step_velocity_reward = 0.0
        step_survival_reward = 0.0
        step_track_penalty = 0.0
        step_steering_penalty = 0.0
        step_acceleration_while_turning_penalty = 0.0
        step_centerline_reward = 0.0
        step_speed_consistency_reward = 0.0
        step_track_return_reward = 0.0
        off_track = False

        # 1. Velocity rewards: Encourage higher speeds
        speed = info.get('speed', 0.0) # Get speed from info if available
        if speed > 0:
            step_velocity_reward = speed * self.velocity_weight
            reward += step_velocity_reward
            self.episode_velocity_rewards += step_velocity_reward

        # 2. Survival rewards: Encourage longer episodes
        step_survival_reward = self.survival_reward
        reward += step_survival_reward
        self.episode_survival_rewards += step_survival_reward

        # 3. Track adherence penalty: Penalize going off-track (onto grass)
        #    Requires RGB observation input to this wrapper.
        if len(obs.shape) == 3 and obs.shape[2] == 3: # Check if observation is RGB
            # Check bottom center pixels for green color (indicative of grass)
            car_area = obs[84:94, 42:54, :]
            green_channel = car_area[:, :, 1]
            red_channel = car_area[:, :, 0]

            # Heuristic: High green and low red likely means off-track
            off_track = np.mean(green_channel) > 150 and np.mean(red_channel) < 100

            if off_track:
                step_track_penalty = self.track_penalty
                reward -= step_track_penalty
                self.episode_track_penalties += step_track_penalty
                self.steps_off_track += 1

                # Add guidance reward to steer back towards the track center
                # Simplified: assumes track is generally ahead
                track_direction = np.array([1.0, 0.0])
                # Approximate car direction based on steering angle
                car_direction = np.array([np.cos(steering * np.pi / 2), np.sin(steering * np.pi / 2)])
                # Reward alignment with track direction
                step_track_return_reward = np.dot(track_direction, car_direction) * self.track_return_weight
                reward += step_track_return_reward
            else:
                # 5. Centerline reward: Reward staying near the center (higher red channel value)
                road_redness = np.mean(red_channel)
                step_centerline_reward = min(road_redness / 200, 1.0) * self.centerline_reward_weight
                reward += step_centerline_reward

        # 4. Steering smoothness penalty: Penalize large changes in steering, especially at high speed
        steering_change = abs(steering - self.last_steering)
        step_steering_penalty = steering_change * self.steering_smooth_weight * (1.0 + speed * 0.1)
        reward -= step_steering_penalty
        self.episode_steering_penalties += step_steering_penalty

        # 6. Speed consistency reward: Penalize large changes in speed
        speed_change = abs(speed - self.last_speed)
        step_speed_consistency_reward = -speed_change * self.speed_consistency_weight
        reward += step_speed_consistency_reward

        # 7. Acceleration while turning penalty: Penalize applying gas during sharp turns
        steering_threshold = 0.4 # Angle threshold for penalty
        gas_threshold = 0.1      # Gas threshold for penalty
        if abs(steering) > steering_threshold and gas > gas_threshold:
            step_acceleration_while_turning_penalty = (
                self.acceleration_while_turning_penalty_weight *
                (gas - gas_threshold) *
                (abs(steering) - steering_threshold)
            )
            reward -= step_acceleration_while_turning_penalty
            self.episode_acceleration_while_turning_penalties += step_acceleration_while_turning_penalty

        # Update state for next step
        self.last_steering = steering
        self.last_speed = speed

        # Add step-wise reward components to info
        info['velocity_rewards'] = step_velocity_reward
        info['survival_rewards'] = step_survival_reward
        info['track_penalties'] = step_track_penalty
        info['steering_penalties'] = step_steering_penalty
        info['acceleration_while_turning_penalties'] = step_acceleration_while_turning_penalty
        info['centerline_rewards'] = step_centerline_reward
        info['speed_consistency_rewards'] = step_speed_consistency_reward
        info['track_return_rewards'] = step_track_return_reward
        info['off_track'] = off_track

        # Add cumulative episode totals to info (useful for final info dict)
        info['episode_velocity_rewards'] = self.episode_velocity_rewards
        info['episode_survival_rewards'] = self.episode_survival_rewards
        info['episode_track_penalties'] = self.episode_track_penalties
        info['episode_steering_penalties'] = self.episode_steering_penalties
        info['episode_acceleration_while_turning_penalties'] = self.episode_acceleration_while_turning_penalties
        info['steps_off_track'] = self.steps_off_track

        return obs, reward, terminated, truncated, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2] # Height, Width
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps: int = 1000):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, k: int):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        assert len(env.observation_space.shape) == 2, f"expects 2D input (H, W), got {env.observation_space.shape}"
        stacked_shape = (k,) + env.observation_space.shape # (k, H, W)
        self.observation_space = Box(low=0, high=255, shape=stacked_shape, dtype=env.observation_space.dtype)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.frames.append(observation)
        return self._get_ob()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob(), info

    def _get_ob(self) -> np.ndarray:
        assert len(self.frames) == self.k, f"Frame buffer size mismatch: expected {self.k}, got {len(self.frames)}"
        return np.stack(self.frames, axis=0)

class CNNFeatureExtractor(nn.Module):
    """
    Architecture:
        - Conv2D (16 filters, 8x8 kernel, stride 4)  -> ReLU Activation -> Dropout (0.1)
        - Conv2D (32 filters, 4x4 kernel, stride 2)  -> ReLU Activation -> Dropout (0.1)
        - Conv2D (64 filters, 3x3 kernel, stride 1)  -> ReLU Activation -> Dropout (0.1)
        - Flatten -> Linear (features_dim) -> Dropout (0.2) -> ReLU Activation

    Includes internal normalization of input observations (division by 255.0).
    Uses Kaiming Normal initialization for convolutional and linear layers.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__()
        assert isinstance(observation_space, spaces.Box), f"expects a Box, got {type(observation_space)}"
        assert len(observation_space.shape) == 3, f"CNNFeatureExtractor expects(k, H, W), got {observation_space.shape}"
        self.features_dim = features_dim
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0), nn.ReLU(), nn.Dropout2d(0.1), # Dropout after activation
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0), nn.ReLU(), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(), nn.Dropout2d(0.1),
            nn.Flatten(), # Flatten the output for the linear layer
        )

        # Compute the flattened size automatically by doing a dummy forward pass
        with torch.no_grad():
            dummy_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_obs / 255.0).shape[1] # Get the flattened size
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.Dropout(0.2), nn.ReLU())
        self._initialize_weights()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normalized_obs = observations.float() / 255.0
        cnn_features = self.cnn(normalized_obs)
        features = self.linear(cnn_features)
        return features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Normal initialization for ReLU non-linearity
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, features_dim: int, action_dim: int, initial_action_std: float = 1.0, fixed_std: bool = False):
        super().__init__()
        self.action_dim = action_dim
        self.initial_action_std = initial_action_std
        self.fixed_std = fixed_std

        hidden_dim = 256 # Dimension of hidden layers
        self.fc1 = nn.Linear(features_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer for the mean of the action distribution
        self.fc_mean = nn.Linear(hidden_dim, action_dim)

        # Output layer for the log standard deviation (learned or fixed)
        if not fixed_std:
            self.fc_logstd = nn.Linear(hidden_dim, action_dim)
            # Initialize log_std weights near zero and bias to initial_action_std
            self.fc_logstd.weight.data.fill_(0.0)
            self.fc_logstd.bias.data.fill_(np.log(self.initial_action_std))
        else:
            # Use a non-trainable parameter for fixed standard deviation
            self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(self.initial_action_std), requires_grad=False)

        # Initialize the mean layer weights orthogonally with small gain for stability
        nn.init.orthogonal_(self.fc_mean.weight, gain=0.01)
        nn.init.constant_(self.fc_mean.bias, 0.0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))

        # Apply tanh activation to constrain the mean to [-1, 1]
        mean = torch.tanh(self.fc_mean(x))

        if not self.fixed_std:
            log_std = self.fc_logstd(x)
            # Clamp log_std to prevent numerical instability
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        else:
            # Expand the fixed log_std to match the batch size
            batch_size = mean.size(0)
            log_std = self.log_std.expand(batch_size, -1)

        return mean, log_std

    def get_action_dist(self, features: torch.Tensor) -> Normal:
        """
        Creates the action distribution for the given features.

        Args:
            features: Feature tensor from the CNN (Batch, features_dim).

        Returns:
            A PyTorch Normal distribution object representing the policy.
        """
        mean, log_std = self.forward(features)
        std = log_std.exp() # Convert log_std to std
        return Normal(mean, std)

    def evaluate_actions(self, features: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the log probability and entropy of given actions under the current policy.

        Used during the PPO update step.

        Args:
            features: Features tensor (Batch, features_dim).
            actions: Actions tensor (Batch, action_dim).

        Returns:
            A tuple containing:
                - log_prob: Log probability of the actions (Batch,).
                - entropy: Entropy of the action distribution (Batch,).
        """
        action_dist = self.get_action_dist(features)
        log_prob = action_dist.log_prob(actions).sum(axis=-1) # Sum log probs across action dimensions
        entropy = action_dist.entropy().sum(axis=-1) # Sum entropy across action dimensions
        return log_prob, entropy

class Critic(nn.Module):
    """
    Critic Network (Value Function) for PPO.

    Takes features extracted by a CNN and outputs a scalar value representing the estimated value of the state.
    """
    def __init__(self, features_dim: int):
        """
        Initializes the Critic network.

        Args:
            features_dim: Dimensionality of the input feature vector from the CNN.
        """
        super().__init__()
        hidden_dim = 256 # Dimension of hidden layers
        self.fc1 = nn.Linear(features_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer for the state value (a single scalar)
        self.fc_value = nn.Linear(hidden_dim, 1)

        # Initialize the value layer weights/biases with small values
        nn.init.uniform_(self.fc_value.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_value.bias, -3e-3, 3e-3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the Critic network.

        Args:
            features: Feature tensor from the CNN (Batch, features_dim).

        Returns:
            The estimated state value (Batch,).
        """
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        value = self.fc_value(x)
        return value.squeeze(-1) # Remove the last dimension (size 1)

class RolloutBuffer:
    def __init__(self, buffer_size: int, observation_space: spaces.Box,
                 action_space: spaces.Box, num_envs: int, gamma: float = 0.99,
                 gae_lambda: float = 0.95, device: str = 'cpu'):
        """
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

    def add(self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            terminated: np.ndarray,
            truncated: np.ndarray,
            value: np.ndarray,
            log_prob: np.ndarray):
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = np.clip(reward, -10.0, 10.0)
        self.dones[self.pos] = (terminated | truncated).astype(np.float32) # Store combined done signal
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob

        if self.pos > 0:
            self.episode_starts[self.pos] = self.dones[self.pos - 1]
        else:
            self.episode_starts[self.pos] = 1.0 # Should be np.ones(self.num_envs) ideally, but broadcast works

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

class PPOAgent:
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box,
                 config: dict, device: str = 'cpu'):
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self.device = torch.device(device)

        # --- Extract Hyperparameters from Config (with defaults for safety otherwise things break) ---
        self.initial_lr = config.get("learning_rate", 1e-4)
        self.lr = self.initial_lr # Current learning rate starts at initial
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.1)
        self.epochs = config.get("ppo_epochs", 5)
        self.batch_size = config.get("batch_size", 64)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.ent_coef = config.get("ent_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        features_dim = config.get("features_dim", 64) # Needed for network init
        self.target_kl = config.get("target_kl", 0.02)
        self.initial_action_std = config.get("initial_action_std", 1.0)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.fixed_std = config.get("fixed_std", False)
        self.lr_warmup_steps = config.get("lr_warmup_steps", 5000)
        self.min_lr = config.get("min_learning_rate", 1e-7) # Store min_lr

        self.steps_done = 0 # Counter for learning rate scheduling

        # --- Network Initialization ---
        self.feature_extractor = CNNFeatureExtractor(observation_space, features_dim).to(self.device)
        self.actor = Actor(features_dim, self.action_dim,
                           initial_action_std=self.initial_action_std,
                           fixed_std=self.fixed_std).to(self.device)
        self.critic = Critic(features_dim).to(self.device)

        # --- Optimizer Setup ---
        # Actor optimizer optimizes only the actor parameters
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.initial_lr, # Use initial_lr here
                                          eps=1e-4, weight_decay=self.weight_decay) # Increased eps
        # Critic optimizer optimizes critic and feature extractor parameters together
        self.critic_optimizer = optim.Adam(
            itertools.chain(self.critic.parameters(), self.feature_extractor.parameters()),
            lr=self.initial_lr, eps=1e-4, weight_decay=self.weight_decay # Use initial_lr here, Increased eps
        )

        print(f"Device: {self.device} | FeatExt: {sum(p.numel() for p in self.feature_extractor.parameters()):,} | Actor: {sum(p.numel() for p in self.actor.parameters()):,} | Critic: {sum(p.numel() for p in self.critic.parameters()):,} | Std: {'Fixed' if self.fixed_std else 'Learned'}({self.initial_action_std}) | LR: {self.initial_lr:.2e}(warmup:{self.lr_warmup_steps:,})")

    def act(self, observation: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Selects an action based on the current observation using the actor network.

        Also returns the estimated value from the critic and the log probability
        of the selected action.

        Args:
            observation: Current environment observation(s) (Batch, k, H, W) as a tensor.

        Returns:
            A tuple containing:
                - action: Sampled action(s) from the policy distribution (Batch, action_dim) as NumPy array.
                - value: Estimated state value(s) from the critic (Batch,) as NumPy array.
                - log_prob: Log probability of the sampled action(s) (Batch,) as NumPy array.
        """
        # Set networks to evaluation mode
        self.feature_extractor.eval()
        self.actor.eval()
        self.critic.eval()

        # Ensure observation is a tensor on the correct device
        if not isinstance(observation, torch.Tensor):
            # Move to device only if it's not already there (minor optimization)
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        elif observation.device != self.device:
            observation_tensor = observation.to(self.device)
        else:
            observation_tensor = observation

        # --- Add batch dimension if missing ---
        if observation_tensor.ndim == 3:
            observation_tensor = observation_tensor.unsqueeze(0) # Add batch dim: (k, H, W) -> (1, k, H, W)
        # -------------------------------------

        # Perform inference without tracking gradients
        with torch.no_grad():
            # Note: Feature extractor handles normalization internally
            features = self.feature_extractor(observation_tensor)
            value = self.critic(features)
            action_dist = self.actor.get_action_dist(features)
            action = action_dist.sample() # Sample action from the distribution
            log_prob = action_dist.log_prob(action).sum(axis=-1) # Calculate log probability

        # Convert results to NumPy arrays for interaction with the environment
        action_np = action.detach().cpu().numpy()
        value_np = value.detach().cpu().numpy()
        log_prob_np = log_prob.detach().cpu().numpy()

        return action_np, value_np, log_prob_np

    def update_learning_rate(self, total_timesteps: int) -> float:
        """
        Implements linear warmup followed by cosine decay.
        """
        self.steps_done += 1 # Increment internal step counter

        # Warmup Phase: Linearly increase LR from 30% to 100% of initial_lr
        if self.steps_done < self.lr_warmup_steps:
            alpha = self.steps_done / self.lr_warmup_steps
            current_lr = self.initial_lr * (0.3 + 0.7 * alpha)
        else:
            progress = min((self.steps_done - self.lr_warmup_steps) / (total_timesteps - self.lr_warmup_steps), 1.0)
            current_lr = self.initial_lr * 0.5 * (1.0 + np.cos(np.pi * progress))

        current_lr = max(current_lr, self.min_lr) # Use self.min_lr
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = current_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = current_lr

        self.lr = current_lr # Store the current LR
        return current_lr

    def learn_mixed_precision(self, rollout_buffer, scaler: torch.cuda.amp.GradScaler):
        """
        Performs the PPO learning update using mixed precision (FP16/FP32).

        Requires a CUDA device and a GradScaler.

        Args:
            rollout_buffer: Buffer containing collected experiences.
            scaler: PyTorch GradScaler for handling mixed precision gradients.

        Returns:
            A dictionary containing training metrics (losses, KL divergence, etc.).
        """
        # Set networks to training mode
        self.feature_extractor.train()
        self.actor.train()
        self.critic.train()

        # Accumulate metrics across all epochs and batches
        all_policy_losses, all_value_losses, all_entropy_losses = [], [], []
        all_kl_divs, clip_fractions = [], []

        # PPO Optimization Loop
        for epoch in range(self.epochs):
            epoch_kl_divs = [] # Track KL divergence per epoch for potential early stopping

            # Iterate over minibatches from the rollout buffer
            for batch in rollout_buffer.get_batches(self.batch_size):
                obs_batch, actions_batch, old_log_probs_batch, advantages_batch, returns_batch = batch

                # Forward pass within autocast context for mixed precision
                with torch.cuda.amp.autocast():
                    # Feature extraction (normalization inside extractor)
                    features = self.feature_extractor(obs_batch)
                    # Get current value estimates and policy evaluation
                    values = self.critic(features)
                    log_probs, entropy = self.actor.evaluate_actions(features, actions_batch)


                    # --- PPO Loss Calculation ---
                    # Ratio of new policy probability to old policy probability
                    ratio = torch.exp(log_probs - old_log_probs_batch)

                    # Clipped Surrogate Objective (Policy Loss)
                    policy_loss_1 = advantages_batch * ratio
                    policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                    value_loss = F.mse_loss(values, returns_batch)

                    # Entropy Bonus (encourages exploration)
                    entropy_loss = -torch.mean(entropy)

                    # Total Loss
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # --- Backward Pass and Optimization --- 
                # Zero gradients before backward pass
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Scale the loss and compute gradients using the scaler
                scaler.scale(loss).backward()

                # Unscale gradients before clipping
                scaler.unscale_(self.actor_optimizer)
                scaler.unscale_(self.critic_optimizer)
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic.parameters(), self.feature_extractor.parameters()),
                    self.max_grad_norm
                )

                # Step the optimizers using the scaler
                scaler.step(self.actor_optimizer)
                scaler.step(self.critic_optimizer)

                # Update the scaler for the next iteration
                scaler.update()

                # --- Logging Metrics (within epoch loop) ---
                with torch.no_grad():
                    # Approximate KL divergence between old and new policies
                    approx_kl = 0.5 * torch.mean((log_probs - old_log_probs_batch)**2).item()
                    # Fraction of samples where the policy ratio was clipped
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float()).item()

                # Store metrics for this batch
                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(entropy_loss.item())
                all_kl_divs.append(approx_kl)
                epoch_kl_divs.append(approx_kl)
                clip_fractions.append(clip_frac)

            # --- End of Epoch KL Check (Optional Early Stopping) ---
            epoch_mean_kl = np.mean(epoch_kl_divs)
            if self.target_kl is not None and epoch_mean_kl > self.target_kl * 1.5:
                print(f"Warning: Early stopping PPO epoch {epoch+1} due to high KL divergence: {epoch_mean_kl:.4f} > {self.target_kl*1.5:.4f}")
                break #if kl divergence is too high, break the loop

        # --- Return Averaged Metrics ---
        avg_metrics = {
            "policy_loss": np.mean(all_policy_losses),
            "value_loss": np.mean(all_value_losses),
            "entropy_loss": np.mean(all_entropy_losses),
            "approx_kl": np.mean(all_kl_divs),
            "clip_fraction": np.mean(clip_fractions),
        }
        return avg_metrics

    def learn(self, rollout_buffer):
        """
        Performs the PPO learning update using standard precision (FP32).

        Args:
            rollout_buffer: Buffer containing collected experiences.

        Returns:
            A dictionary containing training metrics (losses, KL divergence, etc.).
        """
        # Set networks to training mode
        self.feature_extractor.train()
        self.actor.train()
        self.critic.train()

        # Accumulate metrics across all epochs and batches
        all_policy_losses, all_value_losses, all_entropy_losses = [], [], []
        all_kl_divs, clip_fractions = [], []

        # PPO Optimization Loop
        for epoch in range(self.epochs):
            epoch_kl_divs = [] # Track KL divergence per epoch

            # Iterate over minibatches from the rollout buffer
            for batch in rollout_buffer.get_batches(self.batch_size):
                obs_batch, actions_batch, old_log_probs_batch, advantages_batch, returns_batch = batch

                # --- Forward Pass ---
                # Feature extraction (normalization inside extractor)
                features = self.feature_extractor(obs_batch)
                # Get current value estimates and policy evaluation
                values = self.critic(features)
                log_probs, entropy = self.actor.evaluate_actions(features, actions_batch)

                # --- PPO Loss Calculation ---
                # Ratio of new policy probability to old policy probability
                ratio = torch.exp(log_probs - old_log_probs_batch)
                # ratio = torch.clamp(ratio, 0.1, 10.0) # Optional clamping

                # Clipped Surrogate Objective (Policy Loss)
                policy_loss_1 = advantages_batch * ratio
                policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value Function Loss (Mean Squared Error)
                value_loss = F.mse_loss(values, returns_batch)

                # Entropy Bonus
                entropy_loss = -torch.mean(entropy)

                # Total Loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # --- Backward Pass and Optimization ---
                # Zero gradients
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                # Compute gradients
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic.parameters(), self.feature_extractor.parameters()),
                    self.max_grad_norm
                )
                # Update weights
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # --- Logging Metrics ---
                with torch.no_grad():
                    # Approximate KL divergence
                    approx_kl = 0.5 * torch.mean((log_probs - old_log_probs_batch)**2).item()
                    # Clip fraction
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float()).item()

                # Store metrics for this batch
                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(entropy_loss.item())
                all_kl_divs.append(approx_kl)
                epoch_kl_divs.append(approx_kl)
                clip_fractions.append(clip_frac)

            # --- End of Epoch KL Check ---
            epoch_mean_kl = np.mean(epoch_kl_divs)
            if self.target_kl is not None and epoch_mean_kl > self.target_kl * 1.5:
                print(f"Warning: PPO epoch {epoch+1} KL divergence high: {epoch_mean_kl:.4f} > {self.target_kl*1.5:.4f}")
                break # Optional early stopping

        # --- Return Averaged Metrics ---
        avg_metrics = {
            "policy_loss": np.mean(all_policy_losses),
            "value_loss": np.mean(all_value_losses),
            "entropy_loss": np.mean(all_entropy_losses),
            "approx_kl": np.mean(all_kl_divs),
            "clip_fraction": np.mean(clip_fractions),
        }
        return avg_metrics

    def save(self, path: str):
        """
        Saves the state dictionaries of the feature extractor, actor, and critic networks.

        Note: Does not save optimizer states or other training parameters.
              Use checkpoint saving in the training script for full state saving.

        Args:
            path: The file path to save the model weights.
        """
        print(f"Saving model components to {path}")
        torch.save({
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)
        print("Model components saved.")

    def load(self, path: str):
        """
        Loads the state dictionaries for the feature extractor, actor, and critic.

        Args:
            path: The file path from which to load the model weights.
        """
        try:
            print(f"Loading model components from {path}")
            checkpoint = torch.load(path, map_location=self.device)
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            print("Model components loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {path}")
        except KeyError as e:
            print(f"Error: Missing key {e} in checkpoint file {path}. Structure mismatch?")
        except Exception as e:
            print(f"Error loading model components from {path}: {e}")

def load_checkpoint(agent: PPOAgent, checkpoint_path: str, config: dict, device: str):
    """
    Returns:
        A tuple (best_mean_reward, global_step) loaded from the checkpoint,
        or (-np.inf, 0) if loading fails or the checkpoint doesn't exist.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Starting fresh training.")
        return -np.inf, 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loading checkpoint from {checkpoint_path}")

    # Load model weights
    agent.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    print("Model weights loaded successfully.")

    global_step = checkpoint.get('global_step', 0)
    best_mean_reward = checkpoint.get('mean_reward', -np.inf)
    print(f"Resuming from global step {global_step}")
    print(f"Best mean reward from checkpoint: {best_mean_reward:.2f}")

    return best_mean_reward, global_step

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_env(env_id: str, seed: int, frame_stack: int, max_episode_steps: int, idx: int = 0):
    def _init():
        env_seed = seed + idx
        env = gym.make(env_id, continuous=True, domain_randomize=False, render_mode=None)
        env.reset(seed=env_seed)
        env.action_space.seed(env_seed)
        if config["use_reward_shaping"]:
            env = RewardShapingWrapper(env,
                                      velocity_weight=config["velocity_reward_weight"],
                                      survival_reward=config["survival_reward"],
                                      track_penalty=config["track_penalty"],
                                      steering_smooth_weight=config["steering_smooth_weight"],
                                      acceleration_while_turning_penalty_weight=config["acceleration_while_turning_penalty_weight"])

        env = GrayScaleObservation(env)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = FrameStack(env, frame_stack)
        return env
    return _init

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a PPO agent for CarRacing-v3")
    parser.add_argument("--checkpoint", type=str, default=None,  help="Path to a checkpoint file to resume training from.")
    parser.add_argument("--steps", type=int, default=None, help="Override the total number of training timesteps defined in the config.")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed defined in the config.")
    parser.add_argument("--log-dir", type=str, default=None, help="Override the TensorBoard log directory defined in the config.")
    args = parser.parse_args()

    config["checkpoint_path"] = args.checkpoint if args.checkpoint else config["checkpoint_path"]
    config["total_timesteps"] = args.steps if args.steps else config["total_timesteps"]
    config["seed"] = args.seed if args.seed else config["seed"]
    config["log_dir"] = args.log_dir if args.log_dir else config["log_dir"]

    print(f"Mixed Precision: {'Enabled' if config['mixed_precision'] else 'Disabled'}")
    print(f"Resuming from Checkpoint: {config['checkpoint_path'] if config['checkpoint_path'] else 'None'}")

    set_seeds(config["seed"])
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    # --- Environment Setup ---
    print(f"Creating {config['num_envs']} parallel environments...")
    env_fns = [make_env(config["env_id"], config["seed"], config["frame_stack"],
                        config["max_episode_steps"], i) for i in range(config["num_envs"])]

    # Choose between synchronous and asynchronous vectorized environments
    if config["async_envs"]:
        env = gymnasium.vector.AsyncVectorEnv(env_fns)
    else:
        env = gymnasium.vector.SyncVectorEnv(env_fns)

    print(f"Observation Space: {env.single_observation_space}")
    print(f"Action Space: {env.single_action_space}")

    # --- Agent Setup ---
    agent = PPOAgent(
        env.single_observation_space,
        env.single_action_space,
        config=config,  # Pass the entire config dictionary
        device=config["device"]
    )

    # --- Load Checkpoint ---
    best_mean_reward = -np.inf
    global_step = 0
    if config["checkpoint_path"]:
        loaded_reward, loaded_step = load_checkpoint(agent, config["checkpoint_path"], config, config["device"])
        if loaded_reward is not None: # Check if loading was successful
            best_mean_reward = loaded_reward
            global_step = loaded_step
            # Ensure the agent's internal step counter aligns for LR scheduling
            agent.steps_done = global_step
            print(f"Set agent's internal step counter to {agent.steps_done} for LR schedule.")

    # --- Rollout Buffer Setup ---
    # Calculate buffer size per environment
    buffer_size_per_env = config["buffer_size"] // config["num_envs"]
    if config["buffer_size"] % config["num_envs"] != 0:
         print(f"Warning: buffer_size ({config['buffer_size']}) not perfectly divisible by num_envs ({config['num_envs']}). Effective buffer size per env: {buffer_size_per_env}")

    buffer = RolloutBuffer(
        buffer_size_per_env, # Use size per env
        env.single_observation_space,
        env.single_action_space,
        num_envs=config["num_envs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        device=config["device"]
    )

    writer = SummaryWriter(log_dir=config["log_dir"])
    episode_rewards, episode_lengths = deque(maxlen=100), deque(maxlen=100)

    print(f"Starting training from step {global_step}/{config['total_timesteps']}")
    observations, infos = env.reset(seed=config["seed"]) # Initial reset with seed
    num_rollouts = 0
    # Track rewards and lengths for episodes currently in progress
    current_episode_rewards = np.zeros(config["num_envs"], dtype=np.float32)
    current_episode_lengths = np.zeros(config["num_envs"], dtype=np.int32)
    start_time = time.time()

    autocast_context = torch.cuda.amp.autocast() if config["device"] == "cuda" and config["mixed_precision"] else nullcontext()
    scaler = torch.cuda.amp.GradScaler() if config["device"] == "cuda" and config["mixed_precision"] else None

    while global_step < config["total_timesteps"]:
        rollout_episode_rewards = [] # Initialize list HERE
        buffer.reset() # Reset buffer position and full flag before each rollout
        rollout_start_time = time.time()
        steps_per_rollout = buffer.buffer_size # Steps to collect per environment in this rollout
        last_dones = np.zeros(config["num_envs"], dtype=bool) # Track dones from the final step

        # --- Rollout Phase ---
        for step in range(steps_per_rollout):
            # Ensure observations are tensors on the correct device
            obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=config["device"])

            # Agent selects actions based on observations
            with torch.no_grad():
                actions, values, log_probs = agent.act(obs_tensor)

            # Environment steps forward with the selected actions
            next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
            dones = terminateds | truncateds # Combine terminated and truncated flags

            # Update trackers for current episodes
            current_episode_rewards += rewards
            current_episode_lengths += 1

            # Store the transition in the rollout buffer
            buffer.add(observations, actions, rewards, terminateds, truncateds, values, log_probs)

            # Prepare for the next step
            observations = next_observations
            last_dones = dones # Store dones for GAE calculation

            # --- Handle Episode Completions ---
            # Check if any environments finished an episode using VecEnv's info dict
            if "_final_info" in infos:
                # Identify which environments finished
                finished_mask = infos["_final_info"]
                if np.any(finished_mask):
                    # Extract final info for completed episodes
                    final_infos = infos["final_info"][finished_mask]
                    env_indices = np.where(finished_mask)[0] # Get original indices

                    for i, final_info in enumerate(final_infos):
                        if final_info is not None and "episode" in final_info:
                            ep_rew = final_info["episode"]["r"]
                            ep_len = final_info["episode"]["l"]
                            episode_rewards.append(ep_rew) # Add to logging queue (100 ep avg)
                            episode_lengths.append(ep_len)
                            rollout_episode_rewards.append(ep_rew) # Append reward HERE
                            print(f"Env {env_indices[i]} finished: Reward={ep_rew:.2f}, Length={ep_len}, Total Steps={global_step+step*config['num_envs']}")

                            # Reset trackers for the specific environment that finished
                            current_episode_rewards[env_indices[i]] = 0
                            current_episode_lengths[env_indices[i]] = 0

            # Fallback if _final_info is not present (e.g., older Gym versions)
            elif np.any(dones):
                # Collect components for averaging across finished envs in this step
                velocity_rews, survival_rews, track_pens, steering_pens = [], [], [], []
                accel_turn_pens, steps_off, off_track_pcts = [], [], []

                for i in range(config["num_envs"]):
                    if dones[i]:
                        ep_reward = current_episode_rewards[i]
                        ep_length = current_episode_lengths[i]
                        episode_rewards.append(ep_reward)
                        episode_lengths.append(ep_length)
                        rollout_episode_rewards.append(ep_reward) # Append reward HERE too
                        # print(f"Env {i} finished (manual): Reward={ep_reward:.2f}, Length={ep_length}, Total Steps={global_step+step*config['num_envs']}")

                        # --- BEGIN MOVED LOGGING LOGIC ---
                        # Attempt to get detailed info from the info dict of the finished env
                        env_info = infos[i] if isinstance(infos, (list, tuple)) else infos.get(i) # Handle potential dict structure
                        if env_info:
                            if 'episode_velocity_rewards' in env_info: velocity_rews.append(env_info['episode_velocity_rewards'])
                            if 'episode_survival_rewards' in env_info: survival_rews.append(env_info['episode_survival_rewards'])
                            if 'episode_track_penalties' in env_info: track_pens.append(env_info['episode_track_penalties'])
                            if 'episode_steering_penalties' in env_info: steering_pens.append(env_info['episode_steering_penalties'])
                            if 'episode_acceleration_while_turning_penalties' in env_info: accel_turn_pens.append(env_info['episode_acceleration_while_turning_penalties'])
                            if 'steps_off_track' in env_info:
                                steps_off.append(env_info['steps_off_track'])
                                if ep_length > 0: # Use calculated ep_length
                                    off_track_pcts.append(100 * env_info['steps_off_track'] / ep_length)
                        # --- END MOVED LOGGING LOGIC ---

                        # Reset trackers
                        current_episode_rewards[i] = 0
                        current_episode_lengths[i] = 0

                # --- BEGIN MOVED TENSORBOARD LOGGING ---
                # Log averaged components if available (after checking all finished envs)
                # Note: This logging now happens *inside* the rollout loop whenever an episode ends,
                # rather than only at the end of the logging interval.
                # This might lead to more frequent but potentially noisier component logs.
                # Alternatively, accumulate these lists outside this loop and log them
                # during the main logging phase (num_rollouts % log_interval == 0).
                # For simplicity now, we log immediately.
                if velocity_rews: writer.add_scalar("rewards/mean_velocity", np.mean(velocity_rews), global_step)
                if survival_rews: writer.add_scalar("rewards/mean_survival", np.mean(survival_rews), global_step)
                if track_pens: writer.add_scalar("penalties/mean_track", np.mean(track_pens), global_step)
                if steering_pens: writer.add_scalar("penalties/mean_steering", np.mean(steering_pens), global_step)
                if accel_turn_pens: writer.add_scalar("penalties/mean_accel_turn", np.mean(accel_turn_pens), global_step)
                if steps_off: writer.add_scalar("driving/mean_steps_off_track", np.mean(steps_off), global_step)
                if off_track_pcts: writer.add_scalar("driving/mean_percent_off_track", np.mean(off_track_pcts), global_step)
                # --- END MOVED TENSORBOARD LOGGING ---

            # Update global step count (total steps across all envs)
            global_step += config["num_envs"]

            # Check if total timesteps limit is reached
            if global_step >= config["total_timesteps"]:
                print(f"Reached total timesteps ({config['total_timesteps']}). Finishing rollout.")
                break # Exit the inner rollout loop

        # --- Post-Rollout Phase ---
        # Compute advantages and returns after collecting the rollout data
        with torch.no_grad():
            # Get value estimate for the last observation in the rollout
            obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=config["device"])
            features = agent.feature_extractor(obs_tensor) # Pass raw obs, normalization happens inside
            last_values = agent.critic(features).cpu().numpy() # Get value estimates

        # Calculate GAE and returns using collected data and last value estimate
        buffer.compute_returns_and_advantages(last_values, last_dones)

        # --- Learning Phase ---
        # Update agent policy and value function using the collected rollout data
        if config["mixed_precision"] and config["device"] == "cuda":
            metrics = agent.learn_mixed_precision(buffer, scaler) # Use mixed precision update
        else:
            metrics = agent.learn(buffer) # Use standard precision update

        # Update learning rate based on schedule
        current_lr = agent.update_learning_rate(config['total_timesteps'])

        # Increment rollout counter
        num_rollouts += 1

        # --- Logging ---
        if num_rollouts % config["log_interval"] == 0 and len(episode_rewards) > 0:
            # --- Calculate performance metrics ---
            mean_reward_100 = np.mean(episode_rewards)
            mean_length_100 = np.mean(episode_lengths)
            rollout_duration = time.time() - rollout_start_time
            steps_in_rollout = buffer.size() # Get actual number of steps collected
            fps = int(steps_in_rollout / rollout_duration) if rollout_duration > 0 else 0

            # --- Calculate Mean Rollout Reward using the accumulated list ---
            mean_rollout_reward = np.mean(rollout_episode_rewards) if rollout_episode_rewards else -1 # Use -1 if no episodes finished in interval

            # --- Print summary to console ---
            rollout_reward_str = f"{mean_rollout_reward:.2f}({len(rollout_episode_rewards)})" if mean_rollout_reward != -1 else "N/A"
            print(f"Rollout {num_rollouts:3d} | Step {global_step:7d}/{config['total_timesteps']:7d} | "
                    f"Reward100: {mean_reward_100:6.2f} | RewardRoll: {rollout_reward_str:>10s} | "
                    f"Length: {mean_length_100:5.1f} | FPS: {fps:4d} | "
                    f"LR: {current_lr:.2e} | PiLoss: {metrics['policy_loss']:6.4f} | "
                    f"VLoss: {metrics['value_loss']:6.4f} | Ent: {metrics['entropy_loss']:6.4f} | "
                    f"KL: {metrics['approx_kl']:6.4f} | Clip: {metrics['clip_fraction']:6.4f}")

            # --- Log metrics to TensorBoard ---
            writer.add_scalar("ppo/mean_reward_100", mean_reward_100, global_step)
            writer.add_scalar("ppo/mean_length_100", mean_length_100, global_step)
            if mean_rollout_reward != -1:
                writer.add_scalar("ppo/mean_rollout_reward", mean_rollout_reward, global_step)
            writer.add_scalar("ppo/fps", fps, global_step)
            writer.add_scalar("ppo/learning_rate", current_lr, global_step)
            writer.add_scalar("ppo/policy_loss", metrics["policy_loss"], global_step)
            writer.add_scalar("ppo/value_loss", metrics["value_loss"], global_step)
            writer.add_scalar("ppo/entropy", metrics["entropy_loss"], global_step)
            writer.add_scalar("ppo/approx_kl", metrics["approx_kl"], global_step)
            writer.add_scalar("ppo/clip_fraction", metrics["clip_fraction"], global_step)

            # --- Save Best Model ---
            if mean_reward_100 > best_mean_reward and num_rollouts > 500:
                best_mean_reward = mean_reward_100
                best_model_path = os.path.join(config["save_dir"], "best_model.pth")
                print(f"New best mean reward: {best_mean_reward:.2f}. Saving model to {best_model_path}")
                torch.save({
                    'feature_extractor_state_dict': agent.feature_extractor.state_dict(),
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'global_step': global_step,
                    'mean_reward': mean_reward_100, # Save the reward that triggered the save
                    'config': config # Optionally save the config used
                }, best_model_path)

        # --- Save Checkpoint Periodically ---
        if num_rollouts > 0 and num_rollouts % config["save_interval"] == 0:
            checkpoint_path = os.path.join(config["save_dir"], f"checkpoint_{global_step}.pth")
            print(f"Saving checkpoint at step {global_step} to {checkpoint_path}")
            torch.save({
                'feature_extractor_state_dict': agent.feature_extractor.state_dict(),
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                # Save optimizer states if needed for exact resumption
                # 'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                # 'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                'global_step': global_step,
                'config': config,
                'mean_reward': best_mean_reward, # Save current best reward
            }, checkpoint_path)

        # Check again if total timesteps reached after learning phase
        if global_step >= config["total_timesteps"]:
            print("Total timesteps reached. Exiting training loop.")
            break

    env.close()
    writer.close()
