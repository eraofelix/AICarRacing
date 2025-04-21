import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Added for loss functions
from torch.distributions import Normal
import numpy as np
import itertools # For chaining parameters

# Assuming cnn_model.py and env_wrappers.py are in the same directory or accessible
from .cnn_model import CNNFeatureExtractor
from gymnasium import spaces # For type hinting
# We'll need a RolloutBuffer eventually, define a placeholder for type hinting
from typing import Generator, Tuple, Optional

class RolloutBuffer:
    # --- Placeholder --- #
    # This class will be implemented separately
    # It needs to store (observations, actions, rewards, dones, log_probs, values)
    # and compute returns and advantages (likely using GAE)
    # It must provide a method like get_batches or __iter__
    def get_batches(self, batch_size: int, device: torch.device) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        # Placeholder: This should yield batches of tensors on the correct device
        # (observations, actions, old_log_probs, advantages, returns)
        raise NotImplementedError
    # --- End Placeholder --- #

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    """
    Actor Network for PPO (Continuous Actions).
    Takes features from CNN and outputs parameters for action distribution.
    """
    def __init__(self, features_dim: int, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

        # Shared layers (optional, could directly connect features_dim to output)
        # Let's add one hidden layer for flexibility
        hidden_dim = 256
        self.fc1 = nn.Linear(features_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)

    def forward(self, features: torch.Tensor):
        """
        Forward pass to get action distribution parameters.

        :param features: Feature tensor from CNN (Batch, features_dim)
        :return: Action means, Action log standard deviations
        """
        x = torch.relu(self.fc1(features))

        # Output mean for the action distribution
        # Use tanh to bound actions (e.g., steering between -1 and 1)
        # Adjust scaling if needed based on env action space specifics
        mean = torch.tanh(self.fc_mean(x))

        # Output log standard deviation
        log_std = self.fc_logstd(x)
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def get_action_dist(self, features: torch.Tensor):
        """
        Get the action distribution for given features.

        :param features: Feature tensor from CNN (Batch, features_dim)
        :return: Normal distribution object
        """
        mean, log_std = self.forward(features)
        std = log_std.exp() # Standard deviation
        return Normal(mean, std)

    def evaluate_actions(self, features: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the features and actions.

        :param features: Features tensor (Batch, features_dim)
        :param actions: Actions tensor (Batch, action_dim)
        :return: log probability of the actions, entropy of the distribution
        """
        action_dist = self.get_action_dist(features)
        log_prob = action_dist.log_prob(actions).sum(axis=-1) # Sum across action dimensions
        entropy = action_dist.entropy().sum(axis=-1) # Sum across action dimensions
        return log_prob, entropy


class Critic(nn.Module):
    """
    Critic Network for PPO.
    Takes features from CNN and outputs the estimated state value.
    """
    def __init__(self, features_dim: int):
        super().__init__()

        # Shared layers (optional)
        hidden_dim = 256
        self.fc1 = nn.Linear(features_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1) # Output a single value

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to estimate state value.

        :param features: Feature tensor from CNN (Batch, features_dim)
        :return: Estimated state value (Batch, 1)
        """
        x = torch.relu(self.fc1(features))
        value = self.fc_value(x)
        return value.squeeze(-1) # Return shape (Batch,) for easier loss calculation


class PPOAgent:
    """
    PPO Agent implementation combining CNN feature extractor, Actor, and Critic.
    """
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Box,
                 lr: float = 3e-4, # Learning rate for Actor and Critic/CNN
                 gamma: float = 0.99, # Discount factor
                 gae_lambda: float = 0.95, # Factor for GAE
                 clip_epsilon: float = 0.2, # PPO clipping parameter
                 epochs: int = 10, # Number of optimization epochs per rollout
                 batch_size: int = 64, # Minibatch size for optimization
                 vf_coef: float = 0.5, # Value function loss coefficient
                 ent_coef: float = 0.01, # Entropy bonus coefficient
                 max_grad_norm: float = 0.5, # Max gradient norm for clipping
                 features_dim: int = 256, # Dimension after CNN
                 target_kl: Optional[float] = None, # Target KL divergence for early stopping
                 device: str = 'cpu'):

        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda # Store for potential use in buffer
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = torch.device(device)

        # Feature Extractor (CNN)
        self.feature_extractor = CNNFeatureExtractor(observation_space, features_dim).to(self.device)

        # Actor Network
        self.actor = Actor(features_dim, self.action_dim).to(self.device)

        # Critic Network
        self.critic = Critic(features_dim).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(
            itertools.chain(self.critic.parameters(), self.feature_extractor.parameters()),
            lr=lr, eps=1e-5 # Add eps for stability like SB3
        )

        print(f"PPO Agent initialized on device: {self.device}")
        print(f"Feature Extractor: {sum(p.numel() for p in self.feature_extractor.parameters())} parameters")
        print(f"Actor: {sum(p.numel() for p in self.actor.parameters())} parameters")
        print(f"Critic: {sum(p.numel() for p in self.critic.parameters())} parameters")

    def act(self, observation: np.ndarray):
        """
        Select an action based on the current observation, also return value estimate.

        :param observation: Current environment observation (k, H, W)
        :return: action (np.ndarray), value (np.ndarray), log_prob (np.ndarray)
        """
        self.feature_extractor.eval()
        self.actor.eval()
        self.critic.eval() # Critic also needed for value estimate during rollout

        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.feature_extractor(observation_tensor)
            value = self.critic(features) # Get value estimate
            action_dist = self.actor.get_action_dist(features)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(axis=-1)

        action_np = action.detach().cpu().numpy().squeeze(0)
        value_np = value.detach().cpu().numpy().squeeze(0)
        log_prob_np = log_prob.detach().cpu().numpy().squeeze(0)

        return action_np, value_np, log_prob_np

    def learn(self, rollout_buffer: RolloutBuffer):
        """
        Update the agent's networks using data from the rollout buffer.
        Returns a dictionary of training metrics.
        """
        self.feature_extractor.train()
        self.actor.train()
        self.critic.train()

        # Store metrics across epochs and batches
        all_policy_losses = []
        all_value_losses = []
        all_entropy_losses = []
        all_kl_divs = []
        clip_fraction = []

        # Loop for optimization epochs
        for epoch in range(self.epochs):
            epoch_kl_divs = [] # Track KL within this epoch for early stopping
            # Iterate over batches from the rollout buffer
            for batch in rollout_buffer.get_batches(self.batch_size):
                obs_batch, actions_batch, old_log_probs_batch, advantages_batch, returns_batch = batch

                # --- Forward pass for current policy --- #
                features = self.feature_extractor(obs_batch)
                values = self.critic(features) # Shape: (Batch,)
                log_probs, entropy = self.actor.evaluate_actions(features, actions_batch)
                # --------------------------------------- #

                # Normalize advantages (often improves stability)
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

                # --- Calculate Actor (Policy) Loss --- #
                ratio = torch.exp(log_probs - old_log_probs_batch)
                policy_loss_1 = advantages_batch * ratio
                policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                # --------------------------------------- #

                # --- Calculate Critic (Value) Loss --- #
                value_loss = F.mse_loss(values, returns_batch)
                # --------------------------------------- #

                # --- Calculate Entropy Bonus --- #
                entropy_loss = -torch.mean(entropy)
                # ------------------------------- #

                # --- Combine Losses --- #
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                # ---------------------- #

                # --- Optimization Steps --- #
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic.parameters(), self.feature_extractor.parameters()),
                    self.max_grad_norm
                )

                # Apply gradients
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                # -------------------------- #

                # --- Calculate additional metrics for logging --- #
                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs_batch
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    clip_fraction.append(torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float()).cpu().numpy())

                # Store batch metrics
                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(entropy_loss.item())
                all_kl_divs.append(approx_kl)
                epoch_kl_divs.append(approx_kl) # Track KL for early stopping within this epoch
                # -------------------------------------------- #

            # --- KL Divergence Check for Early Stopping (per epoch) --- #
            epoch_mean_kl = np.mean(epoch_kl_divs)
            if self.target_kl is not None and epoch_mean_kl > 1.5 * self.target_kl: # SB3 uses 1.5 * target_kl
                print(f"Early stopping at epoch {epoch+1}/{self.epochs} due to reaching max KL divergence: {epoch_mean_kl:.4f} > {self.target_kl:.4f}")
                break # Stop training epochs for this rollout
            # ------------------------------------------------------------ #

        # Calculate average metrics over all batches and epochs processed
        avg_policy_loss = np.mean(all_policy_losses)
        avg_value_loss = np.mean(all_value_losses)
        avg_entropy_loss = np.mean(all_entropy_losses)
        avg_approx_kl = np.mean(all_kl_divs)
        # Need to compute clip fraction avg separately if early stopping happened
        avg_clip_fraction = np.mean(clip_fraction) # Use last batch clip frac or avg across batches? Let's average.

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy_loss": avg_entropy_loss,
            "approx_kl": avg_approx_kl,
            "clip_fraction": avg_clip_fraction,
        }

    def save(self, path: str):
        pass # Placeholder for saving model state

# Example Usage (minimal change, learn now has structure)
if __name__ == '__main__':
    # ... (previous setup code remains the same) ...
    import gymnasium as gym
    # Need env_wrappers for this test
    try:
        from .env_wrappers import GrayScaleObservation, FrameStack
    except ImportError:
        print("env_wrappers.py not found, skipping agent test.")
        exit()

    k = 4
    dummy_env = gym.make("CarRacing-v3", continuous=True, domain_randomize=False)
    dummy_env = GrayScaleObservation(dummy_env)
    dummy_env = FrameStack(dummy_env, k)

    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space

    print(f"Observation Space: {obs_space}")
    print(f"Action Space: {act_space}")

    agent = PPOAgent(observation_space=obs_space,
                     action_space=act_space,
                     device='cuda' if torch.cuda.is_available() else 'cpu')

    obs, _ = dummy_env.reset()
    print(f"Sample observation shape: {obs.shape}")

    action, value, log_prob = agent.act(obs)
    print(f"Selected action: {action}")
    print(f"Estimated value: {value}")
    print(f"Log probability: {log_prob}")

    # Cannot test learn without a real RolloutBuffer
    print("\nSkipping learn test without RolloutBuffer implementation.")
    # agent.learn(None) # Pass None for now - WILL FAIL without buffer

    dummy_env.close()
    print("\nPPOAgent structure with learn logic seems functional (requires RolloutBuffer).")

# Clean up imports if not used in main agent logic
# del Actor, Critic, LOG_STD_MAX, LOG_STD_MIN, Normal, np, itertools

    # Example Usage (for testing)
    if __name__ == '__main__':
        # Example parameters
        batch_size = 5
        num_features = 256 # Output dim from CNNFeatureExtractor
        num_actions = 3      # For CarRacing: Steering, Gas, Brake

        # Dummy feature input
        dummy_features = torch.randn(batch_size, num_features)

        # Test Actor
        actor = Actor(num_features, num_actions)
        print("Actor Network:")
        print(actor)
        mean, log_std = actor(dummy_features)
        print(f"Actor output shapes: mean={mean.shape}, log_std={log_std.shape}")

        action_dist = actor.get_action_dist(dummy_features)
        print(f"Action distribution type: {type(action_dist)}")
        sampled_actions = action_dist.sample()
        log_probs = action_dist.log_prob(sampled_actions).sum(axis=-1) # Sum log_prob across action dimensions
        print(f"Sampled actions shape: {sampled_actions.shape}")
        print(f"Log probabilities shape: {log_probs.shape}")

        # Test Critic
        critic = Critic(num_features)
        print("\nCritic Network:")
        print(critic)
        value = critic(dummy_features)
        print(f"Critic output shape: {value.shape}")

        assert value.shape == (batch_size, 1), "Critic output shape mismatch!"
        print("\nActor and Critic networks seem functional.") 