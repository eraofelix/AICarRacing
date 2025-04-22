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

class RunningMeanStd:
    """Normalizes observations using running statistics"""
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)

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
    def __init__(self, features_dim: int, action_dim: int, initial_action_std=1.0, fixed_std=False):
        super().__init__()
        self.action_dim = action_dim
        self.initial_action_std = initial_action_std
        self.fixed_std = fixed_std  # Whether to use fixed std or learned

        # SIMPLIFIED: Single hidden layer with fewer neurons
        hidden_dim = 128  # Reduced from 256
        self.fc1 = nn.Linear(features_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        
        if not fixed_std:
            self.fc_logstd = nn.Linear(hidden_dim, action_dim)
            # Initialize log_std with higher values for more exploration
            self.fc_logstd.weight.data.fill_(0.0)  # Initialize to zeros
            self.fc_logstd.bias.data.fill_(np.log(self.initial_action_std))  # Set bias to log of initial std
        else:
            # Fixed log_std parameter (not a network output)
            self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(self.initial_action_std), requires_grad=False)
        
        # Initialize mean layer to produce near-zero outputs initially
        nn.init.uniform_(self.fc_mean.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_mean.bias, -3e-3, 3e-3)

    def forward(self, features: torch.Tensor):
        """
        Forward pass to get action distribution parameters.

        :param features: Feature tensor from CNN (Batch, features_dim)
        :return: Action means, Action log standard deviations
        """
        x = torch.relu(self.fc1(features))

        # Output mean for the action distribution
        # Use tanh to bound actions (e.g., steering between -1 and 1)
        mean = torch.tanh(self.fc_mean(x))

        # Get log_std - either fixed or from network
        if not self.fixed_std:
            log_std = self.fc_logstd(x)
            # Clamp log_std for stability
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        else:
            # Expand fixed log_std to match batch size
            batch_size = mean.size(0)
            log_std = self.log_std.expand(batch_size, -1)
            
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

        # SIMPLIFIED: Single hidden layer with fewer neurons
        hidden_dim = 128  # Reduced from 256
        self.fc1 = nn.Linear(features_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1) # Output a single value
        
        # Initialize value layer to produce near-zero outputs initially
        nn.init.uniform_(self.fc_value.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_value.bias, -3e-3, 3e-3)

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
                 lr: float = 1e-4, # Reduced from 3e-4 to 1e-4
                 gamma: float = 0.99, # Discount factor
                 gae_lambda: float = 0.95, # Factor for GAE
                 clip_epsilon: float = 0.1, # Reduced from 0.2 to 0.1
                 epochs: int = 5, # Reduced from 10 to 5
                 batch_size: int = 64, # Minibatch size for optimization
                 vf_coef: float = 0.5, # Value function loss coefficient
                 ent_coef: float = 0.01, # Entropy bonus coefficient
                 max_grad_norm: float = 0.5, # Max gradient norm for clipping
                 features_dim: int = 64, # Dimension after CNN (REDUCED from 128 to 64)
                 target_kl: Optional[float] = 0.02, # Increased target KL from None to 0.02
                 initial_action_std: float = 1.0, # Increased from 0.5 to 1.0
                 use_obs_norm: bool = False, # Changed from True to False since CNN normalizes
                 weight_decay: float = 1e-5, # Reduced from 1e-4 to 1e-5
                 fixed_std: bool = False, # Changed from True to False
                 lr_warmup_steps: int = 5000, # Number of warmup steps for learning rate
                 device: str = 'cpu'):

        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self.initial_lr = lr # Store initial learning rate
        self.lr = lr # Current learning rate (will be updated by schedule)
        self.gamma = gamma
        self.gae_lambda = gae_lambda # Store for potential use in buffer
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.initial_action_std = initial_action_std
        self.use_obs_norm = use_obs_norm
        self.weight_decay = weight_decay
        self.fixed_std = fixed_std
        self.lr_warmup_steps = lr_warmup_steps
        self.steps_done = 0  # Track total steps for warmup
        self.device = torch.device(device)
        
        # Observation normalization
        if self.use_obs_norm:
            self.obs_rms = RunningMeanStd(shape=observation_space.shape)
            print("Using observation normalization")

        # Feature Extractor (CNN)
        self.feature_extractor = CNNFeatureExtractor(observation_space, features_dim).to(self.device)

        # Actor Network
        self.actor = Actor(features_dim, self.action_dim, 
                          initial_action_std=self.initial_action_std,
                          fixed_std=self.fixed_std).to(self.device)

        # Critic Network
        self.critic = Critic(features_dim).to(self.device)

        # Optimizers - Added weight decay
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5, weight_decay=self.weight_decay)
        self.critic_optimizer = optim.Adam(
            itertools.chain(self.critic.parameters(), self.feature_extractor.parameters()),
            lr=self.lr, eps=1e-5, weight_decay=self.weight_decay
        )

        print(f"PPO Agent initialized on device: {self.device}")
        print(f"Feature Extractor: {sum(p.numel() for p in self.feature_extractor.parameters())} parameters")
        print(f"Actor: {sum(p.numel() for p in self.actor.parameters())} parameters")
        print(f"Critic: {sum(p.numel() for p in self.critic.parameters())} parameters")
        if self.fixed_std:
            print(f"Using fixed action std: {self.initial_action_std}")
        else:
            print(f"Using learned action std (initial: {self.initial_action_std})")

    def act(self, observation: np.ndarray):
        """
        Select an action based on the current observation, also return value estimate.

        :param observation: Current environment observation (k, H, W)
        :return: action (np.ndarray), value (np.ndarray), log_prob (np.ndarray)
        """
        self.feature_extractor.eval()
        self.actor.eval()
        self.critic.eval() # Critic also needed for value estimate during rollout

        # Normalize observations if enabled
        if self.use_obs_norm:
            # Update running mean/std statistics
            self.obs_rms.update(observation.copy())
            # Normalize the observation
            norm_obs = self.obs_rms.normalize(observation)
            observation_tensor = torch.as_tensor(norm_obs, dtype=torch.float32).to(self.device)
        else:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            features = self.feature_extractor(observation_tensor)
            value = self.critic(features) # Get value estimate
            action_dist = self.actor.get_action_dist(features)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(axis=-1)

        # Return batches
        action_np = action.detach().cpu().numpy()
        value_np = value.detach().cpu().numpy()
        log_prob_np = log_prob.detach().cpu().numpy()

        return action_np, value_np, log_prob_np
        
    def update_learning_rate(self, total_timesteps):
        """Update learning rate with warmup and decay"""
        self.steps_done += 1
        
        # Warmup phase: linearly increase learning rate
        if self.steps_done < self.lr_warmup_steps:
            # Gradual warmup from 30% to 100% of initial_lr (changed from 10% to 30%)
            alpha = self.steps_done / self.lr_warmup_steps
            current_lr = self.initial_lr * (0.3 + 0.7 * alpha)
        else:
            # Cosine annealing schedule (replaced linear decay)
            progress = min((self.steps_done - self.lr_warmup_steps) / 
                          (total_timesteps - self.lr_warmup_steps), 1.0)
            current_lr = self.initial_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
            
        # Minimum learning rate
        current_lr = max(current_lr, 1e-6)
        
        # Update optimizers
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = current_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = current_lr
            
        self.lr = current_lr
        return current_lr

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
                
                # Add ratio clipping for numerical stability (reduced upper bound from 10.0 to 5.0)
                ratio = torch.clamp(ratio, 0.0, 5.0)
                
                policy_loss_1 = advantages_batch * ratio
                policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                # --------------------------------------- #

                # --- Calculate Critic (Value) Loss --- #
                # Clippled value loss for stability (added)
                values_clipped = torch.clamp(
                    values,
                    returns_batch - self.clip_epsilon,
                    returns_batch + self.clip_epsilon
                )
                value_loss_original = F.mse_loss(values, returns_batch)
                value_loss_clipped = F.mse_loss(values_clipped, returns_batch)
                value_loss = torch.max(value_loss_original, value_loss_clipped)
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
                    # More conservative KL calculation
                    approx_kl = 0.5 * torch.mean((log_probs - old_log_probs_batch)**2).cpu().numpy()
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
        num_features = 64 # Reduced from 256 to 64
        num_actions = 3      # For CarRacing: Steering, Gas, Brake

        # Dummy feature input
        dummy_features = torch.randn(batch_size, num_features)

        # Test Actor
        actor = Actor(num_features, num_actions, initial_action_std=1.0, fixed_std=False)  # Updated params
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