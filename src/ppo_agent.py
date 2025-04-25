import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import itertools

from .cnn_model import CNNFeatureExtractor
from gymnasium import spaces
from typing import Generator, Tuple, Optional

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
        self.fixed_std = fixed_std

        hidden_dim = 256
        self.fc1 = nn.Linear(features_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        
        if not fixed_std:
            self.fc_logstd = nn.Linear(hidden_dim, action_dim)
            self.fc_logstd.weight.data.fill_(0.0)
            self.fc_logstd.bias.data.fill_(np.log(self.initial_action_std))
        else:
            self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(self.initial_action_std), requires_grad=False)
        
        nn.init.orthogonal_(self.fc_mean.weight, gain=0.01)
        nn.init.constant_(self.fc_mean.bias, 0.0)

    def forward(self, features: torch.Tensor):
        """
        Forward pass to get action distribution parameters.

        :param features: Feature tensor from CNN (Batch, features_dim)
        :return: Action means, Action log standard deviations
        """
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))

        mean = torch.tanh(self.fc_mean(x))

        if not self.fixed_std:
            log_std = self.fc_logstd(x)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        else:
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
        std = log_std.exp()
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
        log_prob = action_dist.log_prob(actions).sum(axis=-1)
        entropy = action_dist.entropy().sum(axis=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """
    Critic Network for PPO.
    Takes features from CNN and outputs the estimated state value.
    """
    def __init__(self, features_dim: int):
        super().__init__()
        hidden_dim = 256
        self.fc1 = nn.Linear(features_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        
        nn.init.uniform_(self.fc_value.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_value.bias, -3e-3, 3e-3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        value = self.fc_value(x)
        return value.squeeze(-1)


class PPOAgent:
    """
    PPO Agent implementation combining CNN feature extractor, Actor, and Critic.
    """
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Box,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.1,
                 epochs: int = 5,
                 batch_size: int = 64,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 features_dim: int = 64,
                 target_kl: Optional[float] = 0.02,
                 initial_action_std: float = 1.0,
                 weight_decay: float = 1e-5,
                 fixed_std: bool = False,
                 lr_warmup_steps: int = 5000,
                 device: str = 'cpu'):

        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self.initial_lr = lr
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.initial_action_std = initial_action_std
        self.weight_decay = weight_decay
        self.fixed_std = fixed_std
        self.lr_warmup_steps = lr_warmup_steps
        self.steps_done = 0
        self.device = torch.device(device)
        
        # Feature Extractor (CNN)
        self.feature_extractor = CNNFeatureExtractor(observation_space, features_dim).to(self.device)

        # Actor Network
        self.actor = Actor(features_dim, self.action_dim, 
                          initial_action_std=self.initial_action_std,
                          fixed_std=self.fixed_std).to(self.device)

        # Critic Network
        self.critic = Critic(features_dim).to(self.device)

        # Optimizers
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

    def act(self, observation: torch.Tensor):
        """
        Select an action based on the current observation, also return value estimate.

        :param observation: Current environment observation (Batch, k, H, W) as tensor
        :return: action (np.ndarray), value (np.ndarray), log_prob (np.ndarray)
        """
        self.feature_extractor.eval()
        self.actor.eval()
        self.critic.eval()

        # If observation is already a tensor on the right device, skip conversion
        if not isinstance(observation, torch.Tensor):
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32).to(self.device)
        else:
            observation_tensor = observation

        with torch.no_grad():
            features = self.feature_extractor(observation_tensor)
            value = self.critic(features)
            action_dist = self.actor.get_action_dist(features)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(axis=-1)

        action_np = action.detach().cpu().numpy()
        value_np = value.detach().cpu().numpy()
        log_prob_np = log_prob.detach().cpu().numpy()

        return action_np, value_np, log_prob_np
        
    def update_learning_rate(self, total_timesteps):
        """Update learning rate with warmup and decay"""
        self.steps_done += 1
        
        # Warmup phase: linearly increase learning rate
        if self.steps_done < self.lr_warmup_steps:
            # Gradual warmup from 30% to 100% of initial_lr
            alpha = self.steps_done / self.lr_warmup_steps
            current_lr = self.initial_lr * (0.3 + 0.7 * alpha)
        else:
            # Cosine annealing schedule
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

    def learn_mixed_precision(self, rollout_buffer, scaler):
        """
        Update the agent's networks using mixed precision for faster training on GPUs.
        
        :param rollout_buffer: Buffer containing collected experiences
        :param scaler: GradScaler for mixed precision training
        :return: Dictionary of training metrics
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
            epoch_kl_divs = []
            
            # Iterate over batches from the rollout buffer
            for batch in rollout_buffer.get_batches(self.batch_size):
                obs_batch, actions_batch, old_log_probs_batch, advantages_batch, returns_batch = batch

                with torch.cuda.amp.autocast():
                    # Normalize observations before feature extraction - REMOVED, handled in CNNModel
                    # obs_batch_norm = obs_batch.float() / 255.0 
                    features = self.feature_extractor(obs_batch) # Use raw obs_batch
                    values = self.critic(features)
                    log_probs, entropy = self.actor.evaluate_actions(features, actions_batch)

                # Normalize advantages - REMOVED, handled globally in RolloutBuffer
                # advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

                # Calculate Actor (Policy) Loss
                ratio = torch.exp(log_probs - old_log_probs_batch)
                ratio = torch.clamp(ratio, 0.0, 5.0)
                
                policy_loss_1 = advantages_batch * ratio
                policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Calculate Critic (Value) Loss
                value_loss = F.mse_loss(values, returns_batch)

                # Calculate Entropy Bonus
                entropy_loss = -torch.mean(entropy)

                # Combine Losses
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization Steps with mixed precision
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                # Use scaler for gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping with the scaler
                scaler.unscale_(self.actor_optimizer)
                scaler.unscale_(self.critic_optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic.parameters(), self.feature_extractor.parameters()),
                    self.max_grad_norm
                )

                # Apply gradients with the scaler
                scaler.step(self.actor_optimizer)
                scaler.step(self.critic_optimizer)
                scaler.update()

                # Calculate additional metrics for logging (outside of autocast)
                with torch.no_grad():
                    approx_kl = 0.25 * torch.mean((log_probs - old_log_probs_batch)**2).cpu().numpy()
                    clip_fraction.append(torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float()).cpu().numpy())

                # Store batch metrics
                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(entropy_loss.item())
                all_kl_divs.append(approx_kl)
                epoch_kl_divs.append(approx_kl)

            # KL Divergence Check (monitoring only)
            epoch_mean_kl = np.mean(epoch_kl_divs)
            if epoch_mean_kl > self.target_kl and epoch == 0:
                print(f"KL divergence high at epoch {epoch+1}: {epoch_mean_kl:.4f} > {self.target_kl:.4f}, but continuing training")

        # Calculate average metrics over all batches and epochs processed
        avg_policy_loss = np.mean(all_policy_losses)
        avg_value_loss = np.mean(all_value_losses)
        avg_entropy_loss = np.mean(all_entropy_losses)
        avg_approx_kl = np.mean(all_kl_divs)
        avg_clip_fraction = np.mean(clip_fraction)

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy_loss": avg_entropy_loss,
            "approx_kl": avg_approx_kl,
            "clip_fraction": avg_clip_fraction,
        }

    def learn(self, rollout_buffer):
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
            epoch_kl_divs = []
            # Iterate over batches from the rollout buffer
            for batch in rollout_buffer.get_batches(self.batch_size):
                obs_batch, actions_batch, old_log_probs_batch, advantages_batch, returns_batch = batch

                # Normalize observations before feature extraction - REMOVED, handled in CNNModel
                # obs_batch_norm = obs_batch.float() / 255.0 
                features = self.feature_extractor(obs_batch) # Use raw obs_batch
                values = self.critic(features)
                log_probs, entropy = self.actor.evaluate_actions(features, actions_batch)

                # Normalize advantages - REMOVED, handled globally in RolloutBuffer
                # advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

                # Calculate Actor (Policy) Loss
                ratio = torch.exp(log_probs - old_log_probs_batch)
                
                ratio = torch.clamp(ratio, 0.0, 5.0)
                
                policy_loss_1 = advantages_batch * ratio
                policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Calculate Critic (Value) Loss
                value_loss = F.mse_loss(values, returns_batch)

                # Calculate Entropy Bonus
                entropy_loss = -torch.mean(entropy)

                # Combine Losses
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization Steps
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

                # Calculate additional metrics for logging
                with torch.no_grad():
                    approx_kl = 0.25 * torch.mean((log_probs - old_log_probs_batch)**2).cpu().numpy()
                    clip_fraction.append(torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float()).cpu().numpy())

                # Store batch metrics
                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(entropy_loss.item())
                all_kl_divs.append(approx_kl)
                epoch_kl_divs.append(approx_kl)

            # KL Divergence Check (monitoring only)
            epoch_mean_kl = np.mean(epoch_kl_divs)
            if epoch_mean_kl > self.target_kl and epoch == 0:
                print(f"KL divergence high at epoch {epoch+1}: {epoch_mean_kl:.4f} > {self.target_kl:.4f}, but continuing training")

        # Calculate average metrics over all batches and epochs processed
        avg_policy_loss = np.mean(all_policy_losses)
        avg_value_loss = np.mean(all_value_losses)
        avg_entropy_loss = np.mean(all_entropy_losses)
        avg_approx_kl = np.mean(all_kl_divs)
        avg_clip_fraction = np.mean(clip_fraction)

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy_loss": avg_entropy_loss,
            "approx_kl": avg_approx_kl,
            "clip_fraction": avg_clip_fraction,
        }

    def save(self, path: str):
        torch.save({
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path) 