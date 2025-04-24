import gymnasium as gym
import torch
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import gymnasium.vector # Added for vectorized environments

# Import custom modules
from src.env_wrappers import GrayScaleObservation, FrameStack, TimeLimit # Added TimeLimit import
from src.ppo_agent import PPOAgent, RunningMeanStd     # Also import RunningMeanStd
from src.rollout_buffer import RolloutBuffer             # Ensure 'src.' prefix

# --- Configuration --- #
config = {
    # Environment settings
    "env_id": "CarRacing-v3",
    "frame_stack": 4,
    "seed": 42, # For reproducibility
    "num_envs": 8, # Number of parallel environments
    "max_episode_steps": 700, # Longer episodes for fine-tuning
    "domain_randomize_intensity": 0.1, # Enable low-level domain randomization
    
    # Dynamic episode length config
    "dynamic_episode_length": True, # Enable/disable dynamic episode length adjustment
    "episode_length_thresholds": {  # Map of mean reward threshold -> new episode length
        100: 800,   # At mean reward 100, increase to 800 steps
        200: 900,   # At mean reward 200, increase to 900 steps
        300: 1000,  # At mean reward 300, increase to 1000 steps
    },
    "last_length_update_reward": 0,  # Track the last threshold that triggered an update

    # Training settings
    "total_timesteps": 10_000_000, # Fine-tuning steps
    "learning_rate": 5e-6, # Reduced LR for fine-tuning
    "buffer_size": 4096,  # Keep buffer size
    "batch_size": 64, # Keep batch size
    "ppo_epochs": 3, # Keep PPO epochs
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2, # Keep clip epsilon
    "vf_coef": 0.7, # Keep value function coefficient
    "ent_coef": 0.02, # Increased entropy coefficient for exploration
    "max_grad_norm": 0.5,
    "target_kl": 0.05, # Increased from 0.015 to allow more aggressive updates initially
    "features_dim": 256, # Keep features_dim

    # Staged exploration settings (for finetuning)
    "staged_exploration": True,
    "exploration_stages": [
        {"steps": 1000000, "action_noise": 0.05, "ent_coef": 0.03, "target_kl": 0.05, "outlier_factor": 1.2},
        {"steps": 3000000, "action_noise": 0.02, "ent_coef": 0.02, "target_kl": 0.02, "outlier_factor": 1.1},
        {"steps": 10000000, "action_noise": 0.01, "ent_coef": 0.01, "target_kl": 0.015, "outlier_factor": 1.0}
    ],
    
    # Outlier emphasis (more conservative for finetuning)
    "emphasize_outliers": True,
    "outlier_factor": 1.2, # More conservative than training
    "outlier_threshold": 1.5, # Standard deviations from mean
    
    # Observation normalization
    "use_obs_norm": True, # Enable observation normalization

    # Domain randomization progression
    "domain_randomize_schedule": True,  # Enable gradual increase in randomization
    "domain_randomize_steps": 5_000_000,  # Steps over which to increase to 0.5 (50% randomization)
    "domain_randomize_check_interval": 100_000,  # Check and update intensity every 100k steps
    "domain_randomize_max": 0.5,  # Maximum randomization level (50%)

    # Reward shaping options - REDUCED BY HALF for second stage
    "use_reward_shaping": True,  # Keep reward shaping enabled
    "velocity_reward_weight": 0.005,   # Reduced from 0.01
    "progress_reward_weight": 0.005,   # Reduced from 0.01
    "centerline_reward_weight": 0.01,  # Reduced from 0.02
    "survival_reward": 0.05,           # Reduced from 0.1

    # Logging and Saving
    "log_interval": 1, # Log stats every N rollouts
    "save_interval": 10, # Save model every N rollouts
    "save_dir": "./models/ppo_carracing_finetuned",
    "log_dir": "./logs/ppo_carracing_finetuned",
    "load_checkpoint_path": "./models/ppo_carracing/best_model.pth", # Load the best model

    # Hardware
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# --- Helper Functions --- #
def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_env(env_id, seed, frame_stack, max_episode_steps):
    """Helper function that returns a function to create and wrap a single environment instance."""
    def _init():
        # Derive a unique seed for this process to ensure envs are different
        process_seed = seed + int.from_bytes(os.urandom(4), 'little')
        set_seeds(process_seed) # Also set seeds for torch/numpy within the process
        # Use partial domain randomization based on current intensity
        random_intensity = config.get("domain_randomize_intensity", 0.0)
        # Only apply full randomization if intensity is 1.0, otherwise use custom wrapper
        domain_random = random_intensity >= 0.01  # Even tiny intensity triggers randomization
        env = gym.make(env_id, continuous=True, domain_randomize=domain_random)
        
        env.reset(seed=process_seed)
        env.action_space.seed(process_seed)
        env = GrayScaleObservation(env)
        
        # Add reward shaping if enabled
        if config.get("use_reward_shaping", False):
            # Wrap environment to add shaped rewards
            env = RewardShapingWrapper(env, 
                                        velocity_weight=config.get("velocity_reward_weight", 0.0),
                                        progress_weight=config.get("progress_reward_weight", 0.0),
                                        centerline_weight=config.get("centerline_reward_weight", 0.0),
                                        survival_reward=config.get("survival_reward", 0.0))
        
        env = TimeLimit(env, max_episode_steps=max_episode_steps)  # Add custom time limit
        env = FrameStack(env, frame_stack)
        # No need to reset again after FrameStack usually
        return env
    # Return the function itself, not the result of calling it
    return _init

# --- Reward Shaping Wrapper --- #
class RewardShapingWrapper(gym.Wrapper):
    """
    Wrapper to add shaped rewards to the CarRacing environment.
    Adds small bonuses for maintaining velocity and making progress.
    """
    def __init__(self, env, velocity_weight=0.01, progress_weight=0.01, centerline_weight=0.02, survival_reward=0.1):
        super().__init__(env)
        self.velocity_weight = velocity_weight
        self.progress_weight = progress_weight
        self.centerline_weight = centerline_weight
        self.survival_reward = survival_reward
        self.last_position = None
        self.last_progress = 0
        self.last_tile_distance = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_position = None
        self.last_progress = 0
        self.last_tile_distance = None
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get additional info if available
        speed = info.get('speed', 0.0) if isinstance(info, dict) else 0.0
        progress = info.get('progress', 0.0) if isinstance(info, dict) else 0.0
        # Get track position info if available
        tile_distance = info.get('tile_distance_to_center', 0.0) if isinstance(info, dict) else 0.0
        is_on_grass = info.get('is_on_grass', False) if isinstance(info, dict) else False
        
        # Add velocity reward component
        if speed > 0:
            # Encourages maintaining some velocity (avoid getting stuck)
            velocity_bonus = speed * self.velocity_weight
            reward += velocity_bonus
        
        # Add progress reward component if progress info is available
        if progress > self.last_progress:
            progress_bonus = (progress - self.last_progress) * self.progress_weight
            reward += progress_bonus
            self.last_progress = progress
            
        # Add centerline reward component
        if tile_distance is not None:
            # Reward for staying closer to the center of the track
            # Assume tile_distance ranges from 0 (center) to 1 (edge)
            centerline_bonus = (1.0 - tile_distance) * self.centerline_weight
            reward += centerline_bonus
            
            # Extra penalty for going off-track to discourage cutting corners
            if is_on_grass:
                reward -= 0.5  # Penalty for being on grass
                
        # Add survival bonus - small reward for each step to encourage not crashing
        reward += self.survival_reward
            
        return obs, reward, terminated, truncated, info

def emphasize_outlier_experiences(advantages, factor=1.2, threshold=1.5):
    """
    Amplify advantages for outlier experiences to emphasize exploration.
    
    Args:
        advantages: Advantage values from buffer
        factor: Multiplier for outlier advantages
        threshold: Number of standard deviations to consider an outlier
        
    Returns:
        Modified advantages and count of outliers
    """
    # Calculate standard deviation of advantages
    adv_mean = np.mean(advantages)
    adv_std = np.std(advantages)
    
    if adv_std < 1e-8:
        # Avoid division by near-zero
        return advantages, 0
    
    # Find outlier advantages (both positive and negative)
    large_adv_mask = np.abs(advantages - adv_mean) > threshold * adv_std
    
    # Count outliers for reporting
    outlier_count = np.sum(large_adv_mask)
    
    if outlier_count > 0:
        # Apply amplification factor to outliers
        advantages[large_adv_mask] *= factor
        print(f"Emphasized {outlier_count} outlier experiences by factor {factor:.2f}")
    
    return advantages, outlier_count

def add_action_noise(actions, noise_std=0.05):
    """
    Add Gaussian noise to actions to increase exploration.
    
    Args:
        actions: Original actions 
        noise_std: Standard deviation of the noise
        
    Returns:
        Actions with added noise, clipped to valid range
    """
    if noise_std <= 0.0:
        return actions
        
    # Add noise to actions
    noise = np.random.normal(0, noise_std, size=actions.shape)
    noisy_actions = actions + noise
    
    # Clip to valid range for CarRacing [-1, 1] for steering, [0, 1] for gas/brake
    # First dimension (steering)
    noisy_actions[:, 0] = np.clip(noisy_actions[:, 0], -1.0, 1.0)
    # Second and third dimensions (gas, brake)
    noisy_actions[:, 1:] = np.clip(noisy_actions[:, 1:], 0.0, 1.0)
    
    return noisy_actions

def update_exploration_parameters(config, global_step):
    """
    Update exploration parameters based on current training step.
    Industry-standard approach: staged exploration with different phases.
    
    Args:
        config: Configuration dictionary
        global_step: Current training step
        
    Returns:
        Updated config
    """
    if not config.get("staged_exploration", False):
        return config
        
    stages = config.get("exploration_stages", [])
    if not stages:
        return config
        
    # Find current stage based on global step
    current_stage = None
    for i, stage in enumerate(stages):
        if global_step < stage["steps"]:
            current_stage = stage
            break
    
    # If we're past all defined stages, use the last one
    if current_stage is None and stages:
        current_stage = stages[-1]
    
    if current_stage:
        # Update exploration parameters
        old_noise = config.get("action_noise_std", 0.0)
        old_ent = config.get("ent_coef", 0.0)
        old_kl = config.get("target_kl", 0.0)
        old_factor = config.get("outlier_factor", 0.0)
        
        config["action_noise_std"] = current_stage.get("action_noise", old_noise)
        config["ent_coef"] = current_stage.get("ent_coef", old_ent)
        config["target_kl"] = current_stage.get("target_kl", old_kl)
        config["outlier_factor"] = current_stage.get("outlier_factor", old_factor)
        
        # Log changes if they happened
        if (old_noise != config["action_noise_std"] or 
            old_ent != config["ent_coef"] or 
            old_kl != config["target_kl"] or
            old_factor != config["outlier_factor"]):
            print(f"\n==== EXPLORATION STAGE UPDATE ====")
            print(f"Updating exploration parameters for step {global_step}:")
            print(f"  Action noise: {old_noise:.3f} -> {config['action_noise_std']:.3f}")
            print(f"  Entropy coef: {old_ent:.3f} -> {config['ent_coef']:.3f}")
            print(f"  Target KL: {old_kl:.3f} -> {config['target_kl']:.3f}")
            print(f"  Outlier factor: {old_factor:.3f} -> {config['outlier_factor']:.3f}")
            print(f"===================================\n")
    
    return config

class ObservationNormalizer:
    """Normalize observations using running statistics."""
    def __init__(self, shape):
        self.running_mean = None
        self.running_var = None
        self.count = 0
        self.shape = shape
        self.epsilon = 1e-8
        
    def update(self, observations):
        """Update running statistics with new observations."""
        if self.running_mean is None:
            self.running_mean = np.zeros(self.shape, dtype=np.float32)
            self.running_var = np.ones(self.shape, dtype=np.float32)
            
        batch_size = observations.shape[0]
        batch_mean = np.mean(observations, axis=0)
        batch_var = np.var(observations, axis=0)
        
        # Update running statistics using Welford's algorithm
        new_count = self.count + batch_size
        delta = batch_mean - self.running_mean
        self.running_mean += delta * batch_size / new_count
        
        # Update variance
        self.running_var = (self.count * self.running_var + batch_size * batch_var +
                           delta**2 * self.count * batch_size / new_count) / new_count
                           
        self.count = new_count
        
    def normalize(self, observations):
        """Normalize observations using current statistics."""
        if self.running_mean is None:
            return observations
            
        return (observations - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

# --- Main Training Loop --- #
if __name__ == "__main__":
    print(f"Using device: {config['device']}")
    print("FINE-TUNING FROM CHECKPOINT:", config["load_checkpoint_path"])
    set_seeds(config["seed"]) # Set seeds early for main process

    # Create vectorized environment FIRST
    print(f"Creating {config['num_envs']} parallel environments...")
    env_fns = [make_env(config["env_id"], config["seed"] + i, config["frame_stack"], config["max_episode_steps"]) 
              for i in range(config["num_envs"])]
    env = gymnasium.vector.AsyncVectorEnv(env_fns)
    print(f"Observation Space: {env.single_observation_space}")
    print(f"Action Space: {env.single_action_space}")
    print(f"Using TimeLimit wrapper with {config['max_episode_steps']} max steps per episode")
    print(f"Domain randomization intensity: {config['domain_randomize_intensity']}")
    print(f"Reward shaping weights: velocity={config['velocity_reward_weight']}, "
          f"progress={config['progress_reward_weight']}, centerline={config['centerline_reward_weight']}, "
          f"survival={config['survival_reward']}")

    # Create Agent - Use attributes from the vector env
    agent = PPOAgent(env.single_observation_space,
                     env.single_action_space,
                     lr=config["learning_rate"], # Pass initial LR
                     gamma=config["gamma"],
                     gae_lambda=config["gae_lambda"],
                     clip_epsilon=config["clip_epsilon"],
                     epochs=config["ppo_epochs"],
                     batch_size=config["batch_size"],
                     vf_coef=config["vf_coef"],
                     ent_coef=config["ent_coef"],
                     max_grad_norm=config["max_grad_norm"],
                     features_dim=config["features_dim"],
                     target_kl=config["target_kl"],
                     device=config["device"])

    # Create reward normalizer
    reward_rms = RunningMeanStd(shape=())  # Create a fresh instance to forget old reward statistics
    print("Created reward normalizer for training stability")

    # Create observation normalizer if enabled
    if config.get("use_obs_norm", False):
        # Initialize with observation shape without batch dimension
        obs_normalizer = ObservationNormalizer(shape=env.single_observation_space.shape)
        print("Created observation normalizer for improved training stability")
    else:
        obs_normalizer = None
        print("Observation normalization disabled")

    # Create Rollout Buffer - Use attributes from the vector env
    # Pass buffer size per env
    buffer = RolloutBuffer(config["buffer_size"] // config["num_envs"],
                           env.single_observation_space,
                           env.single_action_space,
                           num_envs=config["num_envs"],
                           gamma=config["gamma"],
                           gae_lambda=config["gae_lambda"],
                           device=config["device"])

    # Logging setup
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    # Use deque for storing finished episode stats from all envs
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    writer = SummaryWriter(log_dir=config["log_dir"])
    best_mean_reward = -np.inf
    start_time = time.time()

    # --- Load Checkpoint --- #
    start_global_step = 0
    start_num_rollouts = 0
    if config["load_checkpoint_path"] and os.path.exists(config["load_checkpoint_path"]):
        print(f"Loading checkpoint from: {config['load_checkpoint_path']}")
        checkpoint = torch.load(config["load_checkpoint_path"], map_location=config["device"])

        # Try loading state dicts, handle potential errors (e.g., architecture mismatch)
        try:
            agent.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

            # Don't load the training step or rollout counter - start from 0 for this fine-tuning phase
            # But do load best mean reward for reference
            best_mean_reward = checkpoint.get('best_mean_reward', -np.inf)
            
            print(f"Starting fine-tuning with loaded best mean reward: {best_mean_reward:.2f}")
            
            # DO NOT freeze any CNN layers - allow the entire network to adapt to domain randomization
            print("All network layers are trainable for fine-tuning with domain randomization")

            # Apply entropy boost immediately for the fine-tuning phase
            config["normalize_rewards"] = True
            print("Enabled reward normalization for continued training")
            
            if "original_ent_coef" not in config:
                config["original_ent_coef"] = agent.ent_coef  # Store current value
                
            # Boost entropy coefficient
            agent.ent_coef = config["ent_coef"]  # Apply higher entropy directly
            print(f"Set entropy coefficient to {agent.ent_coef:.4f} for exploration during fine-tuning")
        except Exception as e:
            print(f"Error loading checkpoint state_dicts: {e}. Cannot continue fine-tuning.")
            exit(1)
    else:
        print("No checkpoint found or specified. Fine-tuning requires a pre-trained model.")
        exit(1)

    # Initialize environment state - use the vector env reset
    observations, infos = env.reset(seed=config["seed"])
    global_step = start_global_step
    num_rollouts = start_num_rollouts
    # Track rewards and lengths for each active episode in parallel envs
    current_episode_rewards_vec = np.zeros(config["num_envs"], dtype=np.float32)
    current_episode_lengths_vec = np.zeros(config["num_envs"], dtype=np.int32)

    print("Starting fine-tuning...")
    while global_step < config["total_timesteps"]:
        # --- Collect Rollout --- #
        buffer.pos = 0
        buffer.full = False
        rollout_start_time = time.time()

        # Steps per env for this rollout
        steps_per_rollout_per_env = config["buffer_size"] // config["num_envs"]

        # Keep track of last step dones for GAE calculation
        last_terminateds = np.zeros(config["num_envs"], dtype=bool)
        last_truncateds = np.zeros(config["num_envs"], dtype=bool)

        # --- Update Exploration Parameters (industry practice) --- #
        config = update_exploration_parameters(config, global_step)
        # Update agent's entropy coefficient with the current stage value
        agent.ent_coef = config["ent_coef"]
        # Update agent's target KL with the current stage value
        agent.target_kl = config["target_kl"]

        # --- Update Domain Randomization Intensity --- #
        if config.get("domain_randomize_schedule", False) and global_step % config.get("domain_randomize_check_interval", 100_000) == 0:
            # Calculate current intensity based on progress through training
            max_randomize_steps = config.get("domain_randomize_steps", 5_000_000)
            progress = min(global_step / max_randomize_steps, 1.0)
            old_intensity = config.get("domain_randomize_intensity", 0.1)
            max_intensity = config.get("domain_randomize_max", 0.5)
            new_intensity = min(0.1 + (max_intensity - 0.1) * progress, max_intensity)  # Gradually increase from 10% to max_intensity
            
            if new_intensity > old_intensity + 0.05:  # Only update if significant change (>5%)
                config["domain_randomize_intensity"] = new_intensity
                print(f"\n==== DOMAIN RANDOMIZATION UPDATE ====")
                print(f"Increasing domain randomization from {old_intensity:.2f} to {new_intensity:.2f}")
                print(f"====================================\n")
                
                # Log the change
                writer.add_scalar("Config/domain_randomize_intensity", new_intensity, global_step)
        
        for step in range(steps_per_rollout_per_env):
            step_global_step = global_step + step * config["num_envs"] # Estimate step for LR schedule
            
            # Add step counter for debugging
            if step % 50 == 0:
                print(f"Collecting step {step}/{steps_per_rollout_per_env}, timestep: {global_step}")

            # --- Update Learning Rate with improved schedule --- #
            progress = step_global_step / config["total_timesteps"]
            # Use actual base LR instead of hardcoded 5e-5
            base_lr = config["learning_rate"] * 0.5 * (1.0 + np.cos(np.pi * progress))
            # Keep LR higher for longer
            base_lr = max(base_lr, 1e-6) # Lowered minimum LR

            # Apply warmup after episode length change if active
            if config.get("lr_warmup_after_length_change", False):
                if global_step < config["lr_warmup_start_step"] + config["lr_warmup_duration"]:
                    warmup_progress = (global_step - config["lr_warmup_start_step"]) / config["lr_warmup_duration"]
                    warmup_factor = config["lr_warmup_factor"] + (1.0 - config["lr_warmup_factor"]) * warmup_progress
                    new_lr = base_lr * warmup_factor
                else:
                    new_lr = base_lr
                    # Disable warmup after completed
                    config["lr_warmup_after_length_change"] = False
            else:
                new_lr = base_lr

            for param_group in agent.actor_optimizer.param_groups:
                param_group['lr'] = new_lr
            for param_group in agent.critic_optimizer.param_groups:
                param_group['lr'] = new_lr
            agent.lr = new_lr # Update agent's current lr tracker

            # Get action, value, log_prob from agent
            # Normalize observations if enabled
            if obs_normalizer is not None:
                # Update observation statistics
                obs_normalizer.update(observations)
                # Normalize observations before passing to agent
                norm_observations = obs_normalizer.normalize(observations)
                actions, values, log_probs = agent.act(norm_observations)
            else:
                actions, values, log_probs = agent.act(observations)
                
            # Add exploration noise to actions if enabled
            if config.get("action_noise_std", 0.0) > 0.0:
                actions = add_action_noise(actions, config.get("action_noise_std"))

            # Step environment
            next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
            
            # Add reward normalization (keep original for tracking)
            raw_rewards = rewards.copy()
            if config.get("normalize_rewards", True):
                reward_rms.update(raw_rewards)
                rewards = rewards / np.sqrt(reward_rms.var + 1e-8)  # Normalize but don't center
                rewards = np.clip(rewards, -10.0, 10.0)  # Clip normalized rewards
            
            # Debug prints to check if episodes are terminating/truncating
            if any(terminateds) or any(truncateds):
                term_indices = np.where(terminateds)[0]
                trunc_indices = np.where(truncateds)[0]
                print(f"Step {step}: Terminated envs: {term_indices}, Truncated envs: {trunc_indices}")
                print(f"Info keys: {infos.keys()}")
            
            # Update episode trackers
            current_episode_rewards_vec += raw_rewards  # Track original, non-normalized rewards
            current_episode_lengths_vec += 1

            # Store transition in buffer
            buffer.add(observations, actions, rewards, terminateds, truncateds, values, log_probs)

            # Prepare for next iteration
            observations = next_observations
            last_terminateds = terminateds
            last_truncateds = truncateds

            # Check for finished episodes using final_info (Gymnasium >= 0.26)
            if "_final_info" in infos:
                finished_mask = infos["_final_info"]
                if any(finished_mask):
                    print(f"Found finished episodes at indices: {np.where(finished_mask)[0]}")
                    
                final_infos = infos["final_info"][finished_mask]
                for idx, final_info in enumerate(final_infos):
                    print(f"Processing final_info for idx {idx}: {final_info}")
                    if final_info is not None and "episode" in final_info:
                        ep_rew = final_info["episode"]["r"]
                        ep_len = final_info["episode"]["l"]
                        episode_rewards.append(ep_rew)
                        episode_lengths.append(ep_len)
                        # Add episode completion tracking
                        print(f"Detected episode completion. Rewards: {ep_rew:.2f}, Length: {ep_len}")
                        # Find original index and reset trackers
                        original_env_index = np.where(finished_mask)[0][idx]
                        current_episode_rewards_vec[original_env_index] = 0
                        current_episode_lengths_vec[original_env_index] = 0
            elif any(terminateds) or any(truncateds):
                # If _final_info is missing but episodes are ending, manually track them
                print("WARNING: Episodes finished but no _final_info in infos. Manual tracking needed.")
                for i in range(config["num_envs"]):
                    if terminateds[i] or truncateds[i]:
                        # Manually track episode stats
                        ep_reward = current_episode_rewards_vec[i]
                        ep_length = current_episode_lengths_vec[i]
                        episode_rewards.append(ep_reward)
                        episode_lengths.append(ep_length)
                        print(f"Manual episode tracking: Env {i}, Reward: {ep_reward:.2f}, Length: {ep_length}")
                        # Reset trackers
                        current_episode_rewards_vec[i] = 0
                        current_episode_lengths_vec[i] = 0

            # Update global step count AFTER processing step data
            global_step += config["num_envs"]

            # Early exit if total timesteps reached during rollout
            if global_step >= config["total_timesteps"]:
                break

        # --- End Rollout Collection --- #
        num_rollouts += 1
        rollout_duration = time.time() - rollout_start_time
        steps_collected = steps_per_rollout_per_env * config["num_envs"]

        # Compute advantages and returns
        with torch.no_grad():
            # Normalize observations if enabled
            if obs_normalizer is not None:
                norm_observations = obs_normalizer.normalize(observations)
                obs_tensor = torch.as_tensor(norm_observations).to(config["device"])
            else:
                obs_tensor = torch.as_tensor(observations).to(config["device"])
                
            features = agent.feature_extractor(obs_tensor.float() / 255.0)
            last_values = agent.critic(features).squeeze(-1).cpu().numpy() # Shape (num_envs,)

        final_dones = last_terminateds | last_truncateds
        buffer.compute_returns_and_advantages(last_values, final_dones)
        
        # Apply outlier emphasis if enabled
        if config.get("emphasize_outliers", False):
            # Get factor and threshold from config - use the value from current exploration stage
            factor = config.get("outlier_factor", 1.2)
            threshold = config.get("outlier_threshold", 1.5)
            
            # Emphasize outlier experiences in buffer advantages
            buffer.advantages, outlier_count = emphasize_outlier_experiences(
                buffer.advantages, factor=factor, threshold=threshold)
            
            # Log outlier emphasis
            if outlier_count > 0:
                outlier_percentage = 100 * outlier_count / buffer.advantages.size
                print(f"Emphasized {outlier_count} outlier experiences ({outlier_percentage:.2f}% of buffer) with factor {factor:.2f}")
                # Update returns based on modified advantages
                buffer.returns = buffer.advantages + buffer.values

        # --- Update Agent --- #
        update_start_time = time.time()
        metrics = agent.learn(buffer) # Returns dict of losses, kl, etc.
        update_duration = time.time() - update_start_time

        # --- Logging --- #
        if num_rollouts % config["log_interval"] == 0 and len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards)
            mean_length = np.mean(episode_lengths)
            fps = int(steps_collected / rollout_duration) if rollout_duration > 0 else float('inf')
            update_fps = int(steps_collected / update_duration) if update_duration > 0 else float('inf')
            total_duration = time.time() - start_time

            print(f"-- Rollout {num_rollouts} | Timesteps {global_step}/{config['total_timesteps']} --")
            print(f"  Stats (last {len(episode_rewards)} ep): Mean Reward: {mean_reward:.2f}, Mean Length: {mean_length:.1f}")
            print(f"  Speed: Rollout FPS: {fps}, Update FPS: {update_fps}")
            print(f"  Total Time: {total_duration:.2f}s")
            print(f"  LR: {new_lr:.2e}") 
            print(f"  Current max_episode_steps: {config['max_episode_steps']}")
            print(f"  Domain randomization: {config['domain_randomize_intensity']:.2f}")

            # TensorBoard Logging
            writer.add_scalar("Charts/mean_episode_reward", mean_reward, global_step)
            writer.add_scalar("Charts/mean_episode_length", mean_length, global_step)
            writer.add_scalar("Speed/rollout_fps", fps, global_step)
            writer.add_scalar("Speed/update_fps", update_fps, global_step)
            writer.add_scalar("Loss/policy_loss", metrics["policy_loss"], global_step)
            writer.add_scalar("Loss/value_loss", metrics["value_loss"], global_step)
            writer.add_scalar("Loss/entropy_loss", metrics["entropy_loss"], global_step)
            writer.add_scalar("Stats/approx_kl", metrics["approx_kl"], global_step)
            writer.add_scalar("Stats/clip_fraction", metrics["clip_fraction"], global_step)
            writer.add_scalar("Config/learning_rate", new_lr, global_step)
            writer.add_scalar("Config/max_episode_steps", config["max_episode_steps"], global_step)
            writer.add_scalar("Config/domain_randomize_intensity", config["domain_randomize_intensity"], global_step)
            
            # Log exploration parameters
            writer.add_scalar("Exploration/entropy_coefficient", agent.ent_coef, global_step)
            writer.add_scalar("Exploration/target_kl", agent.target_kl, global_step)
            writer.add_scalar("Exploration/action_noise", config.get("action_noise_std", 0.0), global_step)
            writer.add_scalar("Exploration/outlier_factor", config.get("outlier_factor", 1.2), global_step)
            if "outlier_count" in locals() and outlier_count > 0:
                writer.add_scalar("Exploration/outlier_percentage", outlier_percentage, global_step)
                writer.add_scalar("Exploration/outlier_count", outlier_count, global_step)

            # --- Save Best Model --- #
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_save_path = os.path.join(config["save_dir"], "best_model.pth")
                try:
                    torch.save({
                        'feature_extractor_state_dict': agent.feature_extractor.state_dict(),
                        'actor_state_dict': agent.actor.state_dict(),
                        'critic_state_dict': agent.critic.state_dict(),
                        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                        'global_step': global_step,
                        'num_rollouts': num_rollouts,
                        'best_mean_reward': best_mean_reward,
                        'config': config  # Save the current config with episode length
                    }, best_save_path)
                    print(f"** New best model saved to {best_save_path} with mean reward: {best_mean_reward:.2f} **")
                except Exception as e:
                    print(f"Error saving best model: {e}")

        # --- Periodic Saving --- #
        if num_rollouts > 0 and num_rollouts % config["save_interval"] == 0:
            save_path = os.path.join(config["save_dir"], f"ppo_carracing_finetuned_{global_step}.pth")
            try:
                torch.save({
                    'feature_extractor_state_dict': agent.feature_extractor.state_dict(),
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                    'global_step': global_step,
                    'num_rollouts': num_rollouts,
                    'best_mean_reward': best_mean_reward,
                    'config': config  # Save the current config with episode length
                }, save_path)
                print(f"Model saved to {save_path}")
            except Exception as e:
                print(f"Error saving periodic checkpoint: {e}")

        # --- Dynamic Episode Length Adjustment --- #
        if config["dynamic_episode_length"] and len(episode_rewards) > 0:
            for threshold, new_length in sorted(config["episode_length_thresholds"].items()):
                # Check if we've crossed a threshold and haven't updated for this threshold yet
                if mean_reward >= threshold and threshold > config["last_length_update_reward"]:
                    old_max_steps = config["max_episode_steps"]
                    config["max_episode_steps"] = new_length
                    config["last_length_update_reward"] = threshold
                    
                    print(f"\n==== DYNAMIC EPISODE LENGTH UPDATE ====")
                    print(f"Mean reward {mean_reward:.2f} has reached threshold {threshold}")
                    print(f"Increasing max_episode_steps from {old_max_steps} to {new_length}")
                    
                    # Close old environments and create new ones with updated episode length
                    env.close()
                    env_fns = [make_env(config["env_id"], config["seed"] + i, config["frame_stack"], config["max_episode_steps"]) 
                              for i in range(config["num_envs"])]
                    env = gymnasium.vector.AsyncVectorEnv(env_fns)
                    
                    # Reset environment and episode tracking variables
                    observations, infos = env.reset(seed=config["seed"])
                    current_episode_rewards_vec = np.zeros(config["num_envs"], dtype=np.float32)
                    current_episode_lengths_vec = np.zeros(config["num_envs"], dtype=np.int32)
                    
                    # Log the change
                    writer.add_scalar("Config/max_episode_steps", config["max_episode_steps"], global_step)
                    writer.add_text("Events", f"Increased episode length to {new_length} at reward {mean_reward:.2f}", global_step)
                    
                    print(f"Environment reset with new max episode length: {config['max_episode_steps']}")
                    print(f"====================================\n")

                    # Add entropy boost
                    config["original_ent_coef"] = agent.ent_coef
                    config["boosted_ent_coef"] = config["original_ent_coef"] * 1.5  # 50% entropy boost
                    agent.ent_coef = config["boosted_ent_coef"]
                    config["last_entropy_boost_step"] = global_step
                    config["entropy_boost_duration"] = 100000  # 100K steps decay period
                    
                    print(f"Boosted entropy from {config['original_ent_coef']:.4f} to {config['boosted_ent_coef']:.4f} decaying over {config['entropy_boost_duration']} steps")
                    
                    # Add LR warmup after length change
                    config["lr_warmup_after_length_change"] = True
                    config["lr_warmup_start_step"] = global_step
                    config["lr_warmup_duration"] = 100000  # 100K steps
                    config["lr_warmup_factor"] = 0.5  # Start at 50% of current LR
                    break  # Only apply one threshold update at a time

    print(f"Fine-tuning finished after {global_step} timesteps.")
    writer.close() # Close the TensorBoard writer
    env.close() # Close the vectorized environment 