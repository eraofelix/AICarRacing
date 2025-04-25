import gymnasium as gym
import torch
import numpy as np
import os
import time
import argparse  # Add this import for command-line arguments
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import gymnasium.vector

# Import custom modules
from src.env_wrappers import GrayScaleObservation, FrameStack, TimeLimit
from src.ppo_agent import PPOAgent
from src.rollout_buffer import RolloutBuffer

# --- Configuration --- #
# Performance-optimized parameters balanced for your hardware
config = {
    # Environment
    "env_id": "CarRacing-v3",
    "frame_stack": 4,
    "num_envs": 8,  # Keep at 8 to avoid CPU overload
    "max_episode_steps": 1000,
    "seed": 42,
    
    # PPO Core Parameters
    "total_timesteps": 6_000_000,
    "learning_rate": 1e-6,  # Reverted to value used for 680-score run
    "buffer_size": 2048,
    "batch_size": 128,    # Increased from 64 for better GPU utilization, but not as high as 256
    "ppo_epochs": 6,      # Reverted
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.08, # Reverted
    "vf_coef": 1.0,       # Reverted
    "ent_coef": 0.008,    # Reverted
    "max_grad_norm": 0.5,
    "target_kl": 0.2,
    "features_dim": 256,
    
    # Reward shaping
    "use_reward_shaping": True,
    "velocity_reward_weight": 0.005,
    "survival_reward": 0.05, # Reverted
    "track_penalty": 5.0,   # Reverted
    "steering_smooth_weight": 0.3,
    "acceleration_while_turning_penalty_weight": 0.8,
    
    # Performance optimizations - CPU-friendly settings
    "torch_num_threads": 7,   # Leave one core free for system tasks
    "mixed_precision": False, # SETTING TO FALSE to avoid RuntimeError
    "pin_memory": True,       # Speed up data transfer to GPU
    "async_envs": True,       # Use async environments which are more CPU efficient
    
    # Logging and saving
    "log_interval": 1,
    "save_interval": 10,
    "save_dir": "./models/ppo_simple",
    "log_dir": "./logs/ppo_simple",
    
    # Checkpoint to load (set to None to start fresh training)
    #"checkpoint_path": "./models/ppo_simple/best_model.pth",  # Change to your checkpoint path or None
    "checkpoint_path": "./models/ppo_simple/Evaluated679.pth",
    # Hardware
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# --- Helper Functions --- #
def set_seeds(seed):
    """Set seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Set performance-enhancing environment variables
os.environ['OMP_NUM_THREADS'] = str(config["torch_num_threads"])
os.environ['MKL_NUM_THREADS'] = str(config["torch_num_threads"])
torch.set_num_threads(config["torch_num_threads"])  # Limit intraop parallelism

# Configure for better GPU utilization
if config["device"] == "cuda":
    # Set CUDA options to optimize throughput
    torch.backends.cudnn.benchmark = True
    if config["mixed_precision"]:
        # Enable automatic mixed precision for faster training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def make_env(env_id, seed, frame_stack, max_episode_steps, idx=0):
    """Create a single environment with wrappers"""
    def _init():
        # Create unique seeds per environment
        env_seed = seed + idx
        
        # Create environment with continuous actions
        env = gym.make(env_id, continuous=True, domain_randomize=False, render_mode=None)
        
        # Seed the environment
        env.reset(seed=env_seed)
        env.action_space.seed(env_seed)
        
        # Add custom reward shaping if enabled
        if config["use_reward_shaping"]:
            env = RewardShapingWrapper(env, 
                                      velocity_weight=config["velocity_reward_weight"],
                                      survival_reward=config["survival_reward"],
                                      track_penalty=config["track_penalty"],
                                      steering_smooth_weight=config["steering_smooth_weight"],
                                      acceleration_while_turning_penalty_weight=config["acceleration_while_turning_penalty_weight"])
        
        # Apply GrayScaleObservation after reward shaping so RewardShapingWrapper gets RGB
        env = GrayScaleObservation(env)
        
        # Add time limit
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        
        # Add frame stacking
        env = FrameStack(env, frame_stack)
        
        return env
    
    return _init

class RewardShapingWrapper(gym.Wrapper):
    """Reward shaping wrapper with track adherence and steering smoothness rewards"""
    def __init__(self, env, velocity_weight=0.005, survival_reward=0.05, 
                 track_penalty=2.0, steering_smooth_weight=0.1,
                 acceleration_while_turning_penalty_weight=0.5):
        super().__init__(env)
        self.velocity_weight = velocity_weight
        self.survival_reward = survival_reward
        self.track_penalty = track_penalty
        self.steering_smooth_weight = steering_smooth_weight
        self.acceleration_while_turning_penalty_weight = acceleration_while_turning_penalty_weight
        
        # For tracking previous actions
        self.last_steering = 0.0
        
        # For tracking reward components
        self.episode_velocity_rewards = 0.0
        self.episode_survival_rewards = 0.0
        self.episode_track_penalties = 0.0
        self.episode_steering_penalties = 0.0
        self.episode_acceleration_while_turning_penalties = 0.0
        self.steps_off_track = 0
        
        # Racing line rewards
        self.centerline_reward_weight = 0.5  # Increased from 0.2 to emphasize staying on track
        # Add track return reward for recovery behavior
        self.track_return_weight = 0.3
        
        # Add to RewardShapingWrapper.__init__
        self.speed_consistency_weight = 0.05
        self.last_speed = 0.0
        
    def reset(self, **kwargs):
        self.last_steering = 0.0
        
        # Reset reward component trackers
        self.episode_velocity_rewards = 0.0
        self.episode_survival_rewards = 0.0
        self.episode_track_penalties = 0.0
        self.episode_steering_penalties = 0.0
        self.episode_acceleration_while_turning_penalties = 0.0
        self.steps_off_track = 0
        
        # Add to RewardShapingWrapper.reset
        self.last_speed = 0.0
        
        obs, info = self.env.reset(**kwargs)
        
        # Add reward component info
        info['velocity_rewards'] = 0.0
        info['survival_rewards'] = 0.0
        info['track_penalties'] = 0.0
        info['steering_penalties'] = 0.0
        info['acceleration_while_turning_penalties'] = 0.0
        info['steps_off_track'] = 0
        
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract components from action (steering, gas, brake)
        steering = action[0]
        gas = action[1]
        
        # Initialize reward components for this step
        velocity_reward = 0.0
        survival_reward = 0.0
        track_penalty = 0.0
        steering_penalty = 0.0
        acceleration_while_turning_penalty = 0.0
        off_track = False
        
        # 1. Velocity rewards
        speed = info.get('speed', 0.0) if isinstance(info, dict) else 0.0
        if speed > 0:
            velocity_reward = speed * self.velocity_weight
            reward += velocity_reward
            self.episode_velocity_rewards += velocity_reward
        
        # 2. Survival rewards
        survival_reward = self.survival_reward
        reward += survival_reward
        self.episode_survival_rewards += survival_reward
        
        # 3. Track adherence reward/penalty
        # Check if car is on the track by examining pixels in observation
        if len(obs.shape) == 3 and obs.shape[2] == 3:  # RGB observation
            # Check bottom center region for green pixels (off-track)
            car_area = obs[84:94, 42:54, :]  # Adjust region as needed
            # Green channel is high for grass
            green_channel = car_area[:, :, 1]
            red_channel = car_area[:, :, 0]
            
            # If green dominates and red is low, likely off-track
            off_track = np.mean(green_channel) > 150 and np.mean(red_channel) < 100
            
            if off_track:
                track_penalty = self.track_penalty
                reward -= track_penalty
                self.episode_track_penalties += track_penalty
                self.steps_off_track += 1
                
                # Add recovery guidance when off-track to prevent donuts
                # This rewards steering that points back to the track (approximate direction)
                track_direction = np.array([1.0, 0.0])  # Simplified: assume track is ahead
                car_direction = np.array([np.cos(steering * np.pi), np.sin(steering * np.pi)])
                # Reward alignment with track direction
                track_return_reward = np.dot(track_direction, car_direction) * self.track_return_weight
                reward += track_return_reward
        
        # 4. Steering smoothness penalty (speed-dependent)
        steering_change = abs(steering - self.last_steering)
        # Make steering penalty higher at high speeds to prevent spinouts
        steering_penalty = steering_change * self.steering_smooth_weight * (1.0 + speed * 0.1)
        reward -= steering_penalty
        self.episode_steering_penalties += steering_penalty
        
        # 5. Centerline reward - reward staying in the center of the track
        if not off_track:
            # Red channel is high for the road, low for the edges
            # Higher values = closer to center
            road_redness = np.mean(red_channel)
            centerline_reward = min(road_redness / 200, 1.0) * self.centerline_reward_weight
            reward += centerline_reward
        
        # 6. Speed consistency reward
        speed_change = abs(speed - self.last_speed)
        speed_consistency_reward = -speed_change * self.speed_consistency_weight
        reward += speed_consistency_reward
        self.last_speed = speed
        
        # 7. Acceleration while turning penalty (NEW)
        # Penalize applying gas while steering angle is high
        steering_threshold = 0.4 # Tune this - angle above which penalty applies
        gas_threshold = 0.1 # Tune this - gas above which penalty applies
        if abs(steering) > steering_threshold and gas > gas_threshold:
            # Penalty scales with how much gas is applied and how sharp the turn is
            acceleration_while_turning_penalty = (
                self.acceleration_while_turning_penalty_weight * 
                (gas - gas_threshold) * 
                (abs(steering) - steering_threshold)
            )
            reward -= acceleration_while_turning_penalty
            self.episode_acceleration_while_turning_penalties += acceleration_while_turning_penalty
        
        # Store current steering for next step
        self.last_steering = steering
        
        # Add reward components to info
        info['velocity_rewards'] = velocity_reward
        info['survival_rewards'] = survival_reward
        info['track_penalties'] = track_penalty
        info['steering_penalties'] = steering_penalty
        info['acceleration_while_turning_penalties'] = acceleration_while_turning_penalty
        info['off_track'] = off_track
        
        # Add episode totals to info
        info['episode_velocity_rewards'] = self.episode_velocity_rewards
        info['episode_survival_rewards'] = self.episode_survival_rewards
        info['episode_track_penalties'] = self.episode_track_penalties
        info['episode_steering_penalties'] = self.episode_steering_penalties
        info['episode_acceleration_while_turning_penalties'] = self.episode_acceleration_while_turning_penalties
        info['steps_off_track'] = self.steps_off_track
        
        return obs, reward, terminated, truncated, info

def load_checkpoint(agent, checkpoint_path, config, device):
    """Load model weights and optimizer states from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return None, 0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # Load model weights
        agent.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])

        # Optionally load optimizer states if they exist
        # ---> COMMENTED OUT AGAIN to test KL stability <----
        # if 'actor_optimizer_state_dict' in checkpoint and 'critic_optimizer_state_dict' in checkpoint:
        #     agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        #     agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        #     print("Loaded optimizer states")
        # else:
        #     print("Optimizer states not found in checkpoint.")
        print("Skipping optimizer state loading.")

        # Get global step from checkpoint
        global_step = checkpoint.get('global_step', 0)
        print(f"Resuming from step {global_step}")
        
        # Get best mean reward if available
        best_mean_reward = checkpoint.get('mean_reward', -np.inf)
        print(f"Best mean reward from checkpoint: {best_mean_reward:.2f}")
        
        return best_mean_reward, global_step
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return -np.inf, 0 # Return default values on error

# --- Main Training Loop with Performance Optimizations --- #
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train PPO agent for CarRacing")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint file to resume training from")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override total timesteps in config")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed in config")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Override log directory")
    args = parser.parse_args()
    
    # Override config with command-line arguments
    if args.steps is not None:
        config["total_timesteps"] = args.steps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir
    
    print(f"Using device: {config['device']} with {config['num_envs']} environments")
    if config["mixed_precision"]:
        print("Using mixed precision training")
    print(f"Torch threads: {config['torch_num_threads']}")
    
    set_seeds(config["seed"])

    # Create directories
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    # Create vectorized environment
    print(f"Creating {config['num_envs']} parallel environments...")
    env_fns = [make_env(config["env_id"], config["seed"], config["frame_stack"], 
                        config["max_episode_steps"], i) for i in range(config["num_envs"])]
    
    # Use AsyncVectorEnv instead of SyncVectorEnv for better CPU efficiency with 8 envs
    env = gymnasium.vector.AsyncVectorEnv(env_fns) if config["async_envs"] else gymnasium.vector.SyncVectorEnv(env_fns)
    
    print(f"Observation Space: {env.single_observation_space}")
    print(f"Action Space: {env.single_action_space}")
    
    # Create PPO agent
    agent = PPOAgent(
        env.single_observation_space,
        env.single_action_space,
        lr=config["learning_rate"],
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
        device=config["device"]
    )
    
    # Load checkpoint if specified (command-line arg takes priority, then config)
    best_mean_reward = -np.inf
    global_step = 0
    checkpoint_path = args.checkpoint if args.checkpoint else config["checkpoint_path"]
    if checkpoint_path:
        # Load checkpoint and retrieve reward and step count
        loaded_reward, loaded_step = load_checkpoint(agent, checkpoint_path, config, config["device"])
        if loaded_reward is not None:
            best_mean_reward = loaded_reward
            global_step = loaded_step
            # Set the agent's internal step counter for the LR schedule
            agent.steps_done = global_step
            print(f"Set agent steps_done to {agent.steps_done} for LR schedule.")
    
    # Create rollout buffer with optimized settings
    buffer_size_per_env = config["buffer_size"] // config["num_envs"]
    buffer = RolloutBuffer(
        buffer_size_per_env,
        env.single_observation_space,
        env.single_action_space,
        num_envs=config["num_envs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        device=config["device"]
    )
    
    # Setup logging
    writer = SummaryWriter(log_dir=config["log_dir"])
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    
    # Start training
    observations, infos = env.reset(seed=config["seed"])
    num_rollouts = 0
    current_episode_rewards = np.zeros(config["num_envs"], dtype=np.float32)
    current_episode_lengths = np.zeros(config["num_envs"], dtype=np.int32)
    start_time = time.time()
    
    # Setup for mixed precision
    auto_cast = torch.cuda.amp.autocast if config["device"] == "cuda" and config["mixed_precision"] else lambda: nullcontext()
    scaler = torch.cuda.amp.GradScaler() if config["device"] == "cuda" and config["mixed_precision"] else None
    
    # Define a nullcontext for when mixed precision is disabled
    class nullcontext:
        def __enter__(self): return None
        def __exit__(self, *args): pass
    
    print(f"Starting training from step {global_step}/{config['total_timesteps']}")
    try:
        while global_step < config["total_timesteps"]:
            buffer.pos = 0
            buffer.full = False
            rollout_start_time = time.time()
            
            steps_per_rollout = buffer_size_per_env
            last_dones = np.zeros(config["num_envs"], dtype=bool)
            
            # Collect rollout data
            for step in range(steps_per_rollout):
                # Convert observations to tensor for the actor
                obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=config["device"])
                
                with torch.no_grad():
                    actions, values, log_probs = agent.act(obs_tensor)
                
                # Step environment
                next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
                dones = terminateds | truncateds
                
                # Update episode stats
                current_episode_rewards += rewards
                current_episode_lengths += 1
                
                # Store transition in buffer
                buffer.add(observations, actions, rewards, terminateds, truncateds, values, log_probs)
                
                # Prepare for next step
                observations = next_observations
                last_dones = dones
                
                # Check for episode completions
                if "_final_info" in infos:
                    finished_mask = infos["_final_info"]
                    if any(finished_mask):
                        final_infos = infos["final_info"][finished_mask]
                        for idx, final_info in enumerate(final_infos):
                            if final_info is not None and "episode" in final_info:
                                ep_rew = final_info["episode"]["r"]
                                ep_len = final_info["episode"]["l"]
                                episode_rewards.append(ep_rew)
                                episode_lengths.append(ep_len)
                                print(f"Episode completed: reward={ep_rew:.2f}, length={ep_len}")
                                
                                # Reset trackers for this env
                                env_idx = np.where(finished_mask)[0][idx]
                                current_episode_rewards[env_idx] = 0
                                current_episode_lengths[env_idx] = 0
                elif any(dones):
                    # Manual tracking if _final_info missing
                    for i in range(config["num_envs"]):
                        if dones[i]:
                            ep_reward = current_episode_rewards[i]
                            ep_length = current_episode_lengths[i]
                            episode_rewards.append(ep_reward)
                            episode_lengths.append(ep_length)
                            print(f"Episode completed: reward={ep_reward:.2f}, length={ep_length}")
                            
                            # Reset trackers
                            current_episode_rewards[i] = 0
                            current_episode_lengths[i] = 0
                
                # Update global step
                global_step += config["num_envs"]
                
                # Early exit if total timesteps reached
                if global_step >= config["total_timesteps"]:
                    break
            
            # End of rollout - compute returns and advantages
            with torch.no_grad():
                # More efficient tensor handling
                obs_tensor = torch.as_tensor(observations, device=config["device"]).float() / 255.0
                features = agent.feature_extractor(obs_tensor)
                last_values = agent.critic(features).squeeze(-1).cpu().numpy()
            
            # Compute returns and advantages
            buffer.compute_returns_and_advantages(last_values, last_dones)
            
            # Update agent using mixed precision if enabled
            if config["mixed_precision"] and config["device"] == "cuda":
                metrics = agent.learn_mixed_precision(buffer, scaler)
            else:
                metrics = agent.learn(buffer)

            # Update learning rate schedule
            current_lr = agent.update_learning_rate(config['total_timesteps'])

            # Update rollout counter
            num_rollouts += 1
            
            # Logging
            if num_rollouts % config["log_interval"] == 0 and len(episode_rewards) > 0:
                # Calculate metrics
                mean_reward = np.mean(episode_rewards)
                mean_length = np.mean(episode_lengths)
                fps = int(steps_per_rollout * config["num_envs"] / (time.time() - rollout_start_time))
                
                # Print metrics
                print(f"\n====== Rollout {num_rollouts} | Step {global_step}/{config['total_timesteps']} ======")
                print(f"Mean reward: {mean_reward:.2f}")
                print(f"Mean episode length: {mean_length:.1f}")
                print(f"FPS: {fps}")
                print(f"Policy loss: {metrics['policy_loss']:.4f}")
                print(f"Value loss: {metrics['value_loss']:.4f}")
                print(f"KL divergence: {metrics['approx_kl']:.4f}")
                
                # Log to tensorboard
                writer.add_scalar("charts/mean_reward", mean_reward, global_step)
                writer.add_scalar("charts/mean_length", mean_length, global_step)
                writer.add_scalar("charts/fps", fps, global_step)
                writer.add_scalar("losses/policy_loss", metrics["policy_loss"], global_step)
                writer.add_scalar("losses/value_loss", metrics["value_loss"], global_step)
                writer.add_scalar("losses/entropy", metrics["entropy_loss"], global_step)
                writer.add_scalar("losses/approx_kl", metrics["approx_kl"], global_step)
                
                # Log reward components safely for finished environments
                if "_final_info" in infos:
                    final_infos = infos.get("final_info") # Get the final_info dictionary
                    if final_infos is not None:
                        # Filter for environments that actually finished and have info
                        finished_mask = infos.get("_final_info")
                        if finished_mask is not None:
                            valid_final_infos = final_infos[finished_mask]
                            env_indices = np.where(finished_mask)[0]

                            for i, env_info in enumerate(valid_final_infos):
                                env_idx = env_indices[i] # Get the original environment index
                                if env_info is not None:
                                    print(f"DEBUG: Logging info for completed env {env_idx}: {env_info}")

                                    # Log standard episode stats if available
                                    if "episode" in env_info:
                                        writer.add_scalar(f"charts/episode_reward_env_{env_idx}", env_info["episode"]["r"], global_step)
                                        writer.add_scalar(f"charts/episode_length_env_{env_idx}", env_info["episode"]["l"], global_step)

                                    # Log custom reward/penalty components if available
                                    if 'episode_velocity_rewards' in env_info:
                                        writer.add_scalar(f"rewards/velocity_rewards_env_{env_idx}", env_info['episode_velocity_rewards'], global_step)
                                    if 'episode_survival_rewards' in env_info:
                                        writer.add_scalar(f"rewards/survival_rewards_env_{env_idx}", env_info['episode_survival_rewards'], global_step)
                                    if 'episode_track_penalties' in env_info:
                                        writer.add_scalar(f"penalties/track_penalties_env_{env_idx}", env_info['episode_track_penalties'], global_step)
                                    if 'episode_steering_penalties' in env_info:
                                        writer.add_scalar(f"penalties/steering_penalties_env_{env_idx}", env_info['episode_steering_penalties'], global_step)
                                    if 'episode_acceleration_while_turning_penalties' in env_info:
                                        writer.add_scalar(f"penalties/acceleration_while_turning_env_{env_idx}", env_info['episode_acceleration_while_turning_penalties'], global_step)
                                    if 'steps_off_track' in env_info:
                                        writer.add_scalar(f"driving/steps_off_track_env_{env_idx}", env_info['steps_off_track'], global_step)
                                        if 'episode' in env_info and 'l' in env_info['episode'] and env_info['episode']['l'] > 0:
                                            ep_len = env_info['episode']['l']
                                            off_track_pct = 100 * env_info['steps_off_track'] / ep_len
                                            writer.add_scalar(f"driving/percent_off_track_env_{env_idx}", off_track_pct, global_step)
                
                # Save best model
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    best_model_path = os.path.join(config["save_dir"], "best_model.pth")
                    
                    torch.save({
                        'feature_extractor_state_dict': agent.feature_extractor.state_dict(),
                        'actor_state_dict': agent.actor.state_dict(),
                        'critic_state_dict': agent.critic.state_dict(),
                        'global_step': global_step,
                        'mean_reward': mean_reward,
                    }, best_model_path)
                    
                    print(f"New best model saved with mean reward: {best_mean_reward:.2f}")
            
            # Save checkpoint periodically
            if num_rollouts % config["save_interval"] == 0:
                checkpoint_path = os.path.join(config["save_dir"], f"checkpoint_{global_step}.pth")
                
                torch.save({
                    'feature_extractor_state_dict': agent.feature_extractor.state_dict(),
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'global_step': global_step,
                    'config': config,
                    'mean_reward': best_mean_reward,
                }, checkpoint_path)
                
                print(f"Checkpoint saved at step {global_step}")
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    print(f"Training completed after {global_step} steps")
    env.close()
    writer.close() 