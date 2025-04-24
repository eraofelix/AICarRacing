import gymnasium as gym
import torch
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import gymnasium.vector

# Import custom modules
from src.env_wrappers import GrayScaleObservation, FrameStack, TimeLimit
from src.ppo_agent import PPOAgent
from src.rollout_buffer import RolloutBuffer

# --- Configuration --- #
# Simple, proven PPO parameters with minimal complexity
config = {
    # Environment
    "env_id": "CarRacing-v3",
    "frame_stack": 4,
    "num_envs": 8,
    "max_episode_steps": 600,  # Fixed episode length
    "seed": 42,
    
    # PPO Core Parameters (standard values)
    "total_timesteps": 5_000_000,
    "learning_rate": 3e-5,
    "buffer_size": 2048,
    "batch_size": 64,
    "ppo_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,
    "features_dim": 256,
    
    # Simple reward shaping
    "use_reward_shaping": True,
    "velocity_reward_weight": 0.005,
    "survival_reward": 0.05,
    
    # Logging and saving
    "log_interval": 1,
    "save_interval": 10,
    "save_dir": "./models/ppo_simple",
    "log_dir": "./logs/ppo_simple",
    
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

def make_env(env_id, seed, frame_stack, max_episode_steps, idx=0):
    """Create a single environment with wrappers"""
    def _init():
        # Create unique seeds per environment
        env_seed = seed + idx
        
        # Create environment with continuous actions
        env = gym.make(env_id, continuous=True, domain_randomize=False)
        
        # Seed the environment
        env.reset(seed=env_seed)
        env.action_space.seed(env_seed)
        
        # Apply wrappers
        env = GrayScaleObservation(env)
        
        # Add custom reward shaping if enabled
        if config["use_reward_shaping"]:
            env = RewardShapingWrapper(env, 
                                      velocity_weight=config["velocity_reward_weight"],
                                      survival_reward=config["survival_reward"])
        
        # Add time limit
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        
        # Add frame stacking
        env = FrameStack(env, frame_stack)
        
        return env
    
    return _init

class RewardShapingWrapper(gym.Wrapper):
    """Simple reward shaping wrapper with minimal adjustments"""
    def __init__(self, env, velocity_weight=0.005, survival_reward=0.05):
        super().__init__(env)
        self.velocity_weight = velocity_weight
        self.survival_reward = survival_reward
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get speed information if available
        speed = info.get('speed', 0.0) if isinstance(info, dict) else 0.0
        
        # Add velocity reward component
        if speed > 0:
            # Small bonus for maintaining velocity
            velocity_bonus = speed * self.velocity_weight
            reward += velocity_bonus
            
        # Add survival bonus
        reward += self.survival_reward
            
        return obs, reward, terminated, truncated, info

# --- Main Training Loop --- #
if __name__ == "__main__":
    print(f"Using device: {config['device']}")
    set_seeds(config["seed"])

    # Create directories
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    # Create vectorized environment
    print(f"Creating {config['num_envs']} parallel environments...")
    env_fns = [make_env(config["env_id"], config["seed"], config["frame_stack"], 
                        config["max_episode_steps"], i) for i in range(config["num_envs"])]
    env = gymnasium.vector.AsyncVectorEnv(env_fns)
    
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
    
    # Create rollout buffer
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
    best_mean_reward = -np.inf
    
    # Start training
    observations, infos = env.reset(seed=config["seed"])
    global_step = 0
    num_rollouts = 0
    current_episode_rewards = np.zeros(config["num_envs"], dtype=np.float32)
    current_episode_lengths = np.zeros(config["num_envs"], dtype=np.int32)
    start_time = time.time()
    
    print("Starting training...")
    while global_step < config["total_timesteps"]:
        # Reset buffer for each rollout
        buffer.pos = 0
        buffer.full = False
        rollout_start_time = time.time()
        
        # Calculate steps per rollout
        steps_per_rollout = buffer_size_per_env
        
        # Track dones for GAE calculation
        last_dones = np.zeros(config["num_envs"], dtype=bool)
        
        # Collect rollout
        for step in range(steps_per_rollout):
            # Get actions from agent
            with torch.no_grad():
                actions, values, log_probs = agent.act(observations)
            
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
            obs_tensor = torch.as_tensor(observations).to(config["device"])
            features = agent.feature_extractor(obs_tensor.float() / 255.0)
            last_values = agent.critic(features).squeeze(-1).cpu().numpy()
        
        # Compute returns and advantages
        buffer.compute_returns_and_advantages(last_values, last_dones)
        
        # Update agent
        metrics = agent.learn(buffer)
        
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
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                'global_step': global_step,
                'config': config,
            }, checkpoint_path)
            
            print(f"Checkpoint saved at step {global_step}")
    
    # End of training
    print(f"Training completed after {global_step} steps")
    env.close()
    writer.close() 