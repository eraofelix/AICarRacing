import gymnasium as gym
import torch
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import gymnasium.vector # Added for vectorized environments

# Import custom modules
from src.env_wrappers import GrayScaleObservation, FrameStack # Ensure 'src.' prefix
from src.ppo_agent import PPOAgent                     # Ensure 'src.' prefix
from src.rollout_buffer import RolloutBuffer             # Ensure 'src.' prefix

# --- Configuration --- #
config = {
    # Environment settings
    "env_id": "CarRacing-v3",
    "frame_stack": 4,
    "seed": 42, # For reproducibility
    "num_envs": 4, # Number of parallel environments

    # Training settings
    "total_timesteps": 5_000_000, # Total steps to train the agent
    "learning_rate": 3e-5, # Initial learning rate (Lowered further)
    "buffer_size": 2048, # TOTAL steps per rollout across all envs
    "batch_size": 64,
    "ppo_epochs": 4, # Number of optimization epochs per rollout (Reduced)
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "target_kl": 0.04, # Optional: Stop updates if KL exceeds this (Increased)
    "features_dim": 256, # Output dim of CNN

    # Logging and Saving
    "log_interval": 1, # Log stats every N rollouts
    "save_interval": 10, # Save model every N rollouts
    "save_dir": "./models/ppo_carracing",
    "log_dir": "./logs/ppo_carracing",
    "load_checkpoint_path": None, # Set to None to train from scratch
    # TODO: Implement Observation Normalization

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

def make_env(env_id, seed, frame_stack):
    """Helper function that returns a function to create and wrap a single environment instance."""
    def _init():
        # Derive a unique seed for this process to ensure envs are different
        process_seed = seed + int.from_bytes(os.urandom(4), 'little')
        set_seeds(process_seed) # Also set seeds for torch/numpy within the process
        env = gym.make(env_id, continuous=True, domain_randomize=False)
        env.reset(seed=process_seed)
        env.action_space.seed(process_seed)
        env = GrayScaleObservation(env)
        env = FrameStack(env, frame_stack)
        # No need to reset again after FrameStack usually
        return env
    # Return the function itself, not the result of calling it
    return _init

# --- Main Training Loop --- #
if __name__ == "__main__":
    print(f"Using device: {config['device']}")
    set_seeds(config["seed"]) # Set seeds early for main process

    # Create vectorized environment FIRST
    print(f"Creating {config['num_envs']} parallel environments...")
    env_fns = [make_env(config["env_id"], config["seed"] + i, config["frame_stack"]) for i in range(config["num_envs"])]
    env = gymnasium.vector.AsyncVectorEnv(env_fns)
    print(f"Observation Space: {env.single_observation_space}")
    print(f"Action Space: {env.single_action_space}")

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

    # --- Load Checkpoint if specified --- #
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

            start_global_step = checkpoint.get('global_step', 0)
            start_num_rollouts = checkpoint.get('num_rollouts', 0)
            best_mean_reward = checkpoint.get('best_mean_reward', -np.inf)

            print(f"Resuming training from global_step={start_global_step}, num_rollouts={start_num_rollouts}")
            print(f"Loaded best mean reward: {best_mean_reward:.2f}")
        except Exception as e:
            print(f"Error loading checkpoint state_dicts: {e}. Training from scratch.")
            start_global_step = 0
            start_num_rollouts = 0
            best_mean_reward = -np.inf
    else:
        print("No checkpoint found or specified, starting training from scratch.")

    # Initialize environment state - use the vector env reset
    observations, infos = env.reset(seed=config["seed"])
    global_step = start_global_step
    num_rollouts = start_num_rollouts
    # Track rewards and lengths for each active episode in parallel envs
    current_episode_rewards_vec = np.zeros(config["num_envs"], dtype=np.float32)
    current_episode_lengths_vec = np.zeros(config["num_envs"], dtype=np.int32)

    print("Starting training...")
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

        for step in range(steps_per_rollout_per_env):
            step_global_step = global_step + step * config["num_envs"] # Estimate step for LR schedule

            # --- Update Learning Rate --- #
            progress = step_global_step / config["total_timesteps"]
            new_lr = agent.initial_lr * (1.0 - progress)
            new_lr = max(new_lr, 1e-6) # Avoid LR being exactly zero
            for param_group in agent.actor_optimizer.param_groups:
                param_group['lr'] = new_lr
            for param_group in agent.critic_optimizer.param_groups:
                param_group['lr'] = new_lr
            agent.lr = new_lr # Update agent's current lr tracker

            # Get action, value, log_prob from agent
            actions, values, log_probs = agent.act(observations)

            # Step environment
            next_observations, rewards, terminateds, truncateds, infos = env.step(actions)

            # Update episode trackers
            current_episode_rewards_vec += rewards
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
                final_infos = infos["final_info"][finished_mask]
                for idx, final_info in enumerate(final_infos):
                    if final_info is not None and "episode" in final_info:
                        ep_rew = final_info["episode"]["r"]
                        ep_len = final_info["episode"]["l"]
                        episode_rewards.append(ep_rew)
                        episode_lengths.append(ep_len)
                        # Find original index and reset trackers
                        original_env_index = np.where(finished_mask)[0][idx]
                        current_episode_rewards_vec[original_env_index] = 0
                        current_episode_lengths_vec[original_env_index] = 0

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
            obs_tensor = torch.as_tensor(observations).to(config["device"])
            features = agent.feature_extractor(obs_tensor.float() / 255.0)
            last_values = agent.critic(features).squeeze(-1).cpu().numpy() # Shape (num_envs,)

        final_dones = last_terminateds | last_truncateds
        buffer.compute_returns_and_advantages(last_values, final_dones)

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
            print(f"  LR: {new_lr:.2e}") # Print current LR

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
                        'best_mean_reward': best_mean_reward
                    }, best_save_path)
                    print(f"** New best model saved to {best_save_path} with mean reward: {best_mean_reward:.2f} **")
                except Exception as e:
                    print(f"Error saving best model: {e}")

        # --- Periodic Saving --- #
        # Avoid saving at rollout 0 if not resuming from step 0
        save_cond = (start_global_step == 0 and num_rollouts > 0) or \
                    (start_global_step > 0 and num_rollouts > start_num_rollouts)
        if save_cond and num_rollouts % config["save_interval"] == 0:
            save_path = os.path.join(config["save_dir"], f"ppo_carracing_{global_step}.pth")
            try:
                torch.save({
                    'feature_extractor_state_dict': agent.feature_extractor.state_dict(),
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                    'global_step': global_step,
                    'num_rollouts': num_rollouts,
                    'best_mean_reward': best_mean_reward
                }, save_path)
                print(f"Model saved to {save_path}")
            except Exception as e:
                print(f"Error saving periodic checkpoint: {e}")

    print(f"Training finished after {global_step} timesteps.")
    writer.close() # Close the TensorBoard writer
    env.close() # Close the vectorized environment
