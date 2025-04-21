import gymnasium as gym
import torch
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
from env_wrappers import GrayScaleObservation, FrameStack
from ppo_agent import PPOAgent
from rollout_buffer import RolloutBuffer

# --- Configuration --- #
config = {
    # Environment settings
    "env_id": "CarRacing-v3",
    "frame_stack": 4,
    "seed": 42, # For reproducibility

    # Training settings
    "total_timesteps": 5_000_000, # Total steps to train the agent (Increased from 1M)
    "learning_rate": 3e-4, # Initial learning rate
    "buffer_size": 2048, # Steps collected per rollout before update
    "batch_size": 64,
    "ppo_epochs": 10, # Number of optimization epochs per rollout
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "target_kl": 0.015, # Optional: Stop updates if KL exceeds this
    "features_dim": 256, # Output dim of CNN

    # Logging and Saving
    "log_interval": 1, # Log stats every N rollouts
    "save_interval": 10, # Save model every N rollouts
    "save_dir": "./models/ppo_carracing",
    "log_dir": "./logs/ppo_carracing",
    "load_checkpoint_path": "./models/ppo_carracing/ppo_carracing_471040.pth", # Path to load a checkpoint from, None to train from scratch
    # TODO: Implement TensorBoard logging

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
    """Create and wrap the environment."""
    env = gym.make(env_id, continuous=True, domain_randomize=False)
    # Important: Set seed for action space and observation space BEFORE wrapping
    env.reset(seed=seed)
    env.action_space.seed(seed)
    # env.observation_space.seed(seed) # Does not work directly on Dict spaces

    env = GrayScaleObservation(env)
    env = FrameStack(env, frame_stack)
    return env

# --- Main Training Loop --- #
if __name__ == "__main__":
    print(f"Using device: {config['device']}")
    set_seeds(config["seed"]) # Set seeds early

    # Create environment
    env = make_env(config["env_id"], config["seed"], config["frame_stack"])

    # Create Agent
    agent = PPOAgent(env.observation_space,
                     env.action_space,
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
                     device=config["device"])

    # Create Rollout Buffer
    buffer = RolloutBuffer(config["buffer_size"],
                           env.observation_space,
                           env.action_space,
                           gamma=config["gamma"],
                           gae_lambda=config["gae_lambda"],
                           device=config["device"])

    # Logging setup (basic for now)
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    episode_rewards = deque(maxlen=100) # Store last 100 episode rewards for averaging
    episode_lengths = deque(maxlen=100)
    writer = SummaryWriter(log_dir=config["log_dir"]) # Initialize TensorBoard writer
    best_mean_reward = -np.inf # Initialize best mean reward
    current_episode_reward = 0
    current_episode_length = 0
    start_time = time.time()

    # --- Load Checkpoint if specified --- #
    start_global_step = 0
    start_num_rollouts = 0
    if config["load_checkpoint_path"] and os.path.exists(config["load_checkpoint_path"]):
        print(f"Loading checkpoint from: {config['load_checkpoint_path']}")
        checkpoint = torch.load(config["load_checkpoint_path"], map_location=config["device"])

        agent.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        start_global_step = checkpoint.get('global_step', 0)
        start_num_rollouts = checkpoint.get('num_rollouts', 0)
        best_mean_reward = checkpoint.get('best_mean_reward', -np.inf) # Load best reward

        print(f"Resuming training from global_step={start_global_step}, num_rollouts={start_num_rollouts}")
        print(f"Loaded best mean reward: {best_mean_reward:.2f}")
    else:
        print("No checkpoint found or specified, starting training from scratch.")

    # Initialize environment state
    # Reset even if loading checkpoint to ensure consistent starting state for the next rollout
    observation, info = env.reset(seed=config["seed"]) # Use initial seed
    global_step = start_global_step
    num_rollouts = start_num_rollouts

    print("Starting training...")
    while global_step < config["total_timesteps"]:
        # --- Collect Rollout --- #
        buffer.pos = 0 # Reset buffer position for new rollout
        buffer.full = False
        rollout_start_time = time.time()

        for step in range(config["buffer_size"]):
            global_step += 1
            current_episode_length += 1

            # Get action, value, log_prob from agent
            action, value, log_prob = agent.act(observation)

            # Perform action in environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            current_episode_reward += reward

            # Store transition in buffer
            # Important: Store the observation *before* the step
            buffer.add(observation, action, reward, terminated or truncated, value, log_prob)

            # Update observation
            observation = next_observation

            # Handle episode end
            done = terminated or truncated
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                # print(f"End ep {len(episode_rewards)}: R={current_episode_reward:.1f}, L={current_episode_length}, Step={global_step}") # Debug print
                observation, info = env.reset() # Reset automatically handles seeding if done correctly in make_env
                current_episode_reward = 0
                current_episode_length = 0

            # Early exit if total timesteps reached during rollout
            if global_step >= config["total_timesteps"]:
                break

        # --- End Rollout Collection --- #
        num_rollouts += 1
        rollout_duration = time.time() - rollout_start_time

        # Compute advantages and returns
        with torch.no_grad():
            # Get value of the last observation
            obs_tensor = torch.as_tensor(observation).unsqueeze(0).to(config["device"])
            features = agent.feature_extractor(obs_tensor.float() / 255.0)
            last_value = agent.critic(features).cpu().numpy() # Shape (1,)

        buffer.compute_returns_and_advantages(last_value, done)

        # --- Update Learning Rate --- #
        # Linear decay based on global steps
        progress = global_step / config["total_timesteps"]
        new_lr = agent.initial_lr * (1.0 - progress)
        # Ensure non-negative LR
        new_lr = max(new_lr, 0.0)
        # Update LR in optimizers
        for param_group in agent.actor_optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in agent.critic_optimizer.param_groups:
            param_group['lr'] = new_lr
        agent.lr = new_lr # Update agent's current lr tracker

        # --- Update Agent --- #
        update_start_time = time.time()
        metrics = agent.learn(buffer)
        update_duration = time.time() - update_start_time

        # --- Logging --- #
        if num_rollouts % config["log_interval"] == 0 and len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards)
            mean_length = np.mean(episode_lengths)
            fps = int(config["buffer_size"] / rollout_duration)
            update_fps = int(config["buffer_size"] / update_duration) if update_duration > 0 else float('inf')
            total_duration = time.time() - start_time

            print(f"-- Rollout {num_rollouts} | Timesteps {global_step}/{config['total_timesteps']} --")
            print(f"  Stats (last 100 ep): Mean Reward: {mean_reward:.2f}, Mean Length: {mean_length:.1f}")
            print(f"  Speed: Rollout FPS: {fps}, Update FPS: {update_fps}")
            print(f"  Total Time: {total_duration:.2f}s")
            # --- TensorBoard Logging --- #
            writer.add_scalar("Charts/mean_episode_reward", mean_reward, global_step)
            writer.add_scalar("Charts/mean_episode_length", mean_length, global_step)
            writer.add_scalar("Speed/rollout_fps", fps, global_step)
            writer.add_scalar("Speed/update_fps", update_fps, global_step)
            writer.add_scalar("Loss/policy_loss", metrics["policy_loss"], global_step)
            writer.add_scalar("Loss/value_loss", metrics["value_loss"], global_step)
            writer.add_scalar("Loss/entropy_loss", metrics["entropy_loss"], global_step)
            writer.add_scalar("Stats/approx_kl", metrics["approx_kl"], global_step)
            writer.add_scalar("Stats/clip_fraction", metrics["clip_fraction"], global_step)
            writer.add_scalar("Config/learning_rate", new_lr, global_step) # Log scheduled learning rate

            # --- Save Best Model --- #
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_save_path = os.path.join(config["save_dir"], "best_model.pth")
                torch.save({
                    'feature_extractor_state_dict': agent.feature_extractor.state_dict(),
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                    'global_step': global_step,
                    'num_rollouts': num_rollouts,
                    'best_mean_reward': best_mean_reward # Save best reward
                }, best_save_path)
                print(f"** New best model saved to {best_save_path} with mean reward: {best_mean_reward:.2f} **")

        # --- Periodic Saving --- #
        if num_rollouts % config["save_interval"] == 0 and num_rollouts > 0: # Avoid saving at rollout 0
            save_path = os.path.join(config["save_dir"], f"ppo_carracing_{global_step}.pth")
            # Save agent state (feature extractor, actor, critic, optimizers)
            torch.save({
                'feature_extractor_state_dict': agent.feature_extractor.state_dict(),
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                'global_step': global_step,
                'num_rollouts': num_rollouts,
                'best_mean_reward': best_mean_reward # Also save best reward here
                # Optionally save buffer state if needed for exact resume
            }, save_path)
            print(f"Model saved to {save_path}")

    print(f"Training finished after {global_step} timesteps.")
    writer.close() # Close the TensorBoard writer
    env.close() 