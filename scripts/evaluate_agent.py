import gymnasium as gym
import torch
import numpy as np
import os
import time
import argparse
import matplotlib.pyplot as plt  # Import matplotlib

from src.env_wrappers import GrayScaleObservation, FrameStack, TimeLimit
from src.ppo_agent import PPOAgent

# --- Configuration --- #
config = {
    # Environment settings
    "env_id": "CarRacing-v3",
    "frame_stack": 4,
    "seed": 42,
    "max_episode_steps": 1000,

    # Agent settings - use 256 features for ppo_simple model
    "features_dim": 256,  # This matches the ppo_simple model architecture

    # Evaluation settings
    "n_eval_episodes": 100,
    "render_mode": None,  # Set to "human" to watch, None to run faster

    # Hardware
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_env(env_id, seed, frame_stack, render_mode=None, max_episode_steps=600):
    """Create and wrap the environment for evaluation."""
    env = gym.make(env_id, continuous=True, domain_randomize=False, render_mode=render_mode)
    env.reset(seed=seed + 100)
    env.action_space.seed(seed + 100)

    env = GrayScaleObservation(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = FrameStack(env, frame_stack)
    return env

# --- Main Evaluation Loop --- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/ppo_simple/Evaluated679.pth",
                        help="Path to the trained PPO agent model (.pth file)")
    parser.add_argument("--episodes", type=int, default=config["n_eval_episodes"],
                        help="Number of episodes to run for evaluation (default: 100)")
    parser.add_argument("--seed", type=int, default=config["seed"],
                        help="Random seed for environment reset during evaluation")
    parser.add_argument("--render", action='store_true', default=False,
                        help="Enable rendering (default: False)")
    parser.add_argument("--features-dim", type=int, default=config["features_dim"],
                        help="Feature dimension of the model (default: 256)")

    args = parser.parse_args()
    
    # Update config based on args
    config["features_dim"] = args.features_dim
    render_mode = "human" if args.render else None
    eval_seed = args.seed

    print(f"Using device: {config['device']}")
    print(f"Evaluating model: {args.model_path}")
    print(f"Running for {args.episodes} episodes with seed {eval_seed}")
    print(f"Using features_dim: {config['features_dim']}")
    print(f"Rendering: {'Enabled' if render_mode else 'Disabled'}")

    set_seeds(eval_seed)

    # Create environment
    env = make_env(config["env_id"], eval_seed, config["frame_stack"], render_mode, config["max_episode_steps"])

    # Create Agent with the correct features_dim
    agent = PPOAgent(env.observation_space,
                     env.action_space,
                     features_dim=config["features_dim"],
                     device=config["device"])

    # Load the trained model weights
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        exit()

    checkpoint = torch.load(args.model_path, map_location=config["device"])
    print("Loading model state dicts...")
    try:
        agent.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        if 'critic_state_dict' in checkpoint:
             agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # Set agent to evaluation mode
    agent.feature_extractor.eval()
    agent.actor.eval()
    agent.critic.eval()

    episode_rewards = []
    episode_lengths = []

    for episode in range(args.episodes):
        observation, info = env.reset(seed=eval_seed + episode)
        terminated = False
        truncated = False
        current_episode_reward = 0
        current_episode_length = 0

        # Debug info for first episode
        if episode == 0:
            print(f"Observation shape: {observation.shape}")
            with torch.no_grad():
                try:
                    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=config["device"])
                    print(f"Input tensor shape: {obs_tensor.shape}")
                    features = agent.feature_extractor.cnn(obs_tensor / 255.0)
                    print(f"CNN output shape: {features.shape}")
                except Exception as e:
                    print(f"Error during dummy forward pass: {e}")

        while not (terminated or truncated):
            try:
                # Ensure observation has correct shape with batch dimension
                if len(observation.shape) == 3:  # (C, H, W)
                    observation = np.expand_dims(observation, axis=0)  # Add batch dimension
                
                # Get action from agent
                actions, values, log_probs = agent.act(observation)
                
                # Extract the first action from the batch
                action = actions[0]
                
                # Debug first action in first episode
                if episode == 0 and current_episode_length == 0:
                    print(f"Action batch shape: {actions.shape}")
                    print(f"Single action shape: {action.shape}")
                    print(f"Action values: {action}")
                
                # Step environment
                observation, reward, terminated, truncated, info = env.step(action)
                
                current_episode_reward += reward
                current_episode_length += 1
                
                # Optional delay for smoother rendering
                if render_mode == "human":
                    time.sleep(0.01)
            except Exception as e:
                print(f"Error during episode step: {e}")
                print(f"Action shape: {actions.shape if 'actions' in locals() else 'unknown'}")
                terminated = True

        print(f"Episode {episode + 1}: Reward = {current_episode_reward:.2f}, Length = {current_episode_length}")
        episode_rewards.append(current_episode_reward)
        episode_lengths.append(current_episode_length)

    env.close()

    # Calculate and print summary statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    # --- Calculate Floor and Ceiling ---
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)

    print("\n--- Evaluation Summary ---")
    print(f"Number of episodes: {args.episodes}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Min Reward (Floor): {min_reward:.2f}")
    print(f"Max Reward (Ceiling): {max_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.1f} +/- {std_length:.1f}")
    
    # Categorical performance assessment
    if mean_reward >= 900:
        performance = "Exceptional"
    elif mean_reward >= 800:
        performance = "Excellent"
    elif mean_reward >= 700:
        performance = "Very Good"
    elif mean_reward >= 500:
        performance = "Good"
    elif mean_reward >= 300:
        performance = "Decent"
    elif mean_reward >= 200:
        performance = "Fair"
    else:
        performance = "Needs Improvement"
        
    print(f"Performance Rating: {performance}")

    # --- Generate Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label='Episode Reward', marker='o', linestyle='-', markersize=4)
    plt.axhline(mean_reward, color='r', linestyle='--', label=f'Mean Reward ({mean_reward:.2f})')
    plt.axhline(min_reward, color='g', linestyle=':', label=f'Min Reward (Floor) ({min_reward:.2f})')
    plt.axhline(max_reward, color='b', linestyle=':', label=f'Max Reward (Ceiling) ({max_reward:.2f})')
    plt.title(f'Evaluation Results ({args.episodes} Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"evaluation_rewards_{os.path.basename(args.model_path).replace('.pth', '')}_{args.episodes}ep.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

    # Optionally display the plot
    plt.show() # Uncomment this line if you want the plot to pop up automatically 