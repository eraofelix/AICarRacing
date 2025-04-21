import gymnasium as gym
import torch
import numpy as np
import os
import time
import argparse # To specify model path easily

# Import custom modules
from src.env_wrappers import GrayScaleObservation, FrameStack
from src.ppo_agent import PPOAgent # Assuming PPOAgent has load functionality or we load state dicts

# --- Configuration --- #
# Most settings should match the training config the model was trained with
config = {
    # Environment settings
    "env_id": "CarRacing-v3",
    "frame_stack": 4,
    "seed": 42, # Use a different seed than training for fair evaluation if desired

    # Agent settings (must match trained model)
    "features_dim": 256, # Output dim of CNN

    # Evaluation settings
    "n_eval_episodes": 10,
    "render_mode": "human", # Set to "human" to watch, None to run faster

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

def make_env(env_id, seed, frame_stack, render_mode="human"):
    """Create and wrap the environment for evaluation."""
    # Use a specific render_mode, default to human
    env = gym.make(env_id, continuous=True, domain_randomize=False, render_mode=render_mode)
    # Important: Set seed for action space and observation space BEFORE wrapping
    # Use a potentially different seed for evaluation runs
    env.reset(seed=seed + 100) # Offset seed from training
    env.action_space.seed(seed + 100)

    env = GrayScaleObservation(env)
    env = FrameStack(env, frame_stack)
    return env

# --- Main Evaluation Loop --- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/ppo_carracing/best_model.pth",
                        help="Path to the trained PPO agent model (.pth file)")
    parser.add_argument("--episodes", type=int, default=config["n_eval_episodes"],
                        help="Number of episodes to run for evaluation")
    parser.add_argument("--seed", type=int, default=config["seed"],
                        help="Random seed for environment reset during evaluation")
    parser.add_argument("--render", action='store_true', default=False,
                        help="Enable rendering (default: False)")

    args = parser.parse_args()

    render_mode = "human" if args.render else None
    eval_seed = args.seed

    print(f"Using device: {config['device']}")
    print(f"Evaluating model: {args.model_path}")
    print(f"Running for {args.episodes} episodes with seed {eval_seed}")
    print(f"Rendering: {'Enabled' if render_mode else 'Disabled'}")

    set_seeds(eval_seed)

    # Create environment
    env = make_env(config["env_id"], eval_seed, config["frame_stack"], render_mode)

    # Create Agent (with dummy parameters, weights will be loaded)
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
        # Critic is not strictly needed for acting, but load if present
        if 'critic_state_dict' in checkpoint:
             agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print("Model weights loaded successfully.")
    except KeyError as e:
        print(f"Error loading state dict: Missing key {e}")
        exit()
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
        observation, info = env.reset(seed=eval_seed + episode) # Use different seed per episode eval
        terminated = False
        truncated = False
        current_episode_reward = 0
        current_episode_length = 0

        while not (terminated or truncated):
            # Get action from agent (no gradients needed)
            # Note: agent.act currently returns action, value, log_prob. We only need action.
            action, _, _ = agent.act(observation) # Get deterministic action? PPO usually samples.

            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)

            current_episode_reward += reward
            current_episode_length += 1

            # Optional small delay for smoother rendering
            if render_mode == "human":
                time.sleep(0.01)

        print(f"Episode {episode + 1}: Reward = {current_episode_reward:.2f}, Length = {current_episode_length}")
        episode_rewards.append(current_episode_reward)
        episode_lengths.append(current_episode_length)

    env.close()

    # Calculate and print summary statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print("\n--- Evaluation Summary ---")
    print(f"Number of episodes: {args.episodes}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.1f} +/- {std_length:.1f}") 