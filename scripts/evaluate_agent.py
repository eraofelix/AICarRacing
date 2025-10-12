import gymnasium as gym
import torch
import numpy as np
import os
import time
import argparse
import matplotlib.pyplot as plt
import typing

from train_ppo import GrayScaleObservation, FrameStack, TimeLimit
from train_ppo import PPOAgent

config = {
    # Environment settings
    "env_id": "CarRacing-v3",
    "frame_stack": 4,
    "seed": 42, # Seed used for all evaluation graphs
    "max_episode_steps": 1000, # Max steps per evaluation episode
    "n_eval_episodes": 10,        # Number of episodes to run for evaluation (100 for all evaluation graphs)
    "render_mode": "human",       # Set to "human" to watch the agent play
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_env(env_id: str, seed: int, frame_stack: int, render_mode: typing.Union[str, None] = None, max_episode_steps: int = 1000):
    env = gym.make(env_id, continuous=True, domain_randomize=False, render_mode=render_mode)
    env.reset(seed=seed + 100) # Use a different seed offset for evaluation
    env.action_space.seed(seed + 100)
    env = GrayScaleObservation(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = FrameStack(env, frame_stack)
    return env

if __name__ == "__main__":
    HARDCODED_MODEL_PATH = "/Users/kun/code/AICarRacing/models/ppo_carracing_20251012_150026/checkpoint_5683200.pth"
    render_mode = "human"
    set_seeds(config["seed"])
    env = make_env(config["env_id"], config["seed"], config["frame_stack"], render_mode, config["max_episode_steps"])
    agent = PPOAgent(env.observation_space,
                        env.action_space,
                        config=config, # Pass the config dictionary
                        device=config["device"])
    checkpoint = torch.load(HARDCODED_MODEL_PATH, map_location=config["device"]) # Use hardcoded path
    agent.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    if 'critic_state_dict' in checkpoint:
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.feature_extractor.eval()
    agent.actor.eval()
    agent.critic.eval() # Set critic to eval mode as well

    episode_rewards = []
    episode_lengths = []

    for episode in range(config["n_eval_episodes"]):
        observation, info = env.reset(seed=config["seed"] + episode)
        terminated = False
        truncated = False
        current_episode_reward = 0
        current_episode_length = 0

        while not (terminated or truncated):
            with torch.no_grad():
                actions, _, _ = agent.act(observation)
                action = actions[0] # actions is shape (1, action_dim), take the first element

            observation, reward, terminated, truncated, info = env.step(action)
            current_episode_reward += reward
            current_episode_length += 1

            if render_mode == "human":
                time.sleep(0.01)

        print(f"Episode {episode + 1}/{config['n_eval_episodes']}: Reward = {current_episode_reward:.2f}, Length = {current_episode_length}")