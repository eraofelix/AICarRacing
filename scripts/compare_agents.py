#!/usr/bin/env python

import subprocess
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def run_evaluation(random=False, episodes=10, seed=42, render=False, max_steps=1000):
    """Run the evaluation script with the specified parameters."""
    cmd = ["python", "scripts/evaluate_agent.py", 
           "--episodes", str(episodes),
           "--seed", str(seed),
           "--max-steps", str(max_steps)]
    
    if random:
        cmd.append("--random")
    
    if render:
        cmd.append("--render")
    
    # Run the evaluation script and capture output
    print(f"Running {'random' if random else 'PPO'} agent evaluation...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output for debugging
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    # Parse the output to extract metrics
    mean_reward = None
    mean_length = None
    
    for line in result.stdout.split('\n'):
        if "Mean Reward:" in line:
            # Extract mean reward using regex pattern for improved robustness
            match = re.search(r"Mean Reward:\s+([-+]?\d*\.\d+)", line)
            if match:
                mean_reward = float(match.group(1))
        elif "Mean Episode Length:" in line:
            # Extract mean episode length
            match = re.search(r"Mean Episode Length:\s+([-+]?\d*\.\d+)", line)
            if match:
                mean_length = float(match.group(1))
    
    return {
        "agent": "Random" if random else "PPO",
        "mean_reward": mean_reward,
        "mean_length": mean_length,
        "episodes": episodes,
        "seed": seed
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare PPO agent vs Random agent on CarRacing-v3")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes for each agent evaluation (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for evaluation (default: 42)")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during evaluation (default: False)")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum steps per episode (default: 1000)")
    return parser.parse_args()

def plot_comparison(results):
    """Create a bar chart comparing agent performance."""
    agents = [r["agent"] for r in results]
    rewards = [r["mean_reward"] for r in results]
    lengths = [r["mean_length"] for r in results]
    
    # Set up a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot mean rewards
    bars1 = ax1.bar(agents, rewards, color=['skyblue', 'lightgreen'])
    ax1.set_title('Mean Reward Comparison')
    ax1.set_ylabel('Reward')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        if height < 0:
            va = 'top'
            offset = -5
        else:
            va = 'bottom'
            offset = 5
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va=va)
    
    # Plot mean episode lengths
    bars2 = ax2.bar(agents, lengths, color=['skyblue', 'lightgreen'])
    ax2.set_title('Mean Episode Length Comparison')
    ax2.set_ylabel('Steps')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Add overall title and adjust layout
    plt.suptitle(f'PPO vs Random Agent Performance ({results[0]["episodes"]} episodes)', fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'agent_comparison_{results[0]["episodes"]}ep_seed{results[0]["seed"]}.png')
    print(f'Comparison plot saved as: agent_comparison_{results[0]["episodes"]}ep_seed{results[0]["seed"]}.png')
    
    # Show the plot
    plt.show()

def main():
    """Main function to run comparison."""
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs("models/ppo_carracing", exist_ok=True)
    
    # Run evaluations
    results = []
    
    # Run PPO agent evaluation
    ppo_results = run_evaluation(random=False, episodes=args.episodes, 
                              seed=args.seed, render=args.render,
                              max_steps=args.max_steps)
    results.append(ppo_results)
    
    # Run random agent evaluation
    random_results = run_evaluation(random=True, episodes=args.episodes, 
                                 seed=args.seed, render=args.render,
                                 max_steps=args.max_steps)
    results.append(random_results)
    
    # Print summary table
    df = pd.DataFrame(results)
    print("\n--- Comparison Results ---")
    print(df.to_string(index=False))
    
    # Plot comparison
    plot_comparison(results)

if __name__ == "__main__":
    main() 