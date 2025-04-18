"""
Simple Gym Retro example using Airstriker-Genesis
"""

import retro
import numpy as np

def main():
    # Create the environment with the default game that comes with gym-retro
    env = retro.make(game='Airstriker-Genesis', render_mode="human")
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset the environment to get the first observation
    observation = env.reset()
    
    # Run for a fixed number of timesteps
    total_reward = 0
    steps = 0
    
    # Simple agent: random actions
    while True:
        # Sample random actions
        action = env.action_space.sample()
        
        # Take action and get results
        observation, reward, done, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        # Print the current score from info dictionary
        if 'score' in info:
            print(f"Current score: {info['score']}")
        
        # If we reach a terminal state (episode end), break the loop
        if done or steps > 1000:
            break
            
    print(f"Episode finished after {steps} steps with total reward {total_reward:.2f}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main() 