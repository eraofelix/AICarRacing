"""
Simple test script for CarRacing environment
"""
import gymnasium as gym
import time

def main():
    # Create environment
    env = gym.make('CarRacing-v3', render_mode='human')
    
    # Reset environment
    observation, info = env.reset()
    
    # Run 1000 steps with random actions
    for _ in range(1000):
        # Take random action
        action = env.action_space.sample()
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Small delay to make it visible
        time.sleep(0.01)
        
        # If episode is done, reset
        if terminated or truncated:
            observation, info = env.reset()
    
    # Close environment
    env.close()
    print("Test completed successfully!")

if __name__ == "__main__":
    main() 