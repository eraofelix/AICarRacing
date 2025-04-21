import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from collections import deque
import cv2

class GrayScaleObservation(gym.ObservationWrapper):
    """
    Convert the observation to grayscale.
    Reads observation space shape from wrapped env.
    """
    def __init__(self, env):
        super().__init__(env)
        assert len(env.observation_space.shape) == 3 and env.observation_space.shape[2] == 3, \
               f"Expected RGB image shape (H, W, 3), got {env.observation_space.shape}"

        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, obs):
        """Converts the RGB observation to grayscale."""
        # Convert to grayscale using cv2, keep shape (H, W)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Add channel dimension: (H, W) -> (H, W, 1)
        # This might be removed if FrameStack handles it, but let's keep it for consistency for now
        # obs = np.expand_dims(obs, -1)
        return obs

class FrameStack(gym.ObservationWrapper):
    """
    Stack k last frames.

    Returns lazy array, which is much more memory efficient.
    See Also
    --------
    stable_baselines3.common.vec_env.vec_frame_stack.LazyFrames

    :param env: Environment to wrap
    :param k: Number of frames to stack
    """
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)

        # The observation space is now k frames stacked
        # Assumes the input observation space has shape (H, W) from grayscale
        assert len(env.observation_space.shape) == 2, \
               f"FrameStack expects input shape (H, W), got {env.observation_space.shape}"
        stacked_shape = (k,) + env.observation_space.shape
        self.observation_space = Box(
            low=0, high=255, shape=stacked_shape, dtype=env.observation_space.dtype
        )

    def observation(self, observation):
        """Adds the observation to the deque and returns the stacked frames."""
        self.frames.append(observation)
        # Return a lazy frame object or just stack numpy arrays for simplicity first
        # Using np.array is less memory efficient but simpler to start with
        return self._get_ob()

    def reset(self, **kwargs):
        """Clear buffer and re-fill it with the first observation."""
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob(), info

    def _get_ob(self):
        """Get the stacked frames from the deque."""
        assert len(self.frames) == self.k, "Deque length mismatch"
        # Stack along the first dimension (channel dimension for CNN)
        return np.stack(self.frames, axis=0)

# Example Usage (for testing)
if __name__ == '__main__':
    env = gym.make("CarRacing-v3", continuous=True, domain_randomize=False) # Use v3

    print(f"Original Observation space: {env.observation_space}")
    print(f"Original Action space: {env.action_space}")

    # Apply wrappers
    env = GrayScaleObservation(env)
    print(f"After Grayscale: {env.observation_space}")

    num_stack = 4
    env = FrameStack(env, num_stack)
    print(f"After FrameStack: {env.observation_space}")

    # Test reset
    obs, info = env.reset()
    print(f"Reset observation shape: {obs.shape}, dtype: {obs.dtype}")

    # Test step
    action = env.action_space.sample() # Sample a random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step observation shape: {obs.shape}, dtype: {obs.dtype}")

    env.close()
    print("Wrappers seem functional.") 