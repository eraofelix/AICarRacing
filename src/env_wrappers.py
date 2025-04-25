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
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return obs

class TimeLimit(gym.Wrapper):
    """
    Limits the episode length to a specified number of steps.
    Applies on top of the environment's own time limit.
    """
    def __init__(self, env, max_episode_steps=1000):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
            
        return observation, reward, terminated, truncated, info

class FrameStack(gym.ObservationWrapper):
    """
    Stack k last frames.
    """
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)

        assert len(env.observation_space.shape) == 2, \
               f"FrameStack expects input shape (H, W), got {env.observation_space.shape}"
        stacked_shape = (k,) + env.observation_space.shape
        self.observation_space = Box(
            low=0, high=255, shape=stacked_shape, dtype=env.observation_space.dtype
        )

    def observation(self, observation):
        """Adds the observation to the deque and returns the stacked frames."""
        self.frames.append(observation)
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
        return np.stack(self.frames, axis=0) 