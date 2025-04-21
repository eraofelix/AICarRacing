import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces

class CNNFeatureExtractor(nn.Module):
    """
    Convolutional Neural Network for extracting features from image observations.
    Input is expected to be a stack of k frames (e.g., (batch_size, k, 96, 96)).
    Based on the Nature DQN architecture adaptation.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__()

        assert isinstance(observation_space, spaces.Box), \
            "observation_space must be a gymnasium Box space"
        assert len(observation_space.shape) == 3, \
            f"Expected observation shape (k, H, W), got {observation_space.shape}"

        k, height, width = observation_space.shape

        # Define the convolutional layers
        self.cnn = nn.Sequential(
            # Input: (Batch, k, 96, 96)
            nn.Conv2d(in_channels=k, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Output: (Batch, 32, 23, 23) approx
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Output: (Batch, 64, 10, 10) approx
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            # Output: (Batch, 64, 8, 8) approx
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            # Create a dummy input tensor with the correct shape
            # Requires Batch dimension, adds it using unsqueeze(0)
            dummy_input = torch.as_tensor(observation_space.sample()[None]).float()
            # Pass through CNN
            n_flatten = self.cnn(dummy_input).shape[1]

        # Define the linear layer head to get the final features_dim
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.

        :param observations: Input tensor (Batch, k, H, W)
        :return: Feature tensor (Batch, features_dim)
        """
        # Observations are typically uint8, convert to float and normalize
        # Normalization (dividing by 255) is important for NN stability
        observations = observations.float() / 255.0
        features = self.cnn(observations)
        return self.linear(features)

    @property
    def features_dim(self) -> int:
        """Returns the dimension of the output features."""
        return self._features_dim

# Example Usage (for testing)
if __name__ == '__main__':
    # Define a sample observation space like the one from FrameStack
    k = 4
    height = 96
    width = 96
    obs_space = spaces.Box(low=0, high=255, shape=(k, height, width), dtype=np.uint8)

    # Instantiate the CNN model
    features_dim_output = 256 # Example desired output feature dimension
    cnn_model = CNNFeatureExtractor(observation_space=obs_space, features_dim=features_dim_output)
    print(cnn_model)
    print(f"Output features dimension: {cnn_model.features_dim}")

    # Create a dummy batch of observations
    batch_size = 5
    dummy_obs = torch.as_tensor(np.array([obs_space.sample() for _ in range(batch_size)])).float()
    print(f"Input batch shape: {dummy_obs.shape}")

    # Perform a forward pass
    features = cnn_model(dummy_obs)
    print(f"Output features shape: {features.shape}")

    # Check output shape matches expected features_dim
    assert features.shape == (batch_size, features_dim_output), "Output shape mismatch!"
    print("CNN model seems functional.") 