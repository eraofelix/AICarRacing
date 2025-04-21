import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces

class CNNFeatureExtractor(nn.Module):
    """
    CNN Feature Extractor for processing stacked frames (e.g., from CarRacing).
    Outputs a flat feature vector.
    Uses architecture inspired by Nature DQN + BatchNorm.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__()
        assert isinstance(observation_space, spaces.Box), \
            "CNNFeatureExtractor expects a Box observation space."
        # Check if the observation space is images
        # Assuming shape (k, H, W) where k=frame stack
        assert len(observation_space.shape) == 3, \
            f"Expected observation shape (k, H, W), got {observation_space.shape}"

        self.features_dim = features_dim
        n_input_channels = observation_space.shape[0] # Number of stacked frames

        # Define CNN layers (Nature DQN style)
        self.cnn = nn.Sequential(
            # Input shape: (Batch, n_input_channels, 96, 96)
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Shape: (Batch, 32, 23, 23) -> (96-8)/4 + 1 = 22+1 = 23

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Shape: (Batch, 64, 11, 11) -> (23-4)/2 + 1 = 9.5+1 = 10.5 -> 11 (check PyTorch conv docs? Yes, floor((N+2P-K)/S)+1)
            # (23 + 2*0 - 8)/4 + 1 = 23
            # (23 + 2*0 - 4)/2 + 1 = 10.5 -> floor(10.5)+1 = 11

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Shape: (Batch, 64, 9, 9) -> (11-3)/1 + 1 = 8+1 = 9

            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            # Create a dummy observation matching the space shape
            dummy_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_obs).shape[1]
            print(f"CNN output flattened size: {n_flatten}")

        # Define the final linear layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

        # Initialize weights (optional but can help)
        self._initialize_weights()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.

        :param observations: Batch of observations (Batch, k, H, W)
        :return: Feature vector (Batch, features_dim)
        """
        # Normalize observations (assuming they are uint8 pixel values 0-255)
        # If observations are already float [0,1], remove / 255.0
        # Let's keep the normalization here for now.
        normalized_obs = observations / 255.0
        cnn_features = self.cnn(normalized_obs)
        features = self.linear(cnn_features)
        return features

    def _get_conv_out(self, shape):
        # This method might become inaccurate with BatchNorm, better to use dummy forward pass
        # Let's comment it out or remove it if unused elsewhere
        pass
        # o = self.cnn[0](torch.zeros(1, *shape)) # Conv1
        # o = self.cnn[3](torch.zeros(1, *o.shape[1:])) # Conv2
        # o = self.cnn[6](torch.zeros(1, *o.shape[1:])) # Conv3
        # return int(np.prod(o.size()))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

# Example Usage (for testing the extractor)
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