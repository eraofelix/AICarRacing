import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces

class CNNFeatureExtractor(nn.Module):
    """
    Convolutional Neural Network (CNN) Feature Extractor.

    Processes stacked input frames (e.g., k grayscale images) from the environment
    observation space and outputs a flattened feature vector.
    Designed for environments like CarRacing where visual input is primary.

    Architecture:
        - Conv2D (16 filters, 8x8 kernel, stride 4)
        - ReLU Activation
        - Dropout (0.1)
        - Conv2D (32 filters, 4x4 kernel, stride 2)
        - ReLU Activation
        - Dropout (0.1)
        - Conv2D (64 filters, 3x3 kernel, stride 1)
        - ReLU Activation
        - Dropout (0.1)
        - Flatten
        - Linear (features_dim)
        - Dropout (0.2)
        - ReLU Activation

    Includes internal normalization of input observations (division by 255.0).
    Uses Kaiming Normal initialization for convolutional and linear layers.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        """
        Initializes the CNN Feature Extractor.

        Args:
            observation_space: The environment observation space (Box). Expected shape (k, H, W).
            features_dim: The desired dimensionality of the output feature vector.
        """
        super().__init__()
        # Validate observation space type and shape
        assert isinstance(observation_space, spaces.Box), \
            f"CNNFeatureExtractor expects a Box observation space, got {type(observation_space)}"
        assert len(observation_space.shape) == 3, \
            f"CNNFeatureExtractor expects observation shape (k, H, W), got {observation_space.shape}"

        self.features_dim = features_dim
        # Number of input channels (k from frame stacking)
        n_input_channels = observation_space.shape[0]

        # Define the convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Dropout2d(0.1), # Dropout after activation

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Flatten(), # Flatten the output for the linear layer
        )

        # Compute the flattened size automatically by doing a dummy forward pass
        with torch.no_grad():
            # Create a dummy observation batch (add batch dimension)
            dummy_obs = torch.as_tensor(observation_space.sample()[None]).float()
            # Pass through CNN (without normalization here, as it's just for shape)
            n_flatten = self.cnn(dummy_obs / 255.0).shape[1] # Get the flattened size

        # Define the final linear layer(s)
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.Dropout(0.2), # Dropout before final activation
            nn.ReLU()
        )

        # Initialize network weights
        self._initialize_weights()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the CNN and linear layers.

        Args:
            observations: Batch of observations (Batch, k, H, W). Assumed to be
                          uint8 tensors or tensors needing normalization.

        Returns:
            Feature vector (Batch, features_dim).
        """
        # Normalize observations to [0, 1] range
        # Assumes input is e.g., uint8 [0, 255] or float needing scaling
        normalized_obs = observations.float() / 255.0
        # Pass through convolutional layers
        cnn_features = self.cnn(normalized_obs)
        # Pass through linear layer(s)
        features = self.linear(cnn_features)
        return features

    def _initialize_weights(self):
        """Initializes weights using Kaiming Normal for Conv/Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Normal initialization for ReLU non-linearity
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # Initialize bias to zero
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0) 