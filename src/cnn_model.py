import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces

class CNNFeatureExtractor(nn.Module):
    """
    CNN Feature Extractor for processing stacked frames (e.g., from CarRacing).
    Outputs a flat feature vector.
    Uses a simplified architecture for better stability in RL.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__()
        assert isinstance(observation_space, spaces.Box), \
            "CNNFeatureExtractor expects a Box observation space."
        assert len(observation_space.shape) == 3, \
            f"Expected observation shape (k, H, W), got {observation_space.shape}"

        self.features_dim = features_dim
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_obs).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.Dropout(0.2),
            nn.ReLU()
        )

        self._initialize_weights()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.

        :param observations: Batch of observations (Batch, k, H, W)
        :return: Feature vector (Batch, features_dim)
        """
        normalized_obs = observations / 255.0
        cnn_features = self.cnn(normalized_obs)
        features = self.linear(cnn_features)
        return features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0) 