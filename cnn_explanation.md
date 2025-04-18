# CNN Architecture for CarRacing-v3

This document explains the rationale and a typical architecture for a Convolutional Neural Network (CNN) used as the feature extractor within a Reinforcement Learning agent for the Gymnasium `CarRacing-v3` environment.

## 1. Why a CNN?

The `CarRacing-v3` environment provides observations as 96x96x3 (RGB) pixel images. Processing this raw visual data effectively is crucial for the agent to understand the current state and make informed decisions.

-   **Spatial Hierarchies:** Images contain spatial structures (edges, corners, textures, objects). CNNs are specifically designed to exploit this spatial locality and learn hierarchical representations automatically. Early layers detect simple features (like lines), and deeper layers combine these to recognize more complex patterns (like curves in the track, the car itself).
-   **Parameter Sharing:** CNNs use the same filters (kernels) across different parts of the image. This drastically reduces the number of parameters compared to a fully connected network on raw pixels, making the model more efficient to train and less prone to overfitting.
-   **Translation Invariance:** Due to parameter sharing and often pooling layers, CNNs have a degree of translation invariance – they can recognize a feature (like a turn) regardless of its exact position in the image.

A standard feed-forward neural network (Multilayer Perceptron) would require flattening the image into a single huge vector, losing all spatial information and resulting in an intractably large number of parameters.

## 2. Input Preprocessing

Raw RGB frames are usually preprocessed before being fed into the CNN:

1.  **Grayscaling:** Convert the 96x96x3 RGB image to a 96x96x1 grayscale image. Color information is generally not critical for driving in this environment; track edges, the car, and lane markings are distinct in grayscale. This reduces the input channels by a factor of 3, speeding up computation.
2.  **Frame Stacking:** A single frame provides only positional information. To infer dynamics like velocity and acceleration (is the car turning? how fast?), we need information across time. A common technique is to stack the last `k` consecutive grayscale frames (e.g., `k=4`) into a single multi-channel input tensor. The input shape to the CNN then becomes `(k, 96, 96)` or `(batch_size, k, 96, 96)`.

## 3. Core CNN Layers

A typical CNN architecture for this kind of task follows a pattern inspired by successful models in computer vision and RL (like the one used in the DeepMind Nature DQN paper, adapted for the input size):

-   **Convolutional Layers (`nn.Conv2d` in PyTorch):**
    -   Apply a set of learnable filters (kernels) across the input image (or feature maps from the previous layer).
    -   Each filter specializes in detecting a specific pattern (e.g., horizontal edge, vertical edge, specific curve).
    -   Key Parameters:
        -   `in_channels`: Number of input channels (e.g., `k=4` for the first layer).
        -   `out_channels`: Number of filters to apply (determines the number of output feature maps).
        -   `kernel_size`: Spatial dimensions of the filter (e.g., `8x8`, `4x4`, `3x3`). Larger kernels capture larger patterns initially.
        -   `stride`: How many pixels the filter shifts at a time. A stride > 1 downsamples the output spatial dimensions.
        -   `padding`: Adds pixels around the border to control the output size.
-   **Activation Functions (`nn.ReLU` in PyTorch):**
    -   Applied element-wise after convolutions.
    -   Introduce non-linearity, allowing the network to learn complex relationships beyond simple linear combinations. Rectified Linear Unit (ReLU) is the most common choice due to its simplicity and effectiveness.
-   **Pooling Layers (`nn.MaxPool2d` in PyTorch):**
    -   Optional, but common.
    -   Downsample the feature maps, reducing spatial resolution.
    -   Makes the representation more robust to small translations and distortions.
    -   Max pooling takes the maximum value within a small window, retaining the strongest activation for a feature in that region.

## 4. Example Architecture (Conceptual)

Input: `(Batch, 4, 96, 96)` (Stacked Grayscale Frames)

1.  **Conv1:** `nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)` -> `ReLU`
    -   *Purpose:* Detect basic edges and features with a large receptive field and significant downsampling.
    -   *Output Shape (approx):* `(Batch, 32, 23, 23)`
2.  **Conv2:** `nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)` -> `ReLU`
    -   *Purpose:* Combine features from Conv1, smaller receptive field, less downsampling.
    -   *Output Shape (approx):* `(Batch, 64, 10, 10)`
3.  **Conv3:** `nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)` -> `ReLU`
    -   *Purpose:* Further combine features with minimal downsampling, deeper feature extraction.
    -   *Output Shape (approx):* `(Batch, 64, 8, 8)`
4.  **Flatten:** `nn.Flatten()`
    -   *Purpose:* Convert the final 3D feature maps `(Channels, Height, Width)` into a 1D vector suitable for fully connected layers.
    -   *Output Shape:* `(Batch, 64 * 8 * 8)` = `(Batch, 4096)`

## 5. Output

The output of the CNN component is a flat feature vector (e.g., size 4096 in the example above). This vector is a compressed representation of the salient visual information from the input frames.

## 6. Connection to RL Agent

This feature vector is then passed as input to the subsequent parts of the RL agent:

-   **Actor Network:** Uses the features to decide which action to take (e.g., predicts steering, gas, brake values). Typically involves one or more fully connected layers (`nn.Linear`).
-   **Critic Network:** Uses the features to estimate the value of the current state (expected future rewards). Also typically involves fully connected layers.

The CNN layers themselves are trained end-to-end along with the Actor and Critic networks via the RL algorithm's loss function (backpropagation). The gradients flow back from the RL objectives through the fully connected layers and then through the convolutional layers, tuning the filters to extract features that are most relevant for maximizing rewards. 