# Development Log: AI Car Racing Agent

This document tracks the development steps for creating a Reinforcement Learning agent to play the Gymnasium `CarRacing-v3` environment.

## Steps Completed

1.  **Initial Setup & Review (2024-08-01):**
    *   Reviewed existing project files (`setup_instructions.md`, `conda_environment.yml`, `test_carracing.py`).
    *   Confirmed basic environment setup using Conda.
2.  **Planning & CNN Rationale (2024-08-01):**
    *   Discussed the need for a CNN due to image-based observations and an RL algorithm suitable for continuous actions.
    *   Outlined the required components: CNN, RL Agent (PPO suggested), Training/Evaluation Scripts, Utilities.
    *   Created `cnn_explanation.md` detailing the CNN architecture rationale, preprocessing (grayscale, frame stacking), and example structure.
    *   Corrected environment version target from `v2` to `v3`.
3.  **Preprocessing Wrappers (2024-08-01):**
    *   Discussed best practices: separating preprocessing from the model.
    *   Created `env_wrappers.py` with Gymnasium wrappers:
        *   `GrayScaleObservation`: Converts RGB (H, W, 3) to Grayscale (H, W).
        *   `FrameStack`: Stacks `k` consecutive grayscale frames into observation (k, H, W).
    *   Resolved `ModuleNotFoundError` for `cv2` by installing `opencv-python`.
    *   Tested wrappers to confirm output shape `(4, 96, 96)`.
4.  **CNN Model Implementation (2024-08-01):**
    *   Created `cnn_model.py` with the `CNNFeatureExtractor` class (PyTorch `nn.Module`).
    *   Implemented architecture based on `cnn_explanation.md`.
    *   Included automatic calculation of flattened layer size.
    *   Added input normalization (`/ 255.0`) in the forward pass.
    *   Tested model instantiation and forward pass.
5.  **PPO Agent Structure (2024-08-01):**
    *   Discussed RL algorithm options for continuous control (PPO, SAC, TD3) and the implications of procedural track generation (generalization needed).
    *   Selected PPO as the initial algorithm.
    *   Created `ppo_agent.py`.
    *   Implemented `Actor` and `Critic` networks (MLPs taking CNN features).
    *   Added `PPOAgent` class structure (`__init__`, `act`).
    *   Tested agent instantiation and `act` method.
6.  **PPO Update Logic (2024-08-01):**
    *   Implemented the `PPOAgent.learn` method in `ppo_agent.py`.
    *   Included calculation of policy loss (clipped surrogate objective), value loss (MSE), and entropy bonus.
    *   Added optimizer steps with gradient clipping.
    *   Added optional KL divergence tracking/early stopping.
    *   Updated agent hyperparameters (vf_coef, ent_coef, etc.).
7.  **Rollout Buffer (2024-08-01):**
    *   Created `rollout_buffer.py` with the `RolloutBuffer` class.
    *   Implemented storage for observations, actions, rewards, dones, values, log_probs.
    *   Implemented `compute_returns_and_advantages` using GAE.
    *   Implemented `get_batches` generator for shuffling and yielding mini-batches as tensors.
    *   Tested buffer functionality with dummy data.
8.  **Training Script Enhancement (2024-08-01):**
    *   Created initial `train_ppo.py` script structure with config, env setup, agent/buffer creation, and basic training loop.
    *   Updated `PPOAgent.learn` to return detailed training metrics (losses, KL, clip fraction).
    *   Added TensorBoard logging using `SummaryWriter` in `train_ppo.py` for episode stats and agent metrics.
    *   Added `load_checkpoint_path` config option and implemented logic to load checkpoints (model/optimizer state, step counts, best reward).
    *   Implemented saving of the best model (`best_model.pth`) based on achieving a new highest mean episode reward.
    *   Updated periodic checkpoint saving to include the current best mean reward.
9.  **Vectorized Environment Implementation (2024-08-02):**
    *   Implemented parallel environment training using Gymnasium's AsyncVectorEnv
    *   Created environment factory function with proper seeding for each environment instance
    *   Added support for simultaneously collecting experience from multiple environments
    *   Modified buffer to handle multiple environments properly
    *   Implemented tracking of rewards and episode lengths across all environments
    *   Optimized RolloutBuffer to process data from vectorized environments
    *   Ensured each environment instance has unique random seeds for track generation
10. **Solving KL Divergence Issues (2024-08-02):**
    *   Identified high KL divergence causing early stopping during training.
    *   Simplified CNN architecture by:
        *   Reducing feature dimension from 256 to 64
        *   Removing BatchNorm layers for better stability
        *   Eliminating one convolutional layer
        *   Decreasing filter counts (32→16, 64→32)
    *   Modified PPO algorithm parameters:
        *   Reduced learning rate from 3e-4 to 1e-4
        *   Decreased clip epsilon from 0.2 to 0.1
        *   Reduced optimization epochs from 10 to 5
        *   Set target KL threshold to 0.02
        *   Implemented more conservative KL calculation using squared differences
    *   Improved policy network:
        *   Increased initial action standard deviation from 0.5 to 1.0
        *   Switched from fixed standard deviation to learned (state-dependent)
        *   Reduced ratio clipping upper bound from 10.0 to 5.0
        *   Added value function clipping for stability
    *   Optimized learning rate schedule:
        *   Implemented cosine annealing instead of linear decay
        *   Adjusted warmup phase to start from 30% of learning rate instead of 10%
        *   Reduced weight decay from 1e-4 to 1e-5
11. **Buffer and Episode Tracking Improvements (2024-08-02):**
    *   Enhanced RolloutBuffer for better stability:
        *   Added reward clipping (-10 to 10) for numerical stability
        *   Implemented proper per-environment advantage normalization
        *   Added episode tracking to prevent value estimation leaking across episodes
        *   Created shuffling that respects episode boundaries
    *   Added TimeLimit wrapper to env_wrappers.py:
        *   Implemented 400-step episode limit to prevent excessively long episodes
        *   Added proper episode truncation to improve training feedback frequency
12. **Training Output and Performance Optimization (2024-08-02):**
    *   Added debugging to verify episode completion:
        *   Implemented manual episode tracking when vectorized environments don't provide proper info
        *   Added detailed logging of episode terminations and truncations
    *   Optimized for hardware utilization:
        *   Increased parallel environments from 4 to 8 to fully utilize CPU (Ryzen 7 7800X3D)
        *   Doubled buffer size from 2048 to 4096 to maintain steps per environment
        *   Successfully achieved 100% CPU utilization at ~74°C with good cooling
    *   Documented the need to increase episode length later in training:
        *   Plan to gradually increase max steps from 400 to 600, 800, and finally 1000
        *   Strategy to balance early feedback with later full-track navigation
13. **Dynamic Episode Length Adjustment (2024-08-02):**
    *   Implemented automatic episode length increases based on reward thresholds:
        *   Added configuration parameters for reward-based episode length progression
        *   Created thresholds: 200→500 steps, 400→700 steps, 600→900 steps, 800→1000 steps
        *   Implemented environment recreation when thresholds are reached
    *   Enhanced checkpoint system:
        *   Updated checkpoint saving to include full configuration state
        *   Added checkpoint loading logic to restore episode length settings
        *   Ensured environments are recreated with proper episode lengths on load
    *   Added monitoring and logging:
        *   Created TensorBoard tracking for episode length changes
        *   Added console outputs for episode length updates
        *   Included current episode length in regular logging output

## Next Steps

*   Monitor training progress with dynamic episode lengths
*   Develop an evaluation script (`evaluate_agent.py`).
*   Further refine hyperparameters based on training results.
*   Implement model loading and visualization for testing. 