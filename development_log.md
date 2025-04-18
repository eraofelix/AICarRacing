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

## Next Steps

*   Develop the main training script (`train_ppo.py`).
*   Develop an evaluation script (`evaluate_agent.py`).
*   Implement logging (e.g., TensorBoard) in the training script and potentially `PPOAgent.learn`. 