# Development Log: AI Car Racing Agent

This document tracks the development steps for creating a Reinforcement Learning agent to play the Gymnasium `CarRacing-v3` environment.

## Steps Completed

1.  **Initial Setup & Review:**
    *   Reviewed existing project files (`setup_instructions.md`, `conda_environment.yml`, `test_carracing.py`).
    *   Confirmed basic environment setup using Conda.
2.  **Planning & CNN Rationale:**
    *   Discussed the need for a CNN due to image-based observations and an RL algorithm suitable for continuous actions.
    *   Outlined the required components: CNN, RL Agent (PPO suggested), Training/Evaluation Scripts, Utilities.
    *   Created `cnn_explanation.md` detailing the CNN architecture rationale, preprocessing (grayscale, frame stacking), and example structure.
    *   Corrected environment version target from `v2` to `v3`.
3.  **Preprocessing Wrappers:**
    *   Discussed best practices: separating preprocessing from the model.
    *   Created `env_wrappers.py` with Gymnasium wrappers:
        *   `GrayScaleObservation`: Converts RGB (H, W, 3) to Grayscale (H, W).
        *   `FrameStack`: Stacks `k` consecutive grayscale frames into observation (k, H, W).
    *   Resolved `ModuleNotFoundError` for `cv2` by installing `opencv-python`.
    *   Tested wrappers to confirm output shape `(4, 96, 96)`.
4.  **CNN Model Implementation:**
    *   Created `cnn_model.py` with the `CNNFeatureExtractor` class (PyTorch `nn.Module`).
    *   Implemented architecture based on `cnn_explanation.md`.
    *   Included automatic calculation of flattened layer size.
    *   Added input normalization (`/ 255.0`) in the forward pass.
    *   Tested model instantiation and forward pass.
5.  **PPO Agent Structure:**
    *   Discussed RL algorithm options for continuous control (PPO, SAC, TD3) and the implications of procedural track generation (generalization needed).
    *   Selected PPO as the initial algorithm.
    *   Created `ppo_agent.py`.
    *   Implemented `Actor` and `Critic` networks (MLPs taking CNN features).
    *   Added `PPOAgent` class structure (`__init__`, `act`).
    *   Tested agent instantiation and `act` method.
6.  **PPO Update Logic:**
    *   Implemented the `PPOAgent.learn` method in `ppo_agent.py`.
    *   Included calculation of policy loss (clipped surrogate objective), value loss (MSE), and entropy bonus.
    *   Added optimizer steps with gradient clipping.
    *   Added optional KL divergence tracking/early stopping.
    *   Updated agent hyperparameters (vf_coef, ent_coef, etc.).
7.  **Rollout Buffer:**
    *   Created `rollout_buffer.py` with the `RolloutBuffer` class.
    *   Implemented storage for observations, actions, rewards, dones, values, log_probs.
    *   Implemented `compute_returns_and_advantages` using GAE.
    *   Implemented `get_batches` generator for shuffling and yielding mini-batches as tensors.
    *   Tested buffer functionality with dummy data.
8.  **Training Script Enhancement:**
    *   Created initial `train_ppo.py` script structure with config, env setup, agent/buffer creation, and basic training loop.
    *   Updated `PPOAgent.learn` to return detailed training metrics (losses, KL, clip fraction).
    *   Added TensorBoard logging using `SummaryWriter` in `train_ppo.py` for episode stats and agent metrics.
    *   Added `load_checkpoint_path` config option and implemented logic to load checkpoints (model/optimizer state, step counts, best reward).
    *   Implemented saving of the best model (`best_model.pth`) based on achieving a new highest mean episode reward.
    *   Updated periodic checkpoint saving to include the current best mean reward.
9.  **Vectorized Environment Implementation:**
    *   Implemented parallel environment training using Gymnasium's AsyncVectorEnv
    *   Created environment factory function with proper seeding for each environment instance
    *   Added support for simultaneously collecting experience from multiple environments
    *   Modified buffer to handle multiple environments properly
    *   Implemented tracking of rewards and episode lengths across all environments
    *   Optimized RolloutBuffer to process data from vectorized environments
    *   Ensured each environment instance has unique random seeds for track generation
10. **Solving KL Divergence Issues:**
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
11. **Buffer and Episode Tracking Improvements:**
    *   Enhanced RolloutBuffer for better stability:
        *   Added reward clipping (-10 to 10) for numerical stability
        *   Implemented proper per-environment advantage normalization
        *   Added episode tracking to prevent value estimation leaking across episodes
        *   Created shuffling that respects episode boundaries
    *   Added TimeLimit wrapper to env_wrappers.py:
        *   Implemented 400-step episode limit to prevent excessively long episodes
        *   Added proper episode truncation to improve training feedback frequency
12. **Training Output and Performance Optimization:**
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
13. **Dynamic Episode Length Adjustment:**
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
14. **Model Architecture and Training Improvements:**
    *   Enhanced CNN with BatchNorm layers:
        *   Added BatchNorm2d after each convolutional layer before ReLU
        *   Improved training stability and feature normalization
        *   Required starting training from scratch due to architecture changes
    *   Improved policy initialization for better starting behavior:
        *   Used orthogonal initialization for actor's policy head
        *   Helped avoid catastrophic initial actions and balanced exploration
    *   Enabled observation normalization:
        *   Added RunningMeanStd to track observation statistics
        *   Set `use_obs_norm: True` in configuration
        *   Made training more robust to visual/lighting changes
    *   Frozen CNN layers for fine-tuning:
        *   Added code to freeze early CNN layers when loading checkpoints
        *   Prevented "forgetting" of learned visual features
        *   Only applied when continuing from a checkpoint
    *   Increased entropy coefficient:
        *   Raised from 0.005 to 0.01 for better exploration
        *   Helped prevent premature convergence to suboptimal policies
    *   Created fresh training run:
        *   Started from scratch to incorporate architectural changes
        *   Updated logging directories to track new training run separately
        *   Expected sigmoid-shaped learning curve with faster initial progress
15. **Training Stability and Adaptive Parameter Improvements:**
    *   Implemented reward normalization for training stability:
        *   Added RunningMeanStd tracker for rewards
        *   Applied normalization without centering to preserve reward signs
        *   Added clipping (-10 to 10) for numerical stability
    *   Increased buffer size for better policy estimation:
        *   Doubled from 4096 to 8192 steps across all environments
        *   Provided more diverse samples for each policy update 
        *   Better captured full track variations for generalization
    *   Added adaptive entropy boosting during transitions:
        *   Implemented automatic entropy coefficient doubling at episode length increases
        *   Created smooth linear decay back to normal values over 100K steps
        *   Enhanced exploration after environment changes without disrupting existing knowledge
    *   Implemented learning rate warmup after episode length changes:
        *   Added 50K step warmup period starting at 50% of scheduled learning rate
        *   Helped stabilize policy updates during transition to longer episodes
        *   Prevented aggressive policy shifts when encountering new track segments
    *   Enhanced monitoring and visualization:
        *   Added smoothed reward tracking for clearer trend visualization
        *   Implemented entropy coefficient tracking in console output
        *   Created better metrics for evaluating training stability during transitions
    *   Addressed oscillation patterns in reward graph:
        *   Analyzed performance dips occurring at ~400K and ~500K steps
        *   Identified correlation between dips and episode length increases
        *   Designed interventions to reduce variance and improve stability
16. **Extended Training Duration for Complete Learning:**
    *   Expanded total training timesteps from 5M to 15M:
        *   Identified need for longer training based on stable learning pattern
        *   Current 1.2M steps showed promising stability but still below first threshold
        *   Extended horizon to allow full exploration of higher reward thresholds
    *   Implemented training continuation strategy:
        *   Saved checkpoints to enable seamless resumption of training
        *   Modified config to load latest checkpoint while extending training duration
        *   Retained all stability improvements across the extended training
    *   Planned for long-term training management:
        *   Estimated ~50-60 hours total training time with current hardware
        *   Set up periodic checkpoints every 1M steps as safety measures
        *   Created evaluation criteria to determine when to stop training early if performance plateaus
17. **Increased Model Capacity for Better Policy Learning:**
    *   Enhanced CNN feature extractor:
        *   Increased feature dimension from 64 to 256 (4x capacity increase)
        *   Added back third convolutional layer with 64 filters
        *   Improved visual pattern recognition for better track understanding
    *   Expanded neural network architectures:
        *   Doubled hidden layer sizes in both Actor and Critic (128→256)
        *   Added second hidden layer to both networks for deeper policy representation
        *   Maintained weight initialization strategies for stable learning
    *   Optimized GPU resource utilization:
        *   Shifted computational load to underutilized GPU VRAM
        *   Maintained CPU performance for environment simulation
        *   Balanced resource usage to break through performance plateau
    *   Addressed representational bottlenecks:
        *   Identified previous architecture limitations as cause of performance ceiling
        *   Increased model's ability to capture complex driving strategies
        *   Enhanced capacity to handle diverse track layouts and driving scenarios
18. **Addressing Policy Collapse with Domain Randomization:**
    *   Identified and fixed issues causing policy collapse:
        *   Observed reward peaked around 350, then collapsed to ~20 (common with domain randomization)
        *   Diagnosed excessive KL divergence causing frequent early stopping during PPO updates
        *   Determined that BatchNorm layers were making training unstable under domain randomization
    *   Improved CNN architecture resilience:
        *   Removed all BatchNorm layers which caused non-stationarity in feature distribution
        *   Added Dropout2d (10%) after each convolutional layer for better generalization
        *   Added Dropout (20%) in the fully connected layer to combat overfitting
        *   Maintained weight initialization schemes for stable learning
    *   Simplified policy update mechanisms:
        *   Removed value function clipping which prevented necessary large value updates
        *   Replaced with standard MSE loss for more accurate value estimation
        *   Increased value function coefficient from 0.5 to 0.7 to prioritize critic learning
        *   Reduced PPO epochs from 5 to 3 to prevent overtraining on each batch
    *   Enhanced data normalization:
        *   Modified advantage normalization to work globally across environments
        *   Replaced per-environment normalization which caused inconsistent update scales
        *   Ensured more stable policy gradients regardless of environment differences
    *   Implemented gradual domain randomization:
        *   Started with 25% randomization intensity instead of immediate 100%
        *   Created schedule to gradually increase to full randomization over 2M steps
        *   Added monitoring and logging of randomization intensity changes
    *   Added reward shaping for better learning signals:
        *   Created RewardShapingWrapper to provide dense feedback signals
        *   Added small velocity bonus (weight 0.005) to discourage getting stuck
        *   Included progress reward component (weight 0.01) to guide track completion
    *   Modified episode length and threshold strategy:
        *   Increased initial episode length from 500 to 700 steps
        *   Lowered reward thresholds for length increases (75→800, 150→900, 250→1000)
        *   Made thresholds achievable even with domain randomization challenges
    *   Adjusted hyperparameters for stability:
        *   Reduced learning rate from 3e-5 to 1e-5 for smaller, more stable updates
        *   Lowered minimum LR in schedule from 1e-5 to 1e-6 for longer fine-tuning phase
        *   Tightened target KL from 0.025 to 0.015 to detect policy divergence earlier
        *   Extended domain randomization boost/decay periods to 100K steps each

## Next Steps

*   Monitor extended training progress through higher reward thresholds
*   Implement evaluation suite to test model generalization across different tracks
*   Create visualization tools for agent behavior analysis
*   Develop a simple API for deploying trained models in various applications 