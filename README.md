# AI Car Racing

A reinforcement learning project that trains agents to drive in Gymnasium's CarRacing-v3 environment using Proximal Policy Optimization (PPO).

## Project Overview

This project implements:
- A state-of-the-art PPO algorithm with curriculum learning
- Configurable reward shaping and exploration strategies
- Domain randomization for robust policy learning
- Comprehensive evaluation tools

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AICarRacing.git
   cd AICarRacing
   ```

2. Create and activate a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install torch gymnasium[box2d] numpy matplotlib tensorboard
   ```

## Project Structure

```
AICarRacing/
├── src/                   # Core implementation
│   ├── cnn_model.py       # CNN feature extractor architecture
│   ├── ppo_agent.py       # PPO agent implementation
│   ├── env_wrappers.py    # Environment wrappers for preprocessing
│   └── rollout_buffer.py  # Experience storage and advantage computation
├── scripts/               # Training and evaluation scripts
│   ├── train_ppo.py       # Main training script
│   ├── finetune_ppo.py    # Advanced training for domain randomization
│   ├── evaluate_agent.py  # Agent evaluation script
│   └── test_carracing.py  # Quick environment test
├── models/                # Saved model checkpoints
└── logs/                  # Training logs for TensorBoard
```

## Training an Agent

### Basic Training

The main training script uses PPO with curriculum learning:

```bash
python scripts/train_ppo.py
```

Key features:
- Progressive episode length increases based on performance
- Automatic entropy coefficient adjustment for exploration
- Domain randomization scheduled after basic driving skills are learned
- Comprehensive TensorBoard logging

### Monitoring Training

Monitor training progress using TensorBoard:

```bash
tensorboard --logdir=logs
```

View metrics including:
- Mean episode reward
- Episode length
- Policy loss and value loss
- Exploration metrics (entropy, KL divergence)

### Fine-tuning

For advanced domain randomization training:

```bash
python scripts/finetune_ppo.py
```

## Evaluating Agents

To evaluate a trained model:

```bash
python scripts/evaluate_agent.py --model-path models/ppo_carracing/best_model.pth --episodes 10 --render
```

Options:
- `--model-path`: Path to the saved model checkpoint
- `--episodes`: Number of evaluation episodes
- `--seed`: Random seed for deterministic evaluation 
- `--render`: Enable visual rendering (omit for faster evaluation)

## Configuration Options

Key configuration parameters in `train_ppo.py`:

```python
config = {
    # Environment settings
    "max_episode_steps": 300,     # Initial episode length
    "domain_randomize_intensity": 0.0,  # Randomization level
    
    # Training hyperparameters
    "learning_rate": 2e-4,        # Learning rate
    "buffer_size": 4096,          # Experience buffer size
    "batch_size": 64,             # Batch size for updates
    "ppo_epochs": 10,             # Update iterations per batch
    "ent_coef": 0.08,             # Entropy coefficient
    
    # Reward shaping
    "velocity_reward_weight": 0.07,  # Speed incentive
    "progress_reward_weight": 0.03,  # Track progress incentive
}
```

## Environment Details

The Car Racing environment features:
- **Action space**: Steering [-1,1], Gas [0,1], Brake [0,1]
- **Observation**: RGB image (96x96x3), preprocessed to grayscale and stacked
- **Reward**: -0.1 every frame + 1000/N for every track tile visited
- **Episode termination**: All tiles visited or car goes off track

## Training Strategy

The training process follows three key phases:
1. **Fundamentals Phase** (0-1M steps): Build solid driving skills without randomization
2. **Randomization Transition** (1M-2M steps): Introduce and gradually increase domain randomization
3. **Generalization Phase** (2M+ steps): Develop robust policies that work across varied conditions



For more information, visit: [Gymnasium Car Racing](https://gymnasium.farama.org/environments/box2d/car_racing/)
