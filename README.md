# AI Car Racing

A reinforcement learning project that trains an agent to drive a car around a track using the Proximal Policy Optimization (PPO) algorithm.

## What This Program Does

This program uses reinforcement learning to teach a car to drive around a racing track in the CarRacing-v3 environment from Gymnasium (OpenAI Gym's successor). The program:

1. **Trains a PPO agent** to control the car (steering, acceleration, braking)
2. **Evaluates the trained agent's performance** on the racing track
3. **Compares the PPO agent against a random baseline** to measure improvement

The car receives an image of the track as input and must learn to stay on the track while driving as fast as possible.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AICarRacing.git
cd AICarRacing

# Create and activate a virtual environment
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install torch gymnasium[box2d] numpy matplotlib tensorboard pandas
```

## How to Use

### Training an Agent

Train an agent using PPO:

```bash
python scripts/train_ppo.py
```

This will:
- Create an agent with a CNN that processes images from the car's perspective
- Train the agent using PPO, optimizing for rewards from staying on track and driving fast
- Save model checkpoints during training at every 10 rollouts
- Log training progress for visualization with TensorBoard

Monitor training progress:

```bash
tensorboard --logdir=logs
```

### Saved Agents

In BestSavedAgents folder are the two best agents (Evaluated641.pth and Evaluated679.pth) that I was able to train with this script. Lots of room for improvement on both.
Use either as the path if you want to just test the evaluation and see the trained agents without having to build your own.
Both should have a mean reward around 650-700 with the current evaluation settings. 
Both took ~6 hours of training using the train_ppo.py script. Only minor reward shape differences between them. 

[Agent Demos](https://youtube.com/playlist?list=PL896hAqTcFoWO0VgyxWrXyBtKzYf3yxkU&si=Y4v-Y_ToPX4FLTus)

### Evaluating an Agent

Evaluate a trained agent:

```bash
python scripts/evaluate_agent.py
```

To watch the agent drive:

```bash
python scripts/evaluate_agent.py --render
```

### Comparing with a Random Baseline

Compare your trained PPO agent against a random baseline:

```bash
python scripts/compare_agents.py
```

This will:
- Run both your PPO agent and a random agent for several episodes
- Calculate mean reward and episode length for each
- Create bar charts comparing their performance
- Save the comparison results as an image

Additional options:

```bash
python scripts/compare_agents.py --episodes 20 --render
```

## Project Structure

- `src/` - Core implementation files
  - `ppo_agent.py` - The PPO reinforcement learning agent
  - `random_agent.py` - A baseline agent that takes random actions
  - `cnn_model.py` - CNN for processing visual input
  - `env_wrappers.py` - Environment preprocessing (grayscale, frame stacking)
  - `rollout_buffer.py` - Storage for training experiences

- `scripts/` - Training and evaluation scripts
  - `train_ppo.py` - Main training script
  - `evaluate_agent.py` - Evaluates a trained model
  - `compare_agents.py` - Compares PPO agent with random baseline

## Environment Details

The CarRacing-v3 environment:
- **Input**: RGB image (96x96 pixels) from the car's perspective
- **Actions**: Steering (left/right), Gas (accelerate), Brake
- **Reward**: Positive for staying on track, negative per timestep
- **Goal**: Complete the track with maximum reward

For more information, visit: [Gymnasium Car Racing](https://gymnasium.farama.org/environments/box2d/car_racing/)
