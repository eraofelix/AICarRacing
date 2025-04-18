# AI Car Racing

A project using Gymnasium's Car Racing environment for AI training.

## Setup

1. Activate the virtual environment:
   ```
   # On Windows
   .\venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install gymnasium[box2d] numpy matplotlib
   ```

## Available Scripts

- `car_racing_test.py`: Simple test that runs random actions in the environment
- `ai_driver.py`: Template for creating an AI agent to drive the car

## Running the Environment

To test if the environment is working properly:
```
python car_racing_test.py
```

To run the agent template:
```
python ai_driver.py
```

## About the Environment

The Car Racing environment features:
- Action space: Steering [-1,1], Gas [0,1], Brake [0,1]
- Observation: RGB image (96x96x3)
- Reward: -0.1 every frame + 1000/N for every track tile visited
- Episode terminates when all tiles are visited or the car goes too far off track

For more information, visit: [Gymnasium Car Racing](https://gymnasium.farama.org/environments/box2d/car_racing/)
