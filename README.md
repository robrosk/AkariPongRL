# PPO Implementation for Reinforcement Learning

A Proximal Policy Optimization (PPO) implementation for reinforcement learning, designed to work with both Atari environments and standard gymnasium environments.

## Features

- **PPO Algorithm**: Implements Proximal Policy Optimization with Generalized Advantage Estimation (GAE)
- **Actor-Critic Architecture**: Neural network with shared features and separate policy/value heads
- **Environment Flexibility**: Works with Atari environments (when available) and standard gymnasium environments
- **Cross-Platform**: Compatible with both Windows and Mac
- **Fallback Support**: Automatically falls back to standard environments if chosen environment is not available.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install (Both Mac and Windows)
```bash
pip install -r requirements.txt
```

### Basic Training
```bash
python main.py
```

The default configuration uses the Acrobat-v1 environment for training. You can modify the environment in `ReinforcementLearningLoop.py`.

## Project Structure

- `main.py` - Main training loop entry point
- `NeuralNetwork.py` - CNN architecture for actor-critic model
- `Policy.py` - PPO policy implementation with GAE
- `Environment.py` - Environment wrapper - supports image + vector based environments
- `ReinforcementLearningLoop.py` - Main RL training loop
- `utilities.py` - Utility functions for state processing

## Configuration

The training loop can be configured in `main.py`:
- `num_steps_per_epoch`: Number of steps to collect per training iteration
- `num_training_iterations`: Total number of training iterations
- `k_epochs`: Number of epochs per training iteration

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all packages are installed correctly
2. **Environment Not Found**: The system automatically falls back to standard environments

## How It Works

The PPO implementation:
1. Collects experiences using the current policy
2. Computes advantages using GAE
3. Updates the policy using multiple epochs of optimization
4. Repeats until convergence

## License

This project is open source and available under the MIT License.
