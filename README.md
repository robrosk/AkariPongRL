# AkariPongRL - PPO Implementation for Reinforcement Learning

A Proximal Policy Optimization (PPO) implementation for reinforcement learning, designed to work with both Atari environments and standard gymnasium environments.

## Features

- **PPO Algorithm**: Implements Proximal Policy Optimization with Generalized Advantage Estimation (GAE)
- **Actor-Critic Architecture**: Neural network with shared features and separate policy/value heads
- **Environment Flexibility**: Works with Atari environments (when available) and standard gymnasium environments
- **Cross-Platform**: Compatible with both Windows and Mac
- **Fallback Support**: Automatically falls back to standard environments if Atari is unavailable

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install (Both Mac and Windows)
```bash
pip install -r requirements.txt
```

### Manual Install
If you encounter issues with the requirements file, install packages individually:

```bash
# Core dependencies
pip install torch numpy gymnasium gymnasium-notices

# Environment support
pip install ale-py AutoROM AutoROM.accept-rom-license

# Image processing
pip install pillow

# Optional: OpenCV (if needed)
pip install opencv-python  # Windows
pip install opencv-python-headless  # Mac (if opencv-python fails)
```

## Usage

### Basic Training
```bash
python main.py
```

The default configuration uses the CartPole-v1 environment for training. You can modify the environment in `ReinforcementLearningLoop.py`.

### Testing Components
To test individual components without full training:
```bash
python test_simple.py
```

## Project Structure

- `main.py` - Main training loop entry point
- `NeuralNetwork.py` - CNN architecture for actor-critic model
- `Policy.py` - PPO policy implementation with GAE
- `Environment.py` - Environment wrapper with fallback support
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
2. **OpenCV Issues on Mac**: Use `opencv-python-headless` instead of `opencv-python`
3. **Environment Not Found**: The system automatically falls back to standard environments

### Platform-Specific Notes

- **Mac**: May need to use `opencv-python-headless` for OpenCV
- **Windows**: Generally works with standard package installations
- **Both**: The unified requirements.txt should work on both platforms

## How It Works

The PPO implementation:
1. Collects experiences using the current policy
2. Computes advantages using GAE
3. Updates the policy using multiple epochs of optimization
4. Repeats until convergence

The environment wrapper automatically handles:
- Atari environment setup (when available)
- Fallback to standard environments with simulated 4-channel 84x84 input
- Frame stacking and preprocessing

## License

This project is open source and available under the MIT License.
