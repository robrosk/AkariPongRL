import numpy as np
from ReinforcementLearningLoop import ReinforcementLearningLoop

if __name__ == "__main__":
    print("Starting PPO...")
    
    config = {
        "num_steps_per_epoch": 2048,
        "num_training_iterations": 500,
        "k_epochs": 10
    }
    
    rl_loop = ReinforcementLearningLoop()
    rl_loop.run(config)
