import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from ReinforcementLearningLoop import ReinforcementLearningLoop

np.bool8 = np.bool_

if __name__ == "__main__":
    print("Starting PPO...")
    
    config = {
        "num_steps_per_epoch": 2048,
        "num_training_iterations": 500,
        "k_epochs": 4
    }
    
    rl_loop = ReinforcementLearningLoop()
    
    for i in range(config["num_training_iterations"]):
        print(f"--- Iteration {i+1}/{config['num_training_iterations']} ---")
        
        print("Collecting experiences...")
        rl_loop.collect_experiences(num_steps=config["num_steps_per_epoch"])
        
        print("Training...")
        rl_loop.train(k_epochs=config["k_epochs"])
        
    print("Training finished.")
