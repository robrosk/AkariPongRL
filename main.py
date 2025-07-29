import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gym
from gym.wrappers import AtariPreprocessing, FrameStack
import numpy as np
from NeuralNetwork import NeuralNetwork
from ReinforcementLearningLoop import ReinforcementLearningLoop
from Utilities import Utilities

np.bool8 = np.bool_

if __name__ == "__main__":
    print("Starting PPO with Atari Pong...")
    
    num_steps = int(input("Enter the number of steps to run: "))
    
    rl_loop = ReinforcementLearningLoop()
    rl_loop.run(max_steps=num_steps)