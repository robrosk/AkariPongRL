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
    rl_loop = ReinforcementLearningLoop()
    state = rl_loop.reset()
    
    # convert to float32 and normalize to [0, 1]
    state = Utilities.normalize_expand_transpose_state(state)
    
    print(state.shape)
        
    # # Instantiate model
    model = NeuralNetwork(num_actions=rl_loop.environment.get_action_space().n)
    
    logits, value = model(state)
    
    print(logits.shape)
    print(value.shape)
    
    # rl_loop.run(max_steps=1000)