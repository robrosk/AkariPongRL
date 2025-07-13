import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gym
from gym.wrappers import AtariPreprocessing, FrameStack
import numpy as np
from AtariPolicy import AtariCNN
from ReinforcementLearningLoop import ReinforcementLearningLoop

np.bool8 = np.bool_

if __name__ == "__main__":
    rl_loop = ReinforcementLearningLoop()
    state = rl_loop.reset()
    
    # convert to float32 and normalize to [0, 1]
    state = np.array(state).astype(np.float32) / 255.0
    
    state = np.expand_dims(state, axis=0) # shape : (1, 4, 84, 84)
    
    print(state.shape)
    
    # state = np.transpose(state, (0, 2, 3, 1)) # shape: (1, 84, 84, 4)
    
    # # Instantiate model
    # model = AtariCNN(num_actions=rl_loop.environment.get_action_space().n)
    
    # logits, value = model(state)
    
    # rl_loop.run(max_steps=1000)