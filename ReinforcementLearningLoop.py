import numpy as np
from PongEnvironment import PongEnvironment
from NeuralNetwork import NeuralNetwork
from Utilities import Utilities

class ReinforcementLearningLoop:
    def __init__(self, env_id="PongNoFrameskip-v4"):
        self.environment = PongEnvironment(env_id)
        self.model = NeuralNetwork(num_actions=self.environment.get_action_space().n)
        self.state = None
        self.done = False
        self.info = None

    def reset(self):
        self.state, self.info = self.environment.reset()
        self.done = False
        return Utilities.normalize_expand_transpose_state(self.state)

    def step(self, action):
        next_state, reward, done, truncated, info = self.environment.step(action)
        next_state = Utilities.normalize_expand_transpose_state(next_state)
        self.state = next_state
        self.done = done or truncated
        self.info = info
        return next_state, reward, done, truncated, info

    def run(self, max_steps=1000):
        self.reset()
        step_count = 0
        while not self.done and step_count < max_steps:
            actions = self.environment.get_action_space()
            # Placeholder: select a random action
            action = actions.sample()
            next_state, reward, done, truncated, info = self.step(action)
            # Placeholder: add your RL logic here
            step_count += 1
            
    
        self.environment.close()
