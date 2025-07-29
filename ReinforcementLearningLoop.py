import numpy as np
from PongEnvironment import PongEnvironment
from Policy import Policy
from NeuralNetwork import NeuralNetwork
from Utilities import Utilities

class ReinforcementLearningLoop:
    def __init__(self, env_id="PongNoFrameskip-v4"):
        self.environment = PongEnvironment(env_id)
        self.policy = Policy(NeuralNetwork(num_actions=self.environment.get_action_space().n))
        self.replay_buffer = []
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
    
    def get_next_critic_value(self, next_state):
        # Get the critic value for the next state
        _, critic_value = self.policy.model(next_state)
        return critic_value

    def run(self, max_steps=1000):
        current_state = self.reset()
        step_count = 0
        while not self.done and step_count < max_steps:
            action = self.choose_action(current_state)
            next_state, reward, done, truncated, info = self.step(action)   
            self.replay_buffer.append([current_state, action, reward, next_state, done])  
            current_state = next_state
            
            next_critic_value = self.get_next_critic_value(next_state)
            advantage = self.policy.advantange(reward, next_critic_value)
            self.policy.compute_policy_gradient_estimator(action, advantage)
            
            # Placeholder: add your RL logic here
            
            step_count += 1
            
    
        self.environment.close()
        
    def train(self):
        pass
        
    def choose_action(self, state):
        return self.policy.select_action(state, training=False)
