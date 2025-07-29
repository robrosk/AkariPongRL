import numpy as np
from NeuralNetwork import NeuralNetwork
from Utilities import Utilities

"""
Policy class for PPO
"""
class Policy:
    def __init__(self, neural_network):
        self.model = neural_network
        self.actor_prediction = None
        self.log_probs = None
        self.critic_prediction = None
        self.gamma = 0.99
    
    def select_action(self, state, training=True):
        action = None
        
        action_probs = self.get_action_probabilities(state, training)
        log_probs = self.compute_log_probs(action_probs, training)
        
        action = np.random.choice(len(action_probs), p=np.squeeze(action_probs))
        
        return action
    
    def get_action_probabilities(self, state, training):
        # Return full probability distribution
        logits, value = self.model(state)
        
        if training: 
            self.critic_prediction = value
        
        action_probs = Utilities.softmax(logits)
        
        return action_probs
    
    def advantange(self, reward, next_critic_value_prediction):
        return reward + self.gamma * next_critic_value_prediction - self.critic_prediction
    
    def compute_log_probs(self, action_probs, training):
        # For PPO training - compute log probabilities of taken actions
        log_probs = np.log(action_probs)
        
        if training: 
            self.log_probs = log_probs
        
        return log_probs
    
    def compute_policy_gradient_estimator(self, log_probs, action, advantage):
        return log_probs[action] * advantage 
        
    def clipped_surrogate_objective(self, log_probs, reward):
        pass
    

    
    