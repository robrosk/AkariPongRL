import numpy as np
from NeuralNetwork import NeuralNetwork
from Utilities import Utilities
import torch

GAMMA = 0.99
EPSILON = 0.2
LAMBDA = 0.95


"""
Policy class for PPO. This class is stateless.
"""
class Policy:
    def __init__(self, neural_network):
        self.model = neural_network
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00025)

    def select_action(self, state):
        """
        Selects an action for data collection.
        Runs the model and returns the action, its log probability under the
        current policy, and the critic's value estimate.
        """        
        logits, value = self.model(state)
        action_probs = Utilities.softmax(logits)
        log_probs = np.log(np.array(action_probs))
                
        action = np.random.choice(action_probs.shape[1], p=np.squeeze(action_probs))
        
        # Get the log probability of the specific action that was sampled.
        log_prob_action = log_probs[0, action]
        prob_action = action_probs[0, action]
                
        return action, prob_action, log_prob_action, np.squeeze(value)

    def evaluate_actions(self, states, actions):
        """
        Evaluates a batch of states and actions from the replay buffer during training.
        This is used in the optimization phase to calculate the PPO loss.
        """
        logits, values = self.model(states)
        action_probs = Utilities.softmax(logits)
        log_probs = np.log(action_probs)
        
        # Get the log probabilities of the specific actions that were actually taken,
        # according to the *current* policy.
        log_prob_actions = np.array([log_probs[i, action] for i, action in enumerate(actions)])
        
        # Calculate policy entropy to encourage exploration
        entropy = -np.sum(action_probs * log_probs, axis=1) 
        
        # Squeeze values to remove the singleton dimension (e.g., from (N, 1) to (N,)).
        return log_prob_actions, np.squeeze(values), entropy

    def compute_gae(self, rewards, values, next_values, dones):
        """
        Computes the Generalized Advantage Estimation (GAE) for a batch of experiences.
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_advantage = delta
            else:
                delta = rewards[t] + GAMMA * next_values[t] - values[t]
                last_advantage = delta + GAMMA * LAMBDA * last_advantage
            advantages[t] = last_advantage
        return advantages
    
    def ppo_loss(self, old_log_probs, new_log_probs, advantages, new_values, old_values, entropy):
        # TODO: Implement the PPO loss function
        ratio = torch.exp(new_log_probs - old_log_probs) # torch.exp() is used to avoid numerical instability - log_probs can be negative and 
        
        

