import torch
from NeuralNetwork import NeuralNetwork
from utilities import Utilities
import numpy as np

GAMMA = 0.99
EPSILON = 0.2
EPSILON_VALUE = 0.15
LAMBDA = 0.95
VALUE_COEFFICIENT = 1.0
ENTROPY_COEFFICIENT = 0.03
BASE = 0.9

class Policy:
    def __init__(self, neural_network):
        self.model = neural_network
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, eps=1e-5)

    def select_action(self, state_tensor):
        """
        Selects an action for data collection.
        Uses torch.distributions for sampling.
        Returns action, log_prob, and value as tensors.
        """
        with torch.no_grad():
            if self.model.is_discrete:
                # Discrete actions
                logits, value = self.model(state_tensor)
                probs = Utilities.softmax(logits)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action.item(), log_prob, value
            else:
                # Continuous actions
                action_mean, action_log_std, value = self.model(state_tensor)
                dist = torch.distributions.Normal(action_mean, action_log_std.exp())
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)  # Sum across action dimensions
                return action.cpu().numpy(), log_prob, value

    def evaluate_actions(self, states_tensor, actions_tensor):
        """
        Evaluates a batch of states and actions from the replay buffer during training.
        """
        if self.model.is_discrete:
            # Discrete actions
            logits, values = self.model(states_tensor)
            probs = Utilities.softmax(logits)
            dist = torch.distributions.Categorical(probs)
            
            log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy()
            
            return log_probs, values.squeeze(), entropy
        else:
            # Continuous actions
            action_mean, action_log_std, values = self.model(states_tensor)
            dist = torch.distributions.Normal(action_mean, action_log_std.exp())
            
            log_probs = dist.log_prob(actions_tensor).sum(dim=-1)  # Sum across action dimensions
            entropy = dist.entropy().sum(dim=-1)  # Sum across action dimensions
            
            return log_probs, values.squeeze(), entropy

    def compute_gae(self, rewards, values, next_values, dones):
        """
        Computes the Generalized Advantage Estimation (GAE) for a batch of experiences.
        
        If the advantage is > 0, the actions taken are better than average.
        
        If the advantage is < 0, the actions taken are worse than average.
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_advantage = delta
                # advantage = reward - predicted cumulative reward
            else:
                delta = rewards[t] + GAMMA * next_values[t] - values[t]
                # advantage = reward + current predicted cumulative reward - previous predicted cumulative reward
                last_advantage = delta + GAMMA * LAMBDA * last_advantage
                # advantage = reward + current predicted cumulative reward - previous predicted cumulative reward + discount factor * lambda * previous advantage
            advantages[t] = last_advantage
        return advantages
    
    def compute_losses(self, old_log_probs, new_log_probs, advantages, new_values, previous_values, returns, entropy):
        """
        Computes the PPO policy and value losses.
        """
        # Policy Loss (Clipped Surrogate Objective)
        ratio = torch.exp(new_log_probs - old_log_probs)
        first_term = ratio * advantages
        second_term = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages
        policy_loss = -torch.min(first_term, second_term).mean()
        
        # Value Loss (clipped value objective, analogous to policy clipping)
        # previous_values: values predicted by the old policy (no grad)
        # new_values: current value predictions
        V_new = new_values
        V_old = previous_values
        V_clip = V_old + torch.clamp(V_new - V_old, -EPSILON_VALUE, EPSILON_VALUE)
        loss_unclipped = (V_new - returns).pow(2)
        loss_clipped = (V_clip - returns).pow(2)
        value_loss = torch.max(loss_unclipped, loss_clipped).mean()
        
        # Total Loss
        total_loss = policy_loss + VALUE_COEFFICIENT * value_loss - ENTROPY_COEFFICIENT * entropy.mean()
        
        return total_loss, policy_loss, value_loss
