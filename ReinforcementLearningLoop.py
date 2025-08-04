import numpy as np
from PongEnvironment import PongEnvironment
from Policy import Policy
from NeuralNetwork import NeuralNetwork
from Utilities import Utilities
import torch

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
        # action is an int
        next_state_np, reward, done, truncated, info = self.environment.step(action)
        next_state_tensor = Utilities.normalize_expand_transpose_state(next_state_np)
        self.state = next_state_tensor
        self.done = done or truncated
        self.info = info
        return next_state_tensor, reward, done, truncated, info
    
    def collect_experiences(self, num_steps=2048):
        """
        Collects a batch of experience by running the policy in the environment.
        """
        if self.state is None:
            self.state = self.reset()

        for _ in range(num_steps):
            # The policy now works with tensors and returns tensor data
            action, log_prob, value = self.policy.select_action(self.state)

            next_state, reward, done, truncated, info = self.step(action)
            
            # Store the tensor data - `action` is an int.
            self.replay_buffer.append((self.state, action, reward, next_state, done, log_prob, value))
            
            self.state = next_state
            if done or truncated:
                self.state = self.reset()
                
            if _ > 0 and _ % 100 == 0:
                print(f"Collected {_} experiences")
        
    def train(self, k_epochs=4):
        # Unpack experiences. This is a list of tuples of tensors and other data.
        states, actions, rewards, next_states, dones, old_log_probs, old_values = zip(*self.replay_buffer)

        # Convert lists of tensors and numbers into single batched tensors.
        # stack states and next_states, which are already (1, C, H, W), into (N, C, H, W)
        states_tensor = torch.cat(states)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.cat(next_states)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        old_log_probs_tensor = torch.stack(old_log_probs).detach()
        old_values_tensor = torch.stack(old_values).squeeze().detach()

        # Get the value of the last next_state for GAE calculation
        with torch.no_grad():
            _, last_value = self.policy.model(next_states_tensor[-1].unsqueeze(0))

        # Calculate advantages using GAE.
        advantages = self.policy.compute_gae(rewards_tensor, old_values_tensor, torch.cat((old_values_tensor[1:], last_value.flatten())), dones_tensor)
        returns = (advantages + old_values_tensor).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(k_epochs):
            new_log_probs, new_values, entropy = self.policy.evaluate_actions(states_tensor, actions_tensor)
            
            total_loss, policy_loss, value_loss = self.policy.compute_losses(old_log_probs_tensor, new_log_probs, advantages, new_values, returns, entropy)
            
            self.policy.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 0.5)
            self.policy.optimizer.step()

        # Clear the replay buffer for the next collection phase.
        self.replay_buffer = []

        print(f"Training complete. Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")
