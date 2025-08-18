import numpy as np
from Environment import Environment
from Policy import Policy
from NeuralNetwork import NeuralNetwork
from utilities import Utilities
import torch

class ReinforcementLearningLoop:
    def __init__(self, env_id="Acrobot-v1", render_mode="rgb_array"):
        self.env_id = env_id
        self.render_mode = render_mode
        self.environment = Environment(env_id, render_mode=render_mode)
        self.policy = Policy(NeuralNetwork(action_space=self.environment.get_action_space()))
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
        
        if self.policy.model.is_discrete:
            # Discrete actions are integers
            actions_tensor = torch.tensor(actions, dtype=torch.int64)
        else:
            # Continuous actions are numpy arrays
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.cat(next_states)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        old_log_probs_tensor = torch.stack(old_log_probs).detach()
        old_values_tensor = torch.stack(old_values).squeeze().detach()

        # Get the value of the last next_state for GAE calculation
        with torch.no_grad():
            if self.policy.model.is_discrete:
                _, last_value = self.policy.model(next_states_tensor[-1].unsqueeze(0))
            else:
                _, _, last_value = self.policy.model(next_states_tensor[-1].unsqueeze(0))

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
        print(f"Entropy Loss. {entropy.mean()}")

    def run(self, config):
        """
        Run the complete training loop using the provided configuration.
        Each iteration collects experiences and then trains the policy.
        
        Args:
            config: Dictionary containing training parameters:
                - num_steps_per_epoch: Number of steps to collect per iteration
                - num_training_iterations: Total number of training iterations
                - k_epochs: Number of optimization epochs per training step
        """
        print(f"Starting PPO training for {config['num_training_iterations']} iterations...")
        
        for i in range(config['num_training_iterations']):
            print(f"--- Iteration {i+1}/{config['num_training_iterations']} ---")
            
            print("Collecting experiences...")
            self.collect_experiences(num_steps=config['num_steps_per_epoch'])
            
            print("Training...")
            self.train(k_epochs=config['k_epochs'])
            
        print("Training finished.")

    def sample_trained_policy(self, num_episodes=5, render_mode="human"):
        """
        Sample from the trained policy to see how it performs.
        
        Args:
            num_episodes: Number of episodes to run
            render_mode: Rendering mode for evaluation ("human" for visual, "rgb_array" for headless)
        """
        print(f"Sampling {num_episodes} episodes from trained policy...")
        
        # Create a new environment instance with the specified render mode for evaluation
        eval_env = Environment(self.env_id, render_mode=render_mode)
        
        total_rewards = []
        
        for episode in range(num_episodes):
            state, _ = eval_env.reset()
            state_tensor = Utilities.normalize_expand_transpose_state(state)
            episode_reward = 0
            step_count = 0
            
            print(f"\n--- Episode {episode + 1} ---")
            
            while True:
                # Get action from trained policy (no exploration noise)
                with torch.no_grad():
                    if self.policy.model.is_discrete:
                        logits, _ = self.policy.model(state_tensor)
                        action_probs = torch.softmax(logits, dim=-1)
                        action = torch.argmax(action_probs).item()
                    else:
                        action_mean, _, _ = self.policy.model(state_tensor)
                        action = action_mean.cpu().numpy()
                
                # Take step in environment
                next_state, reward, done, truncated, info = eval_env.step(action)
                next_state_tensor = Utilities.normalize_expand_transpose_state(next_state)
                episode_reward += reward
                step_count += 1
                
                # Print step info
                if step_count % 50 == 0:  # Print every 50 steps
                    print(f"Step {step_count}: Action={action}, Reward={reward:.3f}, Total={episode_reward:.3f}")
                
                if done or truncated:
                    break
                    
                state_tensor = next_state_tensor
            
            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1} finished in {step_count} steps with total reward: {episode_reward:.3f}")
        
        # Close the evaluation environment
        eval_env.close()
        
        avg_reward = np.mean(total_rewards)
        print(f"\n=== Sampling Complete ===")
        print(f"Average reward across {num_episodes} episodes: {avg_reward:.3f}")
        print(f"Best episode reward: {max(total_rewards):.3f}")
        print(f"Worst episode reward: {min(total_rewards):.3f}")
        
        return total_rewards
