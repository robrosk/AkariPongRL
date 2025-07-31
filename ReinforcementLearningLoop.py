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

    def collect_experiences(self, num_steps=2048):
        """
        Collects a batch of experience by running the policy in the environment.
        """
        if self.state is None:
            self.state = self.reset()

        for _ in range(num_steps):
            action, prob_action, log_prob_action, critic_value = self.policy.select_action(self.state)

            next_state, reward, done, truncated, info = self.step(action)
            
            # Store the experience from the "old" policy. Note we store the log_prob.
            self.replay_buffer.append((self.state, action, reward, next_state, done, prob_action, log_prob_action, critic_value))
            
            self.state = next_state
            if done or truncated:
                self.state = self.reset()
                
            if _ % 100 == 0:
                print(f"Collected {_} experiences")
        
    def train(self):
        # Unpack the collected experiences. Note that we stored the log_prob from the "old" policy.
        states, actions, rewards, next_states, dones, action_probs, log_probs, values = map(np.array, zip(*self.replay_buffer))

        # TODO: Loop for K epochs to optimize the policy.

        # Clear the replay buffer for the next collection phase.
        self.replay_buffer = []