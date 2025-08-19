import gymnasium as gym
import numpy as np

class Environment:
    def __init__(self, env_id="Acrobot-v1", render_mode="rgb_array"):
        """
        Initialize environment wrapper.
        
        Args:
            env_id: Environment ID (e.g., 'CartPole-v1', 'PongNoFrameskip-v4', 'Acrobot-v1')
            render_mode: Rendering mode - "rgb_array" for training, "human" for visualization
        """
        self.env_id = env_id
        self.render_mode = render_mode
        
        # Simple setup for all environments - neural network will handle architecture selection
        self._setup_env(env_id)
        print(f"Loaded environment: {env_id}")
    
    def _setup_env(self, env_id):
        """Set up environment with minimal wrapper."""
        # Create environment directly - neural network will automatically detect observation type
        base_env = gym.make(env_id, render_mode=self.render_mode)
        self.env = base_env
        
        # Store action and observation spaces
        self._action_space = self.env.action_space
        self._observation_space = self.env.observation_space
    
    def get_env(self):
        return self.env
        
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        return self.env.reset()
    
    def close(self):
        self.env.close()
        
    def get_action_space(self):
        return self.env.action_space
    
    def get_observation_space(self):
        return self.env.observation_space
    
    def get_info(self):
        return self.env.info