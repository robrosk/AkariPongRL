import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import numpy as np

class Environment:
    def __init__(self, env_id="CartPole-v1"):
        """
        Initialize environment wrapper that works with both Atari and standard environments.
        
        Args:
            env_id: Environment ID. For Atari, use IDs like 'ALE/Pong-v5' or 'PongNoFrameskip-v4'.
                   For standard environments, use IDs like 'CartPole-v1'.
        """
        self.env_id = env_id
        
        # Try to set up Atari environment first
        if self._is_atari_env(env_id):
            try:
                self._setup_atari_env(env_id)
                print(f"Successfully loaded Atari environment: {env_id}")
                return
            except Exception as e:
                print(f"Failed to load Atari environment {env_id}: {e}")
                print("Falling back to standard environment...")
        
        # Fall back to standard environment
        self._setup_standard_env(env_id)
        print(f"Using standard environment: {env_id}")
    
    def _is_atari_env(self, env_id):
        """Check if the environment ID is for an Atari game."""
        atari_keywords = ['ALE', 'Atari', 'Pong', 'Breakout', 'SpaceInvaders', 'Qbert', 'BeamRider']
        return any(keyword in env_id for keyword in atari_keywords)
    
    def _setup_atari_env(self, env_id):
        """Set up Atari environment with proper preprocessing."""
        # Create base environment
        self.env = gym.make(env_id, render_mode="rgb_array")
        
        # Apply Atari preprocessing
        self.env = AtariPreprocessing(
            self.env,
            frame_skip=4,
            grayscale_obs=True,
            scale_obs=True,
            terminal_on_life_loss=True
        )
        
        # Apply frame stacking
        self.env = FrameStackObservation(self.env, num_stack=4)
        
        # Store original action and observation spaces
        self._action_space = self.env.action_space
        self._observation_space = self.env.observation_space
    
    def _setup_standard_env(self, env_id):
        """Set up standard environment with wrapper for compatibility."""
        self.env = gym.make(env_id, render_mode="rgb_array")
        self.env = StandardEnvWrapper(self.env)
        
        # Store original action and observation spaces
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


class StandardEnvWrapper:
    """
    Wrapper for standard gymnasium environments to make them compatible with the CNN architecture.
    This converts simple state vectors into 4-channel 84x84 "images" that the neural network can process.
    """
    
    def __init__(self, env):
        self.env = env
        self.frame_stack = []
        self.max_frames = 4
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Convert observation to 84x84 "image" format
        processed_obs = self._process_observation(obs)
        
        # Add to frame stack
        self.frame_stack.append(processed_obs)
        if len(self.frame_stack) > self.max_frames:
            self.frame_stack.pop(0)
        
        # Pad frame stack if needed
        while len(self.frame_stack) < self.max_frames:
            self.frame_stack.insert(0, np.zeros((84, 84), dtype=np.float32))
        
        # Stack frames along channel dimension
        stacked_obs = np.stack(self.frame_stack, axis=0)  # Shape: (4, 84, 84)
        
        return stacked_obs, reward, terminated, truncated, info
    
    def reset(self):
        obs, info = self.env.reset()
        
        # Clear frame stack
        self.frame_stack = []
        
        # Process initial observation
        processed_obs = self._process_observation(obs)
        
        # Initialize frame stack with the same observation repeated
        for _ in range(self.max_frames):
            self.frame_stack.append(processed_obs)
        
        # Stack frames along channel dimension
        stacked_obs = np.stack(self.frame_stack, axis=0)  # Shape: (4, 84, 84)
        
        return stacked_obs, info
    
    def _process_observation(self, obs):
        """
        Convert a simple observation (e.g., CartPole's 4-vector) into an 84x84 "image".
        This is a simplified approach - in practice, you might want more sophisticated processing.
        """
        if isinstance(obs, (list, tuple)):
            obs = np.array(obs)
        
        # Normalize the observation values
        obs = obs.astype(np.float32)
        if obs.max() != obs.min():
            obs = (obs - obs.min()) / (obs.max() - obs.min())
        
        # Create a simple 84x84 representation
        # For CartPole, we'll create a simple visualization
        if len(obs) == 4:  # CartPole: [cart_pos, cart_vel, pole_angle, pole_vel]
            # Create a simple 84x84 representation
            img = np.zeros((84, 84), dtype=np.float32)
            
            # Cart position (horizontal)
            cart_x = int(41.5 + obs[0] * 20)  # Scale and center
            cart_x = max(0, min(83, cart_x))
            
            # Pole angle (vertical position)
            pole_y = int(41.5 + obs[2] * 20)  # Scale and center
            pole_y = max(0, min(83, pole_y))
            
            # Draw cart and pole
            img[pole_y, cart_x] = 1.0  # Pole tip
            img[41, cart_x] = 1.0      # Cart position
            
            return img
        else:
            # For other environments, create a simple pattern
            img = np.zeros((84, 84), dtype=np.float32)
            center = 42
            for i, val in enumerate(obs):
                if i < 84:
                    img[center, i] = val
            
            return img
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        # Return a space that matches our processed observations
        import gymnasium.spaces as spaces
        return spaces.Box(low=0, high=1, shape=(4, 84, 84), dtype=np.float32)
    
    @property
    def info(self):
        return self.env.info
    