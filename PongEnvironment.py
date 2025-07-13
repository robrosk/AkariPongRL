import gym
from gym.wrappers import AtariPreprocessing, FrameStack

class PongEnvironment:
    def __init__(self, env_id: str, sticky_actions: bool = False):
        """
        Create a wrapped Atari environment:
        - AtariPreprocessing applies:
            • frame_skip = 4
            • grayscale + resize to 84×84
            • terminal_on_life_loss = False (default)
        - FrameStack stacks the last 4 frames into a single observation.

        Args:
            env_id (str): e.g. "PongNoFrameskip-v4", "BreakoutNoFrameskip-v4"
            sticky_actions (bool): whether to use sticky actions (default: False)

        Returns:
            gym.Env: the wrapped environment
        """
        # 1. Create the raw Atari env
        #    "frameskip=1" with NoFrameskip variant means we control skipping ourselves below.
        env = gym.make(env_id)

        # 2. AtariPreprocessing does:
        #    - frame_skip=4 (so every env.step(action) advances 4 frames)
        #    - max pooling over the last 2 frames
        #    - converting to grayscale, resizing to 84×84
        #    - optional terminal_on_life_loss (we leave default False)
        #    - optional sticky actions (only if you explicitly pass sticky_actions=True)
        env = AtariPreprocessing(env,
                                frame_skip=4,
                                terminal_on_life_loss=False,
                                screen_size=84,
                                grayscale_obs=True,)

        # 3. FrameStack stacks the last 4 observations along the channel dimension:
        #    after this, observation_space.shape == (4, 84, 84)
        env = FrameStack(env, num_stack=4)

        self.env = env
        
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
    