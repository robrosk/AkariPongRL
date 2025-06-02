import gym
from gym.wrappers import AtariPreprocessing, FrameStack
import numpy as np

np.bool8 = np.bool_

print("hello world")

def make_atari_env(env_id: str, sticky_actions: bool = False):
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

    return env

if __name__ == "__main__":
    env = make_atari_env("PongNoFrameskip-v4")
    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)
    print(info)
    action = env.action_space.sample()  # random action
    print(env.action_space, action)
    print("Action shape:", action.shape if hasattr(action, 'shape') else type(action))
    print("Action space:", env.action_space)
    next_obs, reward, done, truncated, info = env.step(action)
    print("After one step:", next_obs.shape, reward, done)
    env.close()