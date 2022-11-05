import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import working_memory_env
from minigrid.wrappers import RGBImgObsWrapper


def make_env(env_key, seed=None, render_mode=None):
    # env = working_memory_env.DMTSGridEnv(grid_size=4, max_steps=300, render_mode=render_mode)
    env = gym.make(env_key, render_mode=render_mode)
    # env = FlattenObservation(env)
    env.reset(seed=seed)
    return env
