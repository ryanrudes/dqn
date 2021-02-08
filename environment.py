import gym
from config import env_id, seed
from wrapper import FrameStack

env = gym.make(env_id)
env.seed(seed)
env = FrameStack(env)
num_actions = env.action_space.n
