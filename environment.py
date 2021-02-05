import gym
from config import ENV_ID
from wrapper import FrameStack

env = gym.make(ENV_ID)
env = FrameStack(env)
num_actions = env.action_space.n
