import gym
from config import env_id
from wrapper import FrameStack

env = gym.make(env_id)
env = FrameStack(env)
num_actions = env.action_space.n
