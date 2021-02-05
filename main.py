from utils import *
from agent import *
from config import *
from model import make
from wrapper import FrameStack

import tensorflow as tf
import numpy as np
import logging
import mahotas
import datetime
import time
import gym

start = time.time()
logging.basicConfig(level    = logging.DEBUG,
                    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt  = '%m-%d %H:%M',
                    filename = f'logs/{start}.log',
                    filemode = 'w')
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
rootLogger.addHandler(consoleHandler)

env = gym.make("BreakoutDeterministic-v4")
env = FrameStack(env)
logging.info("Started environment")

num_actions = env.action_space.n
frame = 0
episode = 0
highscore = -np.inf
loss = 0
max_epsilon = epsilon
total_frames = epsilon_random_frames + epsilon_greedy_frames
epsilon_decay = (max_epsilon - min_epsilon) / epsilon_greedy_frames

start = time.time()
while frame < total_frames:
    state = env.reset()
    terminal = False
    accumulated_reward = 0
    for t in range(max_steps_per_episode):
        action = epsilon_random(epsilon, state, num_actions)
        next_state, reward, terminal, info = env.step(action)
        if render:
          env.render()
        frame += 1
        accumulated_reward += reward
        memory.store((state, action, next_state, reward, terminal))
        loss = update(memory, minibatch_size, gamma, model, target, optimizer, lossfn)
        if frame > epsilon_random_frames:
            epsilon = max(min_epsilon, epsilon - epsilon_decay)
        if frame % update_target_frequency == 0:
            target.set_weights(model.get_weights())
            logging.info("Updated the target network")
            model.save_weights(f'model/{env.spec.id}-{start}.h5')
            logging.info("Saved model weights")
        if terminal:
            break
        else:
            state = next_state

    episode += 1
    highscore = max(highscore, accumulated_reward)
    remaining = (time.time() - start) / frame * max(0, epsilon_greedy_frames - frame)
    logging.info("Reached terminal state in episode %d" % episode)
    logging.info("Frames: %d" % frame)
    logging.info("Reward: %d" % accumulated_reward)
    logging.info("High score: %d" % highscore)
    logging.info("Loss: %.6f" % loss)
    logging.info("Epsilon: %.6f" % epsilon)
    logging.info("Time remaining: %s" % datetime.timedelta(seconds = remaining))
