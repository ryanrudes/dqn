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

from utils import *
from agent import *
from config import *
from environment import *
logging.info("Started environment")

frame = 0
episode = 0
highscore = -np.inf
loss = 0
updates = 0
max_epsilon = epsilon
total_frames = epsilon_random_frames + epsilon_greedy_frames
epsilon_decay = (max_epsilon - min_epsilon) / epsilon_greedy_frames

start = time.time()
while True:
    state = env.reset()
    terminal = False
    accumulated_reward = 0
    for t in range(max_steps_per_episode):
        action = epsilon_random(epsilon, state)
        next_state, reward, terminal, info = env.step(action)
        memory.store((state, action, next_state, reward, terminal))
        loss = update(model, target, optimizer, lossfn)
        frame += 4
        updates += 1
        accumulated_reward += reward
        if frame > epsilon_random_frames:
            epsilon = max(min_epsilon, epsilon - epsilon_decay * 4)
        if updates % update_target_frequency == 0:
            target.set_weights(model.get_weights())
            logging.info("Updated the target network")
            model.save_weights(f'model/{env.spec.id}-{start}.h5')
            logging.info("Saved model weights")
        if terminal:
            break
        else:
            state = next_state
            if render:
                env.render()

    episode += 1
    highscore = max(highscore, accumulated_reward)
    remaining = (time.time() - start) / frame * max(0, total_frames - frame)
    logging.info("Reached terminal state in episode %d" % episode)
    logging.info("Frames: %d" % frame)
    logging.info("Updates: %d" % updates)
    logging.info("Reward: %d" % accumulated_reward)
    logging.info("High score: %d" % highscore)
    logging.info("Loss: %.6f" % loss)
    logging.info("Epsilon: %.6f" % epsilon)
    logging.info("Exploration duration remaining: %s" % datetime.timedelta(seconds = remaining))
