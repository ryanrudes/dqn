from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *

from replay_memory import ReplayMemory
from queue import deque
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

params = os.environ

max_steps_per_episode = params["MAX_STEPS_PER_EPISODE"]
max_replay_memory = params["MAX_REPLAY_MEMORY"]
num_actions = env.action_space.n
minibatch_size = params["MINIBATCH_SIZE"]
gamma = params["GAMMA"]
update_target_frequency = params["UPDATE_TARGET_FREQUENCY"]
frame = 0
episode = 0
highscore = -np.inf
loss = 0

epsilon = params["EPSILON"]
max_epsilon = epsilon
epsilon_random_frames = params["EPSILON_RANDOM_FRAMES"]
epsilon_greedy_frames = params["EPSILON_GREEDY_FRAMES"]
total_frames = epsilon_random_frames + epsilon_greedy_frames
min_epsilon = params["MIN_EPSILON"]
epsilon_decay = (max_epsilon - min_epsilon) / epsilon_greedy_frames

learning_rate = params["LEARNING_RATE"]
clipnorm = params["CLIPNORM"]

memory = ReplayMemory(max_replay_memory)
model = make()
target = make()
target.set_weights(model.get_weights())

optimizer = Adam(learning_rate, clipnorm = clipnorm)
lossfn = Huber()

start = time.time()
while frame < total_frames:
    state = env.reset()
    terminal = False
    accumulated_reward = 0
    for t in range(max_steps_per_episode):
        action = epsilon_random(epsilon, state)
        next_state, reward, terminal, info = env.step(action)
        # env.render()
        frame += 1
        accumulated_reward += reward
        memory.store((state, action, next_state, reward, terminal))
        state = next_state
        if frame > epsilon_random_frames:
            epsilon = max(min_epsilon, epsilon - epsilon_decay)
        if frame % 32 == 0:
            loss = update(memory, minibatch_size, model, target)
        if frame % update_target_frequency == 0:
            target.set_weights(model.get_weights())
            logging.info("Updated the target network")
            model.save_weights(f'model/{env.spec.id}-{start}.h5')
            logging.info("Saved model weights")
        if terminal:
            break

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
