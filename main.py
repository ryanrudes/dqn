from replay_memory import ReplayMemory
from queue import deque
import tensorflow as tf
import numpy as np
import logging
import mahotas
import datetime
import time
import gym

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *

class FrameStack:
    def __init__(self, env):
        self.env = env
        self.spec = env.spec
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._obs_buffer = deque(maxlen = 4)

    def reset(self):
        for i in range(3):
            self._obs_buffer.append(np.zeros((84, 84, 1)))
        self._obs_buffer.append(self.preprocess(self.env.reset()))
        return self.observe()

    def preprocess(self, frame):
        # return np.expand_dims(mahotas.imresize(mahotas.colors.rgb2grey(frame), (84, 110)), axis = -1)[:, 13:-13] / 255.0
        return np.expand_dims(mahotas.imresize(mahotas.colors.rgb2grey(frame), (84, 84)), axis = -1) / 255.0
    
    def observe(self):
        return np.concatenate(self._obs_buffer, axis = -1)

    def step(self, action):
        total_reward = 0.0
        
        for i in range(4):
            observation, reward, terminal, info = self.env.step(action)
            self._obs_buffer.append(self.preprocess(observation))
            total_reward += reward
            if terminal:
                break

        return self.observe(), total_reward, terminal, info
    
    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        self.env.close()

def make():
    inputs = Input(shape = (84, 84, 4))
    x = Conv2D(32, 8, 4, activation = 'relu')(inputs)
    x = Conv2D(64, 4, 2, activation = 'relu')(x)
    x = Conv2D(64, 3, 1, activation = 'relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation = 'relu')(x)
    q = Dense(num_actions)(x)
    model = Model(inputs, q)
    return model

def epsilon_random(epsilon, state):
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)
    else:
        q = model.predict(np.expand_dims(state, axis = 0))[0]
        return np.argmax(q)
        
def update(memory, minibatch_size, model, target):
    states, actions, next_states, rewards, terminals = memory.sample(minibatch_size)
    pred_values = np.max(target(next_states), axis = 1)
    real_values = np.where(terminals, rewards, rewards + gamma * pred_values)
    with tf.GradientTape() as tape:
        selected_actions_one_hot = tf.one_hot(actions, num_actions)
        selected_action_values = tf.math.reduce_sum(model(states) * selected_actions_one_hot, axis = 1)
        loss = lossfn(real_values, selected_action_values)
    vars = model.trainable_variables
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))
    return loss.numpy()

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

max_steps_per_episode = 10000
max_replay_memory = 100000
num_actions = env.action_space.n
minibatch_size = 32
gamma = 0.99
update_target_frequency = 10000
frame = 0
episode = 0
highscore = -np.inf
loss = 0

epsilon = 1.0
max_epsilon = epsilon
epsilon_random_frames = 50000
epsilon_greedy_frames = 1000000
total_frames = epsilon_random_frames + epsilon_greedy_frames
min_epsilon = 0.1
epsilon_decay = (max_epsilon - min_epsilon) / epsilon_greedy_frames

memory = ReplayMemory(max_replay_memory)
model = make()
target = make()
target.set_weights(model.get_weights())

optimizer = Adam(0.00025, clipnorm = 1.0)
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