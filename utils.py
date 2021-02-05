import numpy as np
import tensorflow as tf

from environment import *
from config import *

def epsilon_random(epsilon, state):
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)
    else:
        q = model.predict(np.expand_dims(state, axis = 0))[0]
        return np.argmax(q)
        
def update(model, target, optimizer, lossfn):
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
