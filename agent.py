from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *

from replay_memory import ReplayMemory
from config import *

memory = ReplayMemory(max_replay_memory)
model = make(num_actions)
target = make(num_actions)
target.set_weights(model.get_weights())

optimizer = Adam(learning_rate, clipnorm = clipnorm)
lossfn = Huber()
