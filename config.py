import os
from dotenv import load_dotenv

load_dotenv()
config = os.environ

max_steps_per_episode = int(config["MAX_STEPS_PER_EPISODE"])
max_replay_memory = int(config["MAX_REPLAY_MEMORY"])
minibatch_size = int(config["MINIBATCH_SIZE"])
gamma = float(config["GAMMA"])
update_target_frequency = int(config["UPDATE_TARGET_FREQUENCY"])
epsilon = float(config["EPSILON"])
epsilon_random_frames = int(config["EPSILON_RANDOM_FRAMES"])
epsilon_greedy_frames = int(config["EPSILON_GREEDY_FRAMES"])
min_epsilon = float(config["MIN_EPSILON"])
learning_rate = float(config["LEARNING_RATE"])
clipnorm = float(config["CLIPNORM"])
render = bool(config["RENDER"])
