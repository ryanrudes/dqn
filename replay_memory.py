from random import sample, choices
from collections import deque
import numpy as np

class ReplayMemory():
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._queue = deque(maxlen = maxsize)
        self.size = 0

    def store(self, item):
        self._queue.append(item)
        self.size += 1

    def sample(self, n):
        if self.size >= n:
            transitions = sample(self._queue, k = n)
        else:
            transitions = choices(self._queue, k = n)

        states = np.array([transition[0] for transition in transitions])
        actions = np.array([transition[1] for transition in transitions])
        next_states = np.array([transition[2] for transition in transitions])
        rewards = np.array([transition[3] for transition in transitions])
        terminals = np.array([transition[4] for transition in transitions])
        return states, actions, next_states, rewards, terminals