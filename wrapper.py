from collections import deque
from gym import Wrapper
import numpy as np
import mahotas

class FrameStack(Wrapper):
    def __init__(self, env):
        super(FrameStack, self).__init__(env)
        self._obs_buffer = deque(maxlen = 4)

    def reset(self):
        for i in range(3):
            self._obs_buffer.append(np.zeros((84, 84)))
        self._obs_buffer.append(self.preprocess(self.env.reset()))
        return self.observe()

    def preprocess(self, frame):
        # Uncomment the below if using an environment other than Atari Breakout
        # In the standard preprocessing, the bottom and top 13 pixels are cropped out, but this crops the agent's paddle
        # out of the frame in Breakout, making it impossible to converge. For all other environments, you should uncomment
        # the line below to use the standard preprocessing algorithm, and comment out the alternative currently in use.
        
        # return mahotas.imresize(mahotas.colors.rgb2grey(frame), (84, 110))[:, 13:-13] / 255.0
        return mahotas.imresize(mahotas.colors.rgb2grey(frame), (84, 84)) / 255.0
    
    def observe(self):
        return np.stack(self._obs_buffer, axis = -1)

    def step(self, action):
        total_reward = 0.0
        
        for i in range(4):
            observation, reward, terminal, info = self.env.step(action)
            self._obs_buffer.append(self.preprocess(observation))
            total_reward += reward
            if terminal:
                break

        return self.observe(), total_reward, terminal, info
