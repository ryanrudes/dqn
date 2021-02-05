from collections import deque
import numpy as np
import mahotas

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
        # Uncomment the below if using an environment other than Atari Breakout
        # In the standard preprocessing, the bottom and top 13 pixels are cropped out, but this crops the agent's paddle
        # out of the frame in Breakout, making it impossible to converge. For all other environments, you should uncomment
        # the line below to use the standard preprocessing algorithm, and comment out the alternative currently in use.
        
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
