from collections import deque

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
