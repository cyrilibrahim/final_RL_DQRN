


class UnityEnvWrapper(object):
    def __init__(self, env, train_mode = True, action_space = 4):
        self.env = env
        self.brain_name = self.env.brain_names[0]
        self.action_space = action_space
        self.train_mode = train_mode
        self.env_info = None

    def reset(self):
        self.env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]

    def get_screen(self):
        return self.env_info.observations[0]


    def step(self, action):
        self.env_info = self.env.step(action)[self.brain_name]
        reward = self.env_info.rewards[0]
        done = self.env_info.local_done[0]

        return None, reward, done, None