import gym
from DQN import DQN

PATH_SAVE = "cartpole"

env = gym.make('CartPole-v0').unwrapped
dqn_train = DQN(env=env, env_type="Gym", nb_action = 2, skip_frame=1)

dqn_train.train_loop()