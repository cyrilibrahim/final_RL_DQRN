from unityagents import UnityEnvironment

from DQN import DQN
from UnityEnvWrapper import UnityEnvWrapper

ENV_PATH = "C:/Users/Bureau/Desktop/RL_DQN_FinalProject/Environments/Unity_environments/POMDP/pomdp"
PATH_SAVE = "save/intersection_env/"


env = UnityEnvironment(file_name=ENV_PATH)
unity_env_wrapper = UnityEnvWrapper(env, action_space = 4, train_mode=True)

dqn_train = DQN(env=unity_env_wrapper, env_type="Unity",nb_action=4, save_path=PATH_SAVE, update_target=3)
dqn_train.train_loop()