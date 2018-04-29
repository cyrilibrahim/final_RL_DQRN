
import torch
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.functional as F
import _pickle as cPickle
import matplotlib.pyplot as plt
import math

# Custom functions
from Buffer import Buffer
from DQN_net import DQN_net
from GymScreenProcessing import GymScreenProcessing
from utils import img_to_tensor
from StateTransition import Transition
from StackFrame import StackFrame

class DQN(object):

    # Initialize Buffer, Networks, global variables
    # and configure CUDA
    def __init__(self, env, buffer_size = 10000, active_cuda = True, nb_episodes = 2000, max_steps = 3500,
                 discount_factor = 0.995,epsilon_greedy_end = 0.01, epsilon_greedy_start = 0.1, batch_size = 128, update_target = 10, env_type="Unity",
                 train = True, save_episode = 800, skip_frame = 4, stack_size = 4, nb_episodes_decay = 100,
                 save_path = "gym_cartpole", nb_action = 2, lr = 0.002, weight_decay = 1e-6, update_plot = 10):

        # Global parameters
        self.env = env
        self.nb_episodes = nb_episodes
        self.max_steps = max_steps
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.update_target = update_target
        self.env_type = env_type
        self.save_episode = save_episode
        self.skip_frame = skip_frame
        self.stack_size = stack_size
        self.stack_frame = StackFrame(self.stack_size)
        self.save_path = save_path
        self.nb_episodes_decay = nb_episodes_decay
        self.nb_action = nb_action
        self.lr = lr
        self.weight_decay = weight_decay
        self.buffer_size = buffer_size
        self.update_plot = update_plot

        self.epsilon_greedy_start  = epsilon_greedy_start
        self.epsilon_greedy_end  = epsilon_greedy_end

        self.episode_iterator = 0

        # Log to see improvment
        self.log_cumulative_reward = []

        #################### PSEUDO CODE STEPS ############################

        # Initialize replay memory D
        self.buffer = Buffer(self.buffer_size)

        # Initialize Q policy network and Q target network
        self.Q_policy_net = DQN_net(self.nb_action)
        self.Q_target_net = DQN_net(self.nb_action)

        # Copy policy weight to target weight
        self.Q_target_net.load_state_dict(self.Q_policy_net.state_dict())

        ############### PYTORCH SPECIFIC INITIALIZATION ###################

        # Adapt to cuda
        self.active_cuda = active_cuda
        if active_cuda:
            self.Q_policy_net.cuda()
            self.Q_target_net.cuda()

        self.FloatTensor = torch.cuda.FloatTensor if active_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if active_cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if active_cuda else torch.ByteTensor
        self.Tensor = self.FloatTensor

        # Use RMSProp DeepMind's parameters
        self.optimizer = torch.optim.RMSprop(self.Q_policy_net.parameters(),lr=self.lr, weight_decay=self.weight_decay)
        # Init class to process each fram (just call gym_screen_processing.get_screen() to have the processed screen)
        self.gym_screen_processing = GymScreenProcessing(self.env, active_cuda)



    def train_loop(self, retrain = False):

        self.update_epislon_greedy()

        print("Train")
        if(self.episode_iterator >= self.nb_episodes):
            if(not retrain):
                "Please pass retrain parameter if you want to retrain the model. Warning: You will loose everything if " \
                "you choose to retrain your network."
                return
        for current_episode in range(self.episode_iterator, self.nb_episodes):

            cumulative_reward = 0

            self.env.reset()
            state = self.get_screen()

            # Init first stack frame
            self.stack_frame.reset_stack()
            # Init 4 first frames
            for i in range(0, self.stack_frame.max_frames):
                self.stack_frame.add_frame(state)

            old_stack = torch.cat(self.stack_frame.get_frames(), dim = 1)


            print("Episode " + str(self.episode_iterator) )

            # Initialize sequence s1 and preprocess (We take difference between two next frame)
            for t in range(0, self.max_steps):
                if(t % self.skip_frame == 0):

                    # Select epsilon greedy action
                    action = self.select_action(Variable(old_stack, volatile = True))
                    # Process the action to the environment
                    env_action = self.get_env_action(action)

                    _, reward, done, _ = self.env.step(env_action)

                    cumulative_reward += reward

                    reward = self.Tensor([reward])

                    next_state = self.get_screen()

                    self.stack_frame.add_frame(next_state)

                    if not done:
                        next_stack = torch.cat(self.stack_frame.get_frames(), dim = 1)
                        not_done_mask = self.ByteTensor(1).fill_(1)
                    else:
                        next_stack = None
                        not_done_mask = self.ByteTensor(1).fill_(0)
                        reward = self.Tensor([-1])

                    self.buffer.push(old_stack, action, next_stack, reward,not_done_mask)

                    self.learn()

                    old_stack = next_stack

                    if done:
                        print("Done")
                        break
                else:
                    self.env.step(env_action)

            print("Episode cumulative reward: ")
            print(cumulative_reward)

            if self.episode_iterator % self.save_episode  == 0 and self.episode_iterator != 0:
                print("Save parameters checkpoint:")
                self.save()
                print("End saving")

            if self.episode_iterator % self.update_plot == 0:
                self.save_plot()


            self.episode_iterator += 1
            self.update_epislon_greedy()

            if current_episode % self.update_target  == 0:
                self.Q_target_net.load_state_dict(self.Q_policy_net.state_dict())


            self.log_cumulative_reward.append(cumulative_reward)



    ################################################ LEARNING FUNCTIONS ################################################

    # Gradient descent on (yi - Q_target(state))^2
    def learn(self):
        if(self.buffer.hasAtLeast(self.batch_size)):

            [batch_state,batch_action,batch_reward, batch_next_state, not_done_batch] = Transition(*zip(*self.buffer.sample(self.batch_size)))
            batch_state = Variable(torch.cat(batch_state, dim = 0))
            batch_action = Variable(torch.cat(batch_action))
            batch_reward = Variable(torch.cat(batch_reward))
            not_done_batch = self.ByteTensor(torch.cat(not_done_batch))
            non_final_next_states = Variable(torch.cat([s for s in batch_next_state if s is not None]),volatile=True)

            Q_s_t_a = self.Q_policy_net(batch_state).gather(1, batch_action)

            Q_s_next_t_a = Variable(torch.zeros(self.batch_size).type(self.Tensor))
            Q_s_next_t_a[not_done_batch] = self.Q_target_net(non_final_next_states).max(1)[0]

            # Target Q_s_t_a value (like supervised learning )
            target_state_value = (Q_s_next_t_a * self.discount_factor) + batch_reward
            target_state_value = Variable(target_state_value.data)

            loss = F.smooth_l1_loss(Q_s_t_a, target_state_value)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()

            for param in self.Q_policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    def select_action(self, state):
        # Greedy action
        if(np.random.uniform()  > self.epsilon_greedy):
            return self.Q_policy_net.forward(state).data.max(1)[1].view(1, 1)
        # Random
        else:
            return self.LongTensor([[random.randrange(self.nb_action)]])

    # Every episodes
    def update_epislon_greedy(self):
        self.epsilon_greedy = self.epsilon_greedy_end + ( self.epsilon_greedy_start -  self.epsilon_greedy_end) * math.exp(-1. * self.episode_iterator / self.nb_episodes_decay)


    ##################################################### SAVE/LOAD FUNCTIONS ##########################################

    def  save(self):
        temp_env = self.env
        temp_gym_screen_proc = self.gym_screen_processing
        temp_buffer = self.buffer
        self.env = None
        self.gym_screen_processing = None
        self.buffer = None

        with open(self.save_path, 'wb') as output:
            cPickle.dump(self, output)
        self.env = temp_env
        self.gym_screen_processing = temp_gym_screen_proc
        self.buffer = temp_buffer


    def load_env(self, env):
        self.env = env

    def init_buffer(self):
        self.buffer = Buffer(self.buffer_size)


    ##################################################### ENVIRONMENT TYPE SPECIFIC ####################################

    def get_env_action(self, action):
        if(self.env_type == "Unity"):
            return action.cpu().numpy()
        else:
            return action[0, 0]

    def get_screen(self):
        if (self.env_type == "Unity"):
            return img_to_tensor(self.env.get_screen())
        else:
            # Gym
            return self.gym_screen_processing.get_screen()


    #################################################### PLOT SPECIFIC FUNCTIONS #######################################
    def save_plot(self):
        plt.plot(self.log_cumulative_reward)
        plt.title("DQN on " + self.save_path)
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative reward")
        plt.savefig("save/"+self.save_path+"_cumulative_rewards.png")

#with open('gym_cartpole.pkl', 'rb') as input:
    #dqn_train = cPickle.load(input)
    #env = UnityEnvironment(file_name="C:/Users/Bureau/Desktop/RL_DQN_FinalProject/POMDP/pomdp")
    #dqn_train.load_env(env)
    #dqn_train.init_buffer(10000)
    #dqn_train.max_steps  = 10000

    #dqn_train.train_mode  = False
 # dqn_train.nb_episodes = 200000
#    print(dqn_train.episode_iterator)
    #dqn_train.train_loop()

    #print(dqn_train.log_cumulative_reward)

    #reward_every_50 = np.mean(np.array(dqn_train.log_cumulative_reward).reshape(-1, 1), axis=1)
    #plt.plot(reward_every_50)
    #plt.title("DQN")
    #plt.xlabel("Episodes (multiply by 177)")
    #plt.ylabel("Cumulative reward")
    #plt.show()