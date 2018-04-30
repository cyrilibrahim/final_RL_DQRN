
import torch
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.functional as F
import _pickle as cPickle
import matplotlib.pyplot as plt
import math

# Custom functions
from Buffer import Buffer, Episode_Buffer
from RDQN_net import RDQN_net
from GymScreenProcessing import GymScreenProcessing
from utils import img_to_tensor
from StateTransition import Transition
from StackFrame import StackFrame

class RDQN(object):

    # Initialize Buffer, Networks, global variables
    # and configure CUDA
    def __init__(self, env, buffer_size = 10000, active_cuda = True, nb_episodes = 2000, max_steps = 3500,
                 discount_factor = 0.995,epsilon_greedy_end = 0.01, epsilon_greedy_start = 0.1, batch_size = 128, update_target = 10, env_type="Unity",
                 train = True, save_episode = 800, skip_frame = 4, stack_size = 4, nb_episodes_decay = 100,
                 save_path = "gym_cartpole", nb_action = 2, lr = 0.002, weight_decay = 1e-6, update_plot = 10, rgb=False,
                 seq_len = 8, nb_samples_episodes = 4):

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
        self.save_path = save_path
        self.nb_episodes_decay = nb_episodes_decay
        self.nb_action = nb_action
        self.lr = lr
        self.weight_decay = weight_decay
        self.buffer_size = buffer_size
        self.update_plot = update_plot
        self.nb_channel = 3 if rgb else 1
        self.epsilon_greedy_start  = epsilon_greedy_start
        self.epsilon_greedy_end  = epsilon_greedy_end
        self.seq_len = seq_len
        self.episode_iterator = 0
        self.nb_samples_episodes = nb_samples_episodes
        # Log to see improvment
        self.log_cumulative_reward = []
        self.log_loss = []

        #################### PSEUDO CODE STEPS ############################

        # Initialize replay memory D
        self.buffer = Episode_Buffer(self.buffer_size, self.seq_len)

        # Initialize Q policy network and Q target network
        self.Q_policy_net = RDQN_net(self.nb_action)
        self.Q_target_net = RDQN_net(self.nb_action)

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

            self.buffer.new_episode(self.episode_iterator)

            cumulative_reward = 0


            self.env.reset()
            state = self.get_screen()

            print("Episode " + str(self.episode_iterator) )

            hx = None
            cx = None

            # Initialize sequence s1 and preprocess (We take difference between two next frame)
            for t in range(0, self.max_steps):


                if(t % self.skip_frame == 0):
                    # Select epsilon greedy action
                    action, hx, cx = self.select_action(Variable(state, volatile = True), hx, cx)
                    # Process the action to the environment
                    env_action = self.get_env_action(action)

                    _, reward, done, _ = self.env.step(env_action)

                    cumulative_reward += reward

                    reward = self.Tensor([reward])

                    next_state = self.get_screen()

                    if not done:
                        not_done_mask = self.ByteTensor(1).fill_(1)
                    else:
                        next_state = None
                        not_done_mask = self.ByteTensor(1).fill_(0)
                        #reward = self.Tensor([-1])

                    self.buffer.push(state, action, next_state, reward,not_done_mask,self.episode_iterator)

                    self.learn()

                    state = next_state

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
        if(self.buffer.hasAtLeast(self.nb_samples_episodes)):

            samples, nb_episodes = self.buffer.sample(self.nb_samples_episodes)

            # At least 1 sampled episode
            if(nb_episodes > 0):

                # Here batches and sequence are mixed like that:
                #  episode 1 t_1
                #  episode 2 t_1
                #  episode.. t_1
                #  episode n t_1
                #  episode 1 t_m
                #  episode 2 t_m
                #  episode.. t_m
                #  episode n t_m
                [batch_state,batch_action,batch_reward, batch_next_state, not_done_batch] = Transition(*zip(*samples))
                batch_state = Variable(torch.cat(batch_state, dim = 0))
                batch_action = Variable(torch.cat(batch_action))
                batch_reward = Variable(torch.cat(batch_reward))
                #batch_next_state = Variable(torch.cat(batch_next_state, dim = 0))
                not_done_batch = self.ByteTensor(torch.cat(not_done_batch))

                non_final_next_states = Variable(torch.cat([s if s is not None else torch.zeros(1, 1, 84, 84).type(self.Tensor) for s in batch_next_state]),volatile=True)
                Q_s_t_a, (_, _) = self.Q_policy_net(batch_state, batch_size = nb_episodes, seq_length=self.seq_len)
                Q_s_t_a = Q_s_t_a.gather(1, batch_action)


                Q_s_next_t_a_result, (_, _) = self.Q_target_net(non_final_next_states, batch_size=nb_episodes, seq_length= self.seq_len)
                Q_s_next_t_a = Q_s_next_t_a_result.max(1)[0]
                Q_s_next_t_a[ 1 - not_done_batch] = 0

                # Target Q_s_t_a value (like supervised learning )
                target_state_value = (Q_s_next_t_a * self.discount_factor) + batch_reward
                target_state_value.detach_()

                target_state_value = Variable(target_state_value.data).unsqueeze_(1)

                assert Q_s_t_a.shape == target_state_value.shape

                loss = F.smooth_l1_loss(Q_s_t_a, target_state_value)

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()

                self.log_loss.append(loss.data[0])


                for param in self.Q_policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

    def select_action(self, state, hx, cx):
        # Greedy action
        if(np.random.uniform()  > self.epsilon_greedy):
            Q_policy_values,(hx, cx) = self.Q_policy_net.forward(state, hx = hx, cx = cx)
            action = Q_policy_values.data.max(1)[1].view(1, 1)
            return action, hx, cx
        # Random
        else:
            return self.LongTensor([[random.randrange(self.nb_action)]]), hx, cx

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
            return action.cpu().numpy()[0,0]

    def get_screen(self):
        if (self.env_type == "Unity"):
            return img_to_tensor(self.env.get_screen())
        elif(self.env_type == "Gridworld"):
            return img_to_tensor(np.expand_dims(self.env.renderEnv(), axis=3))
            #return self.env.renderEnv()
        else:
            # Gym
            return self.gym_screen_processing.get_screen()


    #################################################### PLOT SPECIFIC FUNCTIONS #######################################
    def save_plot(self):
        plt.plot(self.log_cumulative_reward)
        plt.title("DRQN on " + self.save_path)
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative reward")
        plt.savefig("../save/"+self.save_path+"_cumulative_rewards.png")
        plt.clf()
        plt.plot(self.log_loss[100:])
        plt.title("DRQN on " + self.save_path)
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.savefig("../save/"+self.save_path+"_loss.png")
        plt.clf()

import gym

PATH_SAVE = "cartpole"

env = gym.make('CartPole-v0').unwrapped
dqn_train = RDQN(env=env, env_type="Gym", nb_action = 2, skip_frame=1, buffer_size=500, save_path="cartpole", epsilon_greedy_start=0.2, epsilon_greedy_end=0.05, nb_samples_episodes=32, nb_episodes=40000,save_episode=1500)

dqn_train.train_loop()