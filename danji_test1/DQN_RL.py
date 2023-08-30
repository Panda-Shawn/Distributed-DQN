"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import os
import time as t

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('Maze-v0')
env = env.unwrapped
# N_ACTIONS = env.action_space.n
# N_STATES = env.observation_space.shape[0]
N_ACTIONS = 4
N_STATES = 1
# if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
ENV_A_SHAPE = 0


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, type='master'):
        self.eval_net, self.target_net = Net(), Net()

        # for target updating
        self.learn_step_counter = 0
        # for storing memory
        self.memory_counter = 0
        # initialize memory
        if type == 'master':
            self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        if type == 'worker':
            self.memory = np.zeros((MEMORY_CAPACITY // 10, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        save_root = 'model_saved'
        self.eval_path = os.path.join(save_root, 'eval.pth')
        self.target_path = os.path.join(save_root, 'target.pth')

        self.loss = []

    def choose_action(self, x, ty='train'):
        # x = torch.unsqueeze(torch.FloatTensor(x), 0)
        x = torch.unsqueeze(torch.FloatTensor([x]), 0)
        if ty == 'eval':
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(
                ENV_A_SHAPE)
        if ty == 'train':
            # input only one sample
            if np.random.uniform() < EPSILON:   # greedy
                actions_value = self.eval_net.forward(x)
                action = torch.max(actions_value, 1)[1].data.numpy()
                action = action[0] if ENV_A_SHAPE == 0 else action.reshape(
                    ENV_A_SHAPE)  # return the argmax index
            else:   # random
                action = np.random.randint(0, N_ACTIONS)
                action = action if ENV_A_SHAPE == 0 else action.reshape(
                    ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample_transition(self):
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        return b_memory

    def update_gradient(self, fc1_grad, out_grad):
        self.eval_net.fc1.weight.grad.data = fc1_grad
        self.eval_net.out.weight.grad.data = out_grad
        self.optimizer.step()

    def extract_model(self):
        fc1 = self.eval_net.fc1.weight.data
        out = self.eval_net.out.weight.data
        return fc1, out

    def rollout(self):
        is_memory = False
        while not is_memory:
            s = env.reset()
            ep_r = 0
            done = False
            while not done:
                env.render()
                a = self.choose_action(s)

                s_, r, done, info = env.step(a)

                self.store_transition(s, a, r, s_)
                ep_r += r
                if self.memory_counter > MEMORY_CAPACITY:
                    is_memory = True
                s = s_

    def compute_grad(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        # detach from graph, don't backpropagate
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * \
            q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        # detach from graph, don't backpropagate
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * \
            q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss.append(loss.detach().numpy())

    def save_model(self):
        if not os.path.exists('model_saved/'):
            os.mkdir('model_saved')
        # time模块示例
        timet = t.strftime("a, %d %b %Y %H:%M:%S +0000", t.gmtime())
        torch.save(self.eval_net.state_dict(), self.eval_path + timet + '.pth')
        torch.save(self.target_net.state_dict(),
                   self.target_path + timet + '.pth')

    def load_model(self, path):
        self.eval_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))


if __name__ == "__main__":
    dqn = DQN()
    f = open('epr.txt', 'wt', encoding='utf-8')
    print('\nCollecting experience...')
    for i_episode in range(2000):
        s = env.reset()
        ep_r = 0
        while True:
            env.render()
            # print("current state: ", s)
            a = dqn.choose_action(s)
            # print(a)
            # take action
            s_, r, done, info = env.step(a)

            # # modify the reward
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # r = r1 + r2

            dqn.store_transition(s, a, r, s_)

            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    con = 'Ep: {}'.format(i_episode) + \
                        '\tEp_r: {}'.format(round(ep_r, 2))
                    print(con)
                    f.write(con + '\n')

            if done:
                break
            s = s_
        if (i_episode + 1) % 100 == 0:
            dqn.save_model()
    f.close()
    np.save('loss.npy', np.array(dqn.loss))
