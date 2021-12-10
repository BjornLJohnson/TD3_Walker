import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def update_target_model(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def calculate_return(rollout, gamma):
    result = 0
    for i in range(len(rollout)):
        transition = rollout[i]
        result = result + gamma ** i * transition['reward']

    return result


class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        """
        A function to initialize the replay buffer.

        param: init_length : Initial number of transitions to collect
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        param: env : gym environment object
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        total_dim = 2*state_dim+action_dim+2

        self.state_slice = slice(0, state_dim)
        self.action_slice = slice(state_dim, state_dim+action_dim)
        self.reward_ind = state_dim + action_dim
        self.next_state_slice = slice(state_dim+action_dim+1, total_dim-1)
        self.done_ind = -1

        self.max_len = buffer_size
        self.memory = torch.zeros((buffer_size, total_dim)).to(self.device)
        self.curr_ind = 0
        self.stored_transitions = 0

    def add(self, transitions):
        """
        A function to add a dictionary to the buffer
        param: exp : A dictionary consisting of state, action, reward , next state and done flag
        """
        for t in transitions:
            s = torch.tensor(t['state'], dtype=torch.float).to(self.device)
            a = torch.tensor(t['action'], dtype=torch.float).to(self.device)
            r = torch.tensor(t['reward'], dtype=torch.float).to(self.device)
            n = torch.tensor(t['next_state'], dtype=torch.float).to(self.device)
            nd = torch.tensor(t['not done'], dtype=torch.float).to(self.device)

            self.add_one(s, a, r, n, nd)

    def add_one(self, s, a, r, n, nd):
        self.memory[self.curr_ind, self.state_slice] = torch.tensor(s)
        self.memory[self.curr_ind, self.action_slice] = torch.tensor(a)
        self.memory[self.curr_ind, self.reward_ind] = r
        self.memory[self.curr_ind, self.next_state_slice] = torch.tensor(n)
        self.memory[self.curr_ind, self.done_ind] = nd

        self.curr_ind = (self.curr_ind + 1) % self.max_len
        if self.stored_transitions < self.max_len:
            self.stored_transitions += 1

    def sample(self, N):
        """
        A function to sample N points from the buffer
        param: N : Number of samples to obtain from the buffer
        """
        indices = np.random.randint(0, self.stored_transitions, size=N)

        samples = self.memory[indices]

        states = samples[:, self.state_slice].to(self.device)
        actions = samples[:, self.action_slice].to(self.device)
        rewards = samples[:, self.reward_ind].unsqueeze(dim=1).to(self.device)
        next_states = samples[:, self.next_state_slice].to(self.device)
        not_done = samples[:, self.done_ind].unsqueeze(dim=1).to(self.device)

        return states, actions, rewards, next_states, not_done


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        """
        Initialize the network
        param: state_dim : Size of the state space
        param: action_dim: Size of the action space
        """
        super(Actor, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, action_dim)

        self.action_scale = torch.FloatTensor(
            (action_high - action_low) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_high + action_low) / 2.)

    def forward(self, x):
        """
        Define the forward pass
        param: state: The state of the environment
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x) * self.action_scale + self.action_bias

        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the critic
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        """
        super(Critic, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim + action_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 1)

    def forward(self, state, action):
        """
        Define the forward pass of the critic
        """
        x = torch.cat([state, action], axis=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

    def Q1(self, state, action):
        x = torch.cat([state, action], axis=1)

        q1 = self.fc1_1(x)
        q1 = F.relu(q1)
        q1 = self.fc2_1(q1)
        q1 = F.relu(q1)
        q1 = self.fc3_1(q1)

        return q1


class Twin_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the critic
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        """
        super(Twin_Critic, self).__init__()

        self.fc1_1 = torch.nn.Linear(state_dim + action_dim, 256)
        self.fc2_1 = torch.nn.Linear(256, 256)
        self.fc3_1 = torch.nn.Linear(256, 1)

        self.fc1_2 = torch.nn.Linear(state_dim + action_dim, 256)
        self.fc2_2 = torch.nn.Linear(256, 256)
        self.fc3_2 = torch.nn.Linear(256, 1)

    def forward(self, state, action):
        """
        Define the forward pass of the critic
        """
        x = torch.cat([state, action], axis=1)

        q1 = self.fc1_1(x)
        q1 = F.relu(q1)
        q1 = self.fc2_1(q1)
        q1 = F.relu(q1)
        q1 = self.fc3_1(q1)

        q2 = self.fc1_2(x)
        q2 = F.relu(q2)
        q2 = self.fc2_2(q2)
        q2 = F.relu(q2)
        q2 = self.fc3_2(q2)

        return q1, q2

    def Q1(self, state, action):
        x = torch.cat([state, action], axis=1)

        q1 = self.fc1_1(x)
        q1 = F.relu(q1)
        q1 = self.fc2_1(q1)
        q1 = F.relu(q1)
        q1 = self.fc3_1(q1)

        return q1
