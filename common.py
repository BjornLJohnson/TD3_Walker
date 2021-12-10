import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def calculate_return(rollout, gamma):
    result = 0
    for i in range(len(rollout)):
        transition = rollout[i]
        result = result + gamma**i * transition['reward']

    return result


def update_target_model(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def visualize_rollout(env, actor):
    # Evaluate the final policy
    state = env.reset()
    done = False
    epi_return = 0
    while not done:
        s_tens = torch.tensor(state, dtype=torch.float)
        action = actor(s_tens).detach().numpy()
        next_state, r, done, _ = env.step(action)
        epi_return += r
        env.render()
        # time.sleep(0.05)
        state = next_state
    print("Total reward of visualized episode: {}".format(epi_return))


def collect_episode(env, actor=None, noise_cov=None, random=False):
    with torch.no_grad():
        action_dim = env.action_space.shape[0]
        action_min = torch.tensor(env.action_space.low)
        action_max = torch.tensor(env.action_space.high)
        if noise_cov is not None:
            m = torch.distributions.MultivariateNormal(torch.zeros((action_dim,)), noise_cov)
        transitions = []
        s = env.reset()
        done = False
        while not done:
            if random:
                a = env.action_space.sample()
            else:
                s_tens = torch.tensor(s, dtype=torch.float)
                a_tens = actor(s_tens)
                if noise_cov is not None:
                    a_tens += m.sample()
                torch.clamp(a_tens, min=action_min, max=action_max)
                a = a_tens.detach().numpy()
            s_prime, r, done, _ = env.step(a)
            transitions.append({
                'state': s,
                'action': a,
                'reward': r,
                'next_state': s_prime
            })
            s = s_prime
        return transitions


def fill_transition_buffer(env, buffer, num_steps, actor=None, noise_cov=None, random=False):
    steps = 0

    while steps < num_steps:
        transitions = collect_episode(env, actor, noise_cov, random)
        steps += len(transitions)
        buffer.add(transitions)

    return steps


class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        """
        A function to initialize the replay buffer.

        param: init_length : Initial number of transitions to collect
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        param: env : gym environment object
        """

        total_dim = 2*state_dim+action_dim+2

        self.state_slice = slice(0, state_dim)
        self.action_slice = slice(state_dim, state_dim+action_dim)
        self.reward_ind = state_dim + action_dim
        self.next_state_slice = slice(state_dim+action_dim+1, total_dim-1)
        self.done_ind = -1

        self.max_len = buffer_size
        self.memory = torch.zeros((buffer_size, total_dim))
        self.curr_ind = 0
        self.stored_transitions = 0

    def add(self, transitions):
        """
        A function to add a dictionary to the buffer
        param: exp : A dictionary consisting of state, action, reward , next state and done flag
        """
        for t in transitions:
            s = torch.tensor(t['state'], dtype=torch.float)
            a = torch.tensor(t['action'], dtype=torch.float)
            r = torch.tensor(t['reward'], dtype=torch.float)
            n = torch.tensor(t['next_state'], dtype=torch.float)
            nd = torch.tensor(t['not done'], dtype=torch.float)

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
        indices = np.random.choice(self.stored_transitions, N)

        samples = self.memory[indices]

        states = samples[:, self.state_slice]
        actions = samples[:, self.action_slice]
        rewards = samples[:, self.reward_ind].unsqueeze(dim=1)
        next_states = samples[:, self.next_state_slice]
        not_done = samples[:, self.done_ind].unsqueeze(dim=1)

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
